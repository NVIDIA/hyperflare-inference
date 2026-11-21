# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate captured cache writes, rotary arithmetic, and actual bounded proposal probabilities."""

from pathlib import Path
import math
import torch
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.reference.glm_moe_dsa import CheckpointReader
from ignition.reference.glm_moe_dsa_cache import encode_latent, quantize_index
from ignition.reference.glm_moe_dsa_bounds import rms_norm_interval, rounded_bf16_interval
from .glm53_full_draft_reference import check, gamma, UNIT


def proposal_reference(logits, top_p):
    scores = logits.double().flatten()
    ids = scores.argsort(descending=True, stable=True)[:64]
    q = scores[ids].softmax(0)
    top_p = float(torch.tensor(top_p, dtype=torch.float32))
    before = q.cumsum(0) - q
    cutoff = q[before < top_p].min()
    keep = q >= cutoff
    q = q * keep
    q = q / q.sum()
    expected = torch.zeros_like(scores).scatter(0, ids, q)
    maximum = float((scores[ids] - scores[ids].max()).abs().max())
    exponential = math.expm1(gamma(2) * maximum) + 2.5 * UNIT
    relative = (1 + UNIT) ** 8 * (1 + exponential) / (1 - exponential) * (
        (1 + gamma(64)) / (1 - gamma(64))
    ) ** 3 - 1
    return expected, relative


@torch.inference_mode()
def verify_storage_and_probabilities(capture, snapshot, consumed):
    torch.set_num_threads(8)
    reader = CheckpointReader(
        inspect_indexed_checkpoint(Path(snapshot)), torch.device("cpu"), torch.float64
    )
    norm = reader.floating("model.layers.78.self_attn.kv_a_layernorm.weight")
    native = [
        torch.load(capture / f"consumed-{consumed}-r{rank}.pt", mmap=True) for rank in range(8)
    ]
    result = {"consumed": consumed, "ranks": [], "failures": []}
    for rank, payload in enumerate(native):
        reports = []
        frequencies = payload["rotary_frequencies"].double()
        frequency_ref = 8000000 ** (-torch.arange(32, dtype=torch.float64) / 32)
        # CUDA 13.2 documents at most four ulps from correctly rounded powf.
        # Add half an ulp to exact and one ulp for the reciprocal.
        frequency_relative = (4.5 * UNIT + UNIT) / (1 - 4.5 * UNIT)
        frequency_check = check(
            frequencies,
            (frequency_ref * (1 - frequency_relative), frequency_ref * (1 + frequency_relative)),
        )
        for step in range(2):
            data = payload["cases"]["repeat-0"][step]
            report = {"frequencies": frequency_check}
            report["reference_latent_normalization"] = check(
                data["reference_latent_norm"], rms_norm_interval(data["latent_norm_input"], norm, 1e-6)
            )
            expected = encode_latent(data["reference_latent_norm"], data["reference_rotary_key"])
            record = data["cache"][consumed - 1 + step][None]
            report["scale_bytes_exact"] = {
                "passed": torch.equal(expected[:, 512:528], record[:, 512:528])
            }
            scales = expected[:, 512:528].contiguous().view(torch.float32)
            inverse = scales.reciprocal()
            low, high = inverse, inverse
            for _ in range(2):
                low = torch.nextafter(low, torch.full_like(low, -torch.inf))
                high = torch.nextafter(high, torch.full_like(high, torch.inf))
            source = data["reference_latent_norm"].float().reshape(-1, 4, 128)
            endpoints = (
                torch.stack((source * low[:, :, None], source * high[:, :, None]))
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .double()
            )
            actual = (
                record[:, :512].contiguous().view(torch.float8_e4m3fn).double().reshape_as(source)
            )
            report["latent_encoding"] = check(actual, (endpoints.amin(0), endpoints.amax(0)))
            report["addressed_rotary_tail_exact"] = {
                "passed": torch.equal(record[:, 528:], expected[:, 528:])
                and int(data["write_slots"][0]) == consumed - 1 + step
            }
            if data["reused_selection"]:
                report["retained_selection_skips_index_writer"] = {"passed": "stored_index_bytes" not in data}
            else:
                index_bytes, index_scales = quantize_index(data["index_key_rotation_output"], query=False)
                report["planar_index_record_exact"] = {
                    "passed": torch.equal(data["stored_index_bytes"], index_bytes)
                    and torch.equal(data["stored_index_scales"], index_scales)
                }
            report["replicated_cache"] = {
                "passed": torch.equal(data["cache"], native[0]["cases"]["repeat-0"][step]["cache"])
            }
            if step:
                report["prefix_cache_unchanged"] = {
                    "passed": torch.equal(
                        data["cache"][:-1], payload["cases"]["repeat-0"][0]["cache"]
                    )
                }
            angle = (data["positions"][:, None].float() * payload["rotary_frequencies"]).double()
            trig_ref = torch.stack((angle.cos(), angle.sin()), -1)
            trig_error = (data["rotary_values"].double() - trig_ref).abs()
            # The SM120 compiled _angles kernel inlines CUDA 13.4.3 libdevice
            # __nv_sinf/__nv_cosf, whose full-range accuracy is two ulps
            # from correctly rounded FP32. Include half an ulp to exact.
            report["trigonometry"] = check(
                data["rotary_values"],
                (trig_ref - 2.5 * UNIT * trig_ref.abs(), trig_ref + 2.5 * UNIT * trig_ref.abs()),
            )
            report["trigonometry"]["max_absolute_error"] = float(trig_error.max())
            rotations = ["query_rotation", "key_rotation"]
            if not data["reused_selection"]:
                rotations.extend(("index_query_rotation", "index_key_rotation"))
            for name in rotations:
                source = data[name + "_input"].double()
                actual = data[name + "_output"]
                pairs = source[..., -64:].reshape(*source.shape[:-1], 32, 2)
                cosine, sine = data["rotary_values"].double().unbind(-1)
                if source.ndim == 3:
                    cosine, sine = cosine[:, None], sine[:, None]
                a, b = pairs.unbind(-1)
                exact = torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2)
                magnitude = torch.stack(
                    (
                        a.abs() * cosine.abs() + b.abs() * sine.abs(),
                        a.abs() * sine.abs() + b.abs() * cosine.abs(),
                    ),
                    -1,
                ).flatten(-2)
                measurement = "reference_index_key_rotation" if name == "index_key_rotation" else name
                report[measurement] = check(
                    actual[..., -64:],
                    rounded_bf16_interval(
                        exact - gamma(3) * magnitude, exact + gamma(3) * magnitude
                    ),
                )
                report[measurement]["unchanged_unrotated"] = torch.equal(
                    actual[..., :-64], data[name + "_input"][..., :-64]
                )
            for top_p, record in data["proposal_probabilities"].items():
                expected, relative = proposal_reference(data["logits"], float(top_p))
                probability = record["probabilities"].double().flatten()
                report["proposal_" + top_p] = check(
                    probability, (expected * (1 - relative), expected * (1 + relative))
                )
                report["proposal_" + top_p].update(
                    support_exact=torch.equal(probability > 0, expected > 0),
                    valid=bool(record["valid"].all()),
                    sampled_supported=bool(expected[int(record["sampled"][0])] > 0),
                    ranks_equal=torch.equal(
                        record["probabilities"],
                        native[0]["cases"]["repeat-0"][step]["proposal_probabilities"][top_p][
                            "probabilities"
                        ],
                    ),
                    repeat_exact=all(
                        torch.equal(
                            record["probabilities"],
                            payload["cases"][f"repeat-{r}"][step]["proposal_probabilities"][top_p][
                                "probabilities"
                            ],
                        )
                        for r in (1, 2)
                    ),
                    tv=float((probability - expected).abs().sum() / 2),
                    relative_allowance=relative,
                )
                wrong, _ = proposal_reference(data["logits"].double() * 2, float(top_p))
                report["proposal_" + top_p]["wrong_temperature_changes_law"] = bool(
                    ((wrong - expected).abs() > relative * expected).any()
                )
                wrong = expected.clone()
                donor = int((expected > 0).nonzero()[-1])
                recipient = int((expected == 0).nonzero()[0])
                moved = wrong[donor] / 2
                wrong[donor] -= moved
                wrong[recipient] += moved
                report["proposal_" + top_p]["wrong_support_rejected"] = bool(
                    ((wrong - expected).abs() > relative * expected).any()
                )
            reports.append(report)
            for name, measurement in report.items():
                failed = measurement.get("out_of_bound", 0) > 0 or any(
                    measurement.get(key, True) is False
                    for key in (
                        "passed",
                        "unchanged_unrotated",
                        "support_exact",
                        "valid",
                        "sampled_supported",
                        "repeat_exact",
                        "ranks_equal",
                        "wrong_support_rejected",
                    )
                )
                if failed:
                    result["failures"].append((rank, step, name, measurement))
        result["ranks"].append(reports)
    print("consumed", consumed, "failures", result["failures"], flush=True)
    print("trigonometry measurements", result["ranks"][0][0]["trigonometry"], flush=True)
    assert not result["failures"], result
    return result
