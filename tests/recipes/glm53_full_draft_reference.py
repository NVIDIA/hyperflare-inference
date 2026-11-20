# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check prepared draft stages on fixed operands using independent FP64 arithmetic and intervals."""

from pathlib import Path
import torch
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.reference.glm_moe_dsa import CheckpointReader
from ignition.reference.glm_moe_dsa_attention import fp32_dot_allowance
from ignition.reference.glm_moe_dsa_attention_bounds import sparse_attention_allowance
from ignition.reference.glm_moe_dsa_bounds import (
    projection_interval,
    rms_norm_interval,
    swiglu_interval,
    sigmoid_relative_allowance,
    rounded_bf16_interval,
)

UNIT = 2**-23


def gamma(count):
    return count * UNIT / (1 - count * UNIT)


def check(value, bounds):
    value = value.double()
    lower, upper = bounds
    center = (lower + upper) / 2
    outside = (value < lower) | (value > upper) | ~torch.isfinite(value)
    negative = value * 1.2
    return {
        "cells": value.numel(),
        "out_of_bound": int(outside.sum()),
        "max_excess": float(torch.maximum(lower - value, value - upper).clamp_min(0).max()),
        "relative_l2_from_center": float((value - center).norm() / center.norm().clamp_min(1e-30)),
        "relative_l2_radius": float(((upper - lower) / 2).norm() / center.norm().clamp_min(1e-30)),
        "wrong_20_percent_rejected": bool(((negative < lower) | (negative > upper)).any()),
    }


def dot_check(value, exact, magnitude, count):
    allowance = fp32_dot_allowance(exact, magnitude, count, bf16_output=False)
    return check(value, rounded_bf16_interval(exact - allowance, exact + allowance))


@torch.inference_mode()
def verify_arithmetic(capture, snapshot, fixtures, consumed):
    torch.set_num_threads(8)
    reader = CheckpointReader(
        inspect_indexed_checkpoint(Path(snapshot)), torch.device("cpu"), torch.float64
    )
    fixture = torch.load(Path(fixtures) / "long.pt", mmap=True)
    prefix = "model.layers.78."

    def weight(name):
        return reader.floating(prefix + name).double()

    values = [
        torch.load(capture / f"consumed-{consumed}-r{rank}.pt", mmap=True) for rank in range(8)
    ]
    joint = weight("self_attn.kv_b_proj.weight").reshape(64, 448, 512)
    output = weight("self_attn.o_proj.weight")
    joined_weight = weight("eh_proj.weight")
    input_weight = torch.cat(
        [
            weight("self_attn." + name + ".weight")
            for name in ("q_a_proj", "kv_a_proj_with_mqa", "indexer.wk")
        ]
    )
    q_weight = weight("self_attn.q_b_proj.weight")
    index_weight = weight("self_attn.indexer.wq_b.weight")
    results = []
    for step in range(2):
        reports = [{} for _ in range(8)]
        first = values[0]["cases"]["repeat-0"][step]
        token = (
            int(fixture["token_ids"][consumed]) if step == 0 else int(values[0]["recursive_token"])
        )
        embedded = reader.floating("model.embed_tokens.weight", token).double()[None]
        previous = (
            fixture["hidden"][consumed - 1 : consumed].bfloat16()
            if step == 0
            else values[0]["cases"]["repeat-0"][0]["normalized"]
        )
        reports[0]["embedding_normalization"] = check(
            first["joined_input"][:, :6144],
            rms_norm_interval(embedded, weight("enorm.weight"), 1e-5),
        )
        reports[0]["preceding_normalization"] = check(
            first["joined_input"][:, 6144:],
            rms_norm_interval(previous, weight("hnorm.weight"), 1e-5),
        )
        reports[0]["joined_projection"] = check(
            first["joined"],
            projection_interval(first["joined_input"], joined_weight, bf16_output=True),
        )
        reports[0]["input_normalization"] = check(
            first["input_layernorm_output"],
            rms_norm_interval(first["joined"], weight("input_layernorm.weight"), 1e-5),
        )
        reports[0]["input_projection"] = check(
            first["input_proj"],
            projection_interval(first["input_proj_input"], input_weight, bf16_output=True),
        )
        reports[0]["query_normalization"] = check(
            first["q_b_proj_input"],
            rms_norm_interval(
                first["input_proj"][:, :2048], weight("self_attn.q_a_layernorm.weight"), 1e-6
            ),
        )
        if step == 0:
            reports[0]["head_weights_projection"] = check(
                first["indexer.weights_proj"],
                projection_interval(
                    first["indexer.weights_proj_input"],
                    weight("self_attn.indexer.weights_proj.weight"),
                    bf16_output=False,
                ),
            )
            reports[0]["index_query_projection"] = check(
                first["index_query_projection"],
                projection_interval(
                    first["index_query_projection_input"], index_weight, bf16_output=True
                ),
            )
        reports[0]["attention_residual_exact"] = {
            "passed": torch.equal(
                first["post_attention_layernorm_input"],
                (first["joined"].float() + first["attention"].float()).bfloat16(),
            )
        }
        reports[0]["post_attention_normalization"] = check(
            first["post_attention_layernorm_output"],
            rms_norm_interval(
                first["post_attention_layernorm_input"],
                weight("post_attention_layernorm.weight"),
                1e-5,
            ),
        )
        reports[0]["router_projection"] = check(
            first["router_logits"],
            projection_interval(
                first["router_logits_input"], weight("mlp.gate.weight"), bf16_output=False
            ),
        )
        logits = first["router_logits"].double()
        scores = logits.sigmoid()
        bias = weight("mlp.gate.e_score_correction_bias")
        corrected = scores + bias
        ids = first["expert_ids"].long()
        mask = torch.ones_like(corrected, dtype=torch.bool).scatter_(1, ids, False)
        score_radius = scores * sigmoid_relative_allowance(logits) + UNIT * (
            scores.abs() + bias.abs()
        )
        minimum = (corrected + score_radius).gather(1, ids).min()
        maximum = (corrected - score_radius).masked_fill(~mask, -torch.inf).max()
        reports[0]["router_cutoff"] = {
            "passed": bool(maximum <= minimum),
            "selected_ids": ids.tolist(),
            "minimum_selected": float(corrected.gather(1, ids).min()),
            "maximum_omitted": float(corrected.masked_fill(~mask, -torch.inf).max()),
        }
        chosen = scores.gather(1, ids)
        expected = chosen / chosen.sum(1, keepdim=True) * 2.5
        rel = 2 * sigmoid_relative_allowance(logits).max() + gamma(12)
        reports[0]["route_weights"] = check(
            first["expert_weights"], (expected * (1 - rel), expected * (1 + rel))
        )
        for rank in range(8):
            data = values[rank]["cases"]["repeat-0"][step]
            report = reports[rank]
            replicated = (
                "joined_input",
                "joined",
                "input_layernorm_input",
                "input_layernorm_output",
                "input_proj",
                "post_attention_layernorm_input",
                "post_attention_layernorm_output",
                "router_logits",
                "expert_ids",
                "expert_weights",
                "normalized",
                "logits",
                "selected",
            )
            report["replicated_stage_values"] = {
                "passed": all(torch.equal(data[name], first[name]) for name in replicated)
            }
            report["repeat_exact"] = {
                "passed": all(
                    all(
                        torch.equal(
                            data[name], values[rank]["cases"][f"repeat-{repeat}"][step][name]
                        )
                        for name in data
                        if torch.is_tensor(data[name])
                    )
                    for repeat in (1, 2)
                )
            }
            report["position"] = {"passed": int(data["positions"][0]) == consumed - 1 + step}
            report["retained_selection"] = {
                "passed": torch.equal(
                    data["retained_selection"], values[rank]["cases"]["repeat-0"][0]["selected"]
                )
            }
            q_matrix = q_weight[rank * 2048 : (rank + 1) * 2048]
            report["query_projection"] = check(
                data["q_b_proj"],
                projection_interval(data["q_b_proj_input"], q_matrix, bf16_output=True),
            )
            local = joint[rank * 8 : (rank + 1) * 8]
            source = data["absorbed_query_input"].double().reshape(8, 192)
            exact = torch.einsum("hn,hnk->hk", source, local[:, :192])
            magnitude = torch.einsum("hn,hnk->hk", source.abs(), local[:, :192].abs())
            report["query_absorption"] = dot_check(
                data["absorbed_query"].reshape(8, 512), exact, magnitude, 192
            )
            allowance = sparse_attention_allowance(
                data["query"][0],
                data["cache"],
                data["selected"][0],
                chunks_per_split=data["attention_split_plan"]["chunks_per_split"],
                softmax_scale=1 / 16,
            )
            report["attention"] = check(
                data["attended_latent"][0],
                (allowance.expected - allowance.absolute, allowance.expected + allowance.absolute),
            )
            source = data["expanded_value_input"].double().reshape(8, 512)
            exact = torch.einsum("hk,hvk->hv", source, local[:, 192:])
            magnitude = torch.einsum("hk,hvk->hv", source.abs(), local[:, 192:].abs())
            report["value_expansion"] = dot_check(
                data["expanded_value"].reshape(8, 256), exact, magnitude, 512
            )
            report["attention_output_projection"] = check(
                data["attention_input"],
                projection_interval(
                    data["expanded_value"].reshape(1, -1),
                    output[:, rank * 2048 : (rank + 1) * 2048],
                    bf16_output=True,
                ),
            )
            begin, end = rank * 256, (rank + 1) * 256
            up_weight = torch.cat(
                (
                    weight("mlp.shared_experts.gate_proj.weight")[begin:end],
                    weight("mlp.shared_experts.up_proj.weight")[begin:end],
                )
            )
            report["shared_gate_up"] = check(
                data["shared_gate_up"],
                projection_interval(data["shared_gate_up_input"], up_weight, bf16_output=True),
            )
            gate, up = data["shared_gate_up"].double().chunk(2, -1)
            report["shared_activation"] = check(
                data["shared_down_input"], swiglu_interval(gate, gate, up, up)
            )
            report["shared_down"] = check(
                data["shared_down"],
                projection_interval(
                    data["shared_down_input"],
                    weight("mlp.shared_experts.down_proj.weight")[:, begin:end],
                    bf16_output=True,
                ),
            )
            for slot, expert in enumerate(data["expert_ids"][0].tolist()):
                stem = f"mlp.experts.{expert}."
                gate_bounds = projection_interval(
                    data["expert_contributions_input"],
                    weight(stem + "gate_proj.weight")[begin:end],
                    bf16_output=True,
                )
                up_bounds = projection_interval(
                    data["expert_contributions_input"],
                    weight(stem + "up_proj.weight")[begin:end],
                    bf16_output=True,
                )
                actual = data["expert_intermediate"][slot : slot + 1]
                report[f"expert_{slot}_activation"] = check(
                    actual, swiglu_interval(*gate_bounds, *up_bounds)
                )
                report[f"expert_{slot}_down"] = check(
                    data["expert_contributions"][:, slot],
                    projection_interval(
                        actual, weight(stem + "down_proj.weight")[:, begin:end], bf16_output=True
                    ),
                )
            products = (
                data["expert_contributions"].double() * data["expert_weights"].double()[:, :, None]
            )
            exact = products.sum(1)
            error = gamma(10) * products.abs().sum(1)
            lower, upper = rounded_bf16_interval(exact - error, exact + error)
            report["local_mlp_sum"] = check(
                data["mlp_input"],
                rounded_bf16_interval(
                    lower + data["shared_expert"].double(), upper + data["shared_expert"].double()
                ),
            )
            print("checked", consumed, "step", step, "rank", rank, flush=True)
        for name in ("attention", "mlp"):
            partials = torch.stack(
                [value["cases"]["repeat-0"][step][name + "_input"].double() for value in values]
            )
            reports[0][name + "_collective"] = dot_check(
                first[name], partials.sum(0), partials.abs().sum(0), 8
            )
            reports[0][name + "_collective"]["ranks_equal"] = all(
                torch.equal(first[name], value["cases"]["repeat-0"][step][name]) for value in values
            )
        residual = (
            first["post_attention_layernorm_input"].float() + first["mlp"].float()
        ).bfloat16()
        reports[0]["final_normalization"] = check(
            first["normalized"],
            rms_norm_interval(residual, weight("shared_head.norm.weight"), 1e-5),
        )
        head = reader.raw("lm_head.weight")
        for start in range(0, len(head), 8192):
            stop = min(len(head), start + 8192)
            reports[0][f"vocabulary_{start}_{stop}"] = check(
                first["logits"][:, start:stop],
                projection_interval(first["normalized"], head[start:stop], bf16_output=True),
            )
        del head
        results.append(reports)
    controls = {}
    if consumed == 2305:
        correct = values[0]["cases"]["repeat-0"][0]
        for label in ("wrong_hidden", "shifted_position", "recent_selection"):
            altered = values[0]["cases"][label][0]
            if label == "wrong_hidden":
                bounds = rms_norm_interval(
                    fixture["hidden"][consumed - 1 : consumed].bfloat16(),
                    weight("hnorm.weight"),
                    1e-5,
                )
                controls[label] = (
                    check(altered["joined_input"][:, 6144:], bounds)["out_of_bound"] > 0
                )
            elif label == "shifted_position":
                controls[label] = int(altered["positions"][0]) != consumed - 1
            else:
                bound = sparse_attention_allowance(
                    correct["query"][0],
                    correct["cache"],
                    correct["selected"][0],
                    chunks_per_split=correct["attention_split_plan"]["chunks_per_split"],
                    softmax_scale=1 / 16,
                )
                controls[label] = (
                    check(
                        altered["attended_latent"][0],
                        (bound.expected - bound.absolute, bound.expected + bound.absolute),
                    )["out_of_bound"]
                    > 0
                )
    failures = []
    for step, reports in enumerate(results):
        for rank, report in enumerate(reports):
            for stage, value in report.items():
                if (
                    value.get("out_of_bound", 0) > 0
                    or value.get("passed", True) is False
                    or value.get("ranks_equal", True) is False
                    or value.get("wrong_20_percent_rejected", True) is False
                ):
                    failures.append((step, rank, stage, value))
    result = {
        "capture": str(capture),
        "consumed": consumed,
        "steps": results,
        "negative_controls": controls,
        "failures": failures,
    }
    print("failures", len(failures), failures, "controls", controls, flush=True)
    assert not failures and all(controls.values()), result
    return result
