# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from ignition.components.paged_batch import gather_token_steering_scale
from ignition.components.sampling import (
    FINISHED,
    AdmissionBuffers,
    DeviceRequestState,
    RequestAdmission,
)
from ignition.preparation import runtime_core_pb2 as wire
from ignition.recipes.deepseek41.steering import (
    SteeringSidecar,
    steer_attention,
    steer_shared_expert,
    steered_identity,
)


def _write_synthetic_sidecar(path: Path, *, seed: int = 1017) -> bytes:
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(4, 37):
        raw = torch.randn(5120, generator=generator, dtype=torch.float32)
        direction = (raw / raw.norm()).to(torch.bfloat16).contiguous()
        gain = (1.0 + 0.02 * torch.randn(5120, generator=generator)).to(torch.bfloat16).contiguous()
        alpha = torch.tensor(3.0 + 0.01 * (layer - 4), dtype=torch.float32)
        prefix = f"layers.{layer}"
        tensors[f"{prefix}.alpha"] = alpha
        tensors[f"{prefix}.attn_row_gain"] = gain
        tensors[f"{prefix}.direction"] = direction
    save_file(tensors, str(path))
    return hashlib.sha256(path.read_bytes()).digest()


def test_sidecar_inventory_identity_and_validation(tmp_path: Path) -> None:
    sidecar_a = tmp_path / "steering_a.safetensors"
    sidecar_b = tmp_path / "steering_b.safetensors"
    expected_sha256 = _write_synthetic_sidecar(sidecar_a, seed=1017)
    _write_synthetic_sidecar(sidecar_b, seed=2026)

    steering_a = SteeringSidecar.load(sidecar_a, torch.device("cpu"))
    steering_b = SteeringSidecar.load(sidecar_b, torch.device("cpu"))
    assert steering_a.sha256 == expected_sha256
    assert steering_a.sha256 != steering_b.sha256
    assert tuple(steering_a.layers) == tuple(range(4, 37))

    base_identity = "ab" * 32
    assert steered_identity(base_identity, None) == base_identity
    id_a = steered_identity(base_identity, steering_a.sha256)
    id_b = steered_identity(base_identity, steering_b.sha256)
    assert len(id_a) == 64 and id_a != base_identity and id_a != id_b

    raw_tensors = load_file(str(sidecar_a))
    for idx, layer in steering_a.layers.items():
        assert layer.direction.shape == (5120,)
        assert layer.direction.dtype == torch.float32
        assert layer.attention_gain_delta.shape == (5120,)
        assert layer.attention_gain_delta.dtype == torch.float32
        assert torch.allclose(1.0 + layer.attention_gain_delta, raw_tensors[f"layers.{idx}.attn_row_gain"].float())
        assert layer.alpha > 0.0

    bad_path = tmp_path / "bad_alpha.safetensors"
    _write_synthetic_sidecar(bad_path)
    bad_tensors = load_file(str(bad_path))
    bad_tensors["layers.10.alpha"] = torch.tensor(0.0, dtype=torch.float32)
    save_file(bad_tensors, str(bad_path))
    with pytest.raises(ValueError, match="non-positive-alpha"):
        SteeringSidecar.load(bad_path, torch.device("cpu"))


def test_runtime_core_schema_round_trips_steering_metadata() -> None:
    defaults = wire.DefaultSamplingParameters(
        steering_scale=0.0,
        ranges=wire.SamplingRanges(
            steering_scale=wire.SamplingFloatRange(min=0.0, max=1.0)
        ),
    )
    decoded = wire.DefaultSamplingParameters.FromString(defaults.SerializeToString())
    assert decoded.HasField("steering_scale")
    assert decoded.steering_scale == 0.0
    assert decoded.ranges.HasField("steering_scale")
    assert decoded.ranges.steering_scale.min == 0.0
    assert decoded.ranges.steering_scale.max == 1.0


def test_mixed_rows_preserve_unsteered_bits_and_steer_independently(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "steering.safetensors"
    _write_synthetic_sidecar(sidecar_path)
    steering = SteeringSidecar.load(sidecar_path, torch.device("cpu"))
    layer = steering.layers[18]
    parameters = torch.zeros((8, 7), dtype=torch.float32)
    parameters[:4, 6] = torch.tensor([0.0, 1.0, 0.5, 0.0])
    row_slots = torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32)
    requests = torch.tensor([0, 1, 1, 2, 3, 4], dtype=torch.int64)
    positions = torch.tensor([10, 0, 1, 5, 22, -1], dtype=torch.int64)
    scale = gather_token_steering_scale(parameters, row_slots, requests, positions)
    assert scale[:, 0].tolist() == [0.0, 1.0, 1.0, 0.5, 0.0, 0.0]

    generator = torch.Generator().manual_seed(42)
    attention = torch.randn(6, 5120, dtype=torch.bfloat16, generator=generator)
    shared = torch.randn(6, 5120, dtype=torch.bfloat16, generator=generator)
    attention_out = steer_attention(attention, layer, scale)
    shared_out = steer_shared_expert(shared, layer, scale)
    for row in (0, 4, 5):
        assert torch.equal(attention_out[row], attention[row])
        assert torch.equal(shared_out[row], shared[row])
    for row in (1, 2, 3):
        assert not torch.equal(shared_out[row], shared[row])
        assert not torch.equal(attention_out[row], attention[row])
        s_val = float(scale[row, 0].item())
        att_f = attention[row].float()
        ref_att = (
            (att_f - s_val * layer.alpha * (att_f @ layer.direction) * layer.direction)
            * (1.0 + s_val * layer.attention_gain_delta)
        ).to(torch.bfloat16)
        assert torch.equal(attention_out[row], ref_att)


def test_shared_expert_projection_commutes_with_tp4_sum(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "steering.safetensors"
    _write_synthetic_sidecar(sidecar_path)
    steering = SteeringSidecar.load(sidecar_path, torch.device("cpu"))
    scale = torch.tensor([[1.0], [0.0], [0.75], [1.0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(7)
    for layer_id in (4, 15, 24, 36):
        layer = steering.layers[layer_id]
        partials = [
            (torch.randn(4, 5120, generator=generator) + 12.0 * layer.direction).to(torch.bfloat16)
            for _ in range(4)
        ]
        # Each rank rounds its steered shared-expert partial to BF16 before the TP4 reduction.
        rank_steered = [steer_shared_expert(partial, layer, scale) for partial in partials]
        assert all(item.dtype == torch.bfloat16 for item in rank_steered)
        reduced_steered = sum(item.float() for item in rank_steered).to(torch.bfloat16)
        unsteered_reduced = sum(partial.float() for partial in partials).to(torch.bfloat16)
        post_reduce_reference = steer_shared_expert(unsteered_reduced, layer, scale)

        # Steering moves active rows substantially (>25% norm delta) while leaving scale==0 bit-exact.
        effect = (post_reduce_reference[0].float() - unsteered_reduced[0].float()).norm() / unsteered_reduced[0].float().norm()
        assert float(effect) > 0.25
        assert torch.equal(reduced_steered[1], unsteered_reduced[1])
        # Four per-rank BF16 roundings before summation agree with post-sum projection within BF16 precision.
        relative = (reduced_steered.float() - post_reduce_reference.float()).norm() / post_reduce_reference.float().norm()
        assert float(relative) < 5e-3


def test_request_state_keeps_legacy_width_and_rejects_atomically() -> None:
    unsteered = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=4)
    steered = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=4, steering=True)
    unsteered_staging = AdmissionBuffers.allocate("cpu", vocab_size=32, capacity=4)
    steered_staging = AdmissionBuffers.allocate(
        "cpu", vocab_size=32, capacity=4, steering=True
    )
    assert unsteered.parameters.shape == (4, 6)
    assert unsteered_staging.admission_parameters.shape == (4, 6)
    assert steered.parameters.shape == (4, 7)
    assert steered_staging.admission_parameters.shape == (4, 7)

    baseline = RequestAdmission(generation=5, seed=99, max_new_tokens=0, steering_scale=0.0)
    unsteered.admit(0, baseline, [1, 2])
    unsteered_staging.stage(0, 3, baseline, [1, 2])
    prior_seed_bits = int(unsteered_staging.admission_seed_bits[0].item())

    steered_req = RequestAdmission(generation=9, seed=2, steering_scale=1.0)
    steered.admit(0, steered_req, [1, 2])
    steered_staging.stage(0, 0, steered_req, [1, 2])
    assert steered.parameters[0, 6].item() == 1.0
    assert steered_staging.admission_parameters[0, 6].item() == 1.0

    with pytest.raises(ValueError, match="steering-enabled"):
        unsteered.admit(0, steered_req, [1, 2])
    assert int(unsteered.generation[0].item()) == 5
    assert int(unsteered.status[0].item()) == FINISHED
    assert int(unsteered.seed[0].item()) == 99

    with pytest.raises(ValueError, match="steering-enabled"):
        unsteered_staging.stage(0, 1, steered_req, [1, 2])
    assert int(unsteered_staging.admission_slots[0].item()) == 3
    assert int(unsteered_staging.admission_generations[0].item()) == 5
    assert int(unsteered_staging.admission_seed_bits[0].item()) == prior_seed_bits


def test_metadata_pack_and_set_propagate_steering_scale_under_fullgraph_compile(
    tmp_path: Path,
) -> None:
    from ignition.components.paged_batch import Metadata

    sidecar_path = tmp_path / "steering.safetensors"
    _write_synthetic_sidecar(sidecar_path)
    steering = SteeringSidecar.load(sidecar_path, torch.device("cpu"))
    layer = steering.layers[24]

    parameters = torch.zeros((4, 7), dtype=torch.float32)
    parameters[1, 6] = 0.0
    parameters[2, 6] = 1.0
    row_slot_ids = torch.tensor([1, 2], dtype=torch.int32)
    starts = torch.tensor([10, 20], dtype=torch.int64)
    lengths = torch.tensor([2, 3], dtype=torch.int32)
    states = torch.tensor([1, 2], dtype=torch.int64)
    pages = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    candidates = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int64)
    live_rows = torch.tensor([2], dtype=torch.int32)

    def packed_step(hidden: torch.Tensor) -> torch.Tensor:
        packed = Metadata.pack(
            starts,
            lengths,
            states,
            pages,
            candidates,
            live_rows,
            token_capacity=6,
            sample_count=8,
            max_query_tokens=4,
            parameters=parameters,
            row_slot_ids=row_slot_ids,
        )
        assert packed.steering_scale is not None
        return steer_shared_expert(hidden, layer, packed.steering_scale)

    compiled = torch.compile(packed_step, backend="eager", fullgraph=True)
    hidden = torch.randn(6, 5120, dtype=torch.bfloat16)
    out = compiled(hidden)
    assert torch.equal(out[0], hidden[0])
    assert torch.equal(out[1], hidden[1])
    assert not torch.equal(out[2], hidden[2])
    assert not torch.equal(out[3], hidden[3])
    assert not torch.equal(out[4], hidden[4])
    assert torch.equal(out[5], hidden[5])

    allocated = Metadata.allocate(6, 2, 2, torch.device("cpu"), samples=8, steering=True)
    allocated.set(
        starts,
        lengths,
        states,
        pages,
        candidates,
        live_rows,
        parameters=parameters,
        row_slot_ids=row_slot_ids,
    )
    assert allocated.steering_scale is not None
    assert allocated.steering_scale.squeeze(1).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]


def test_direct_execution_packs_steering_and_isolates_prefix_cache() -> None:
    from ignition.direct.packing import pack
    from ignition.direct.scheduler import (
        AdmissionError,
        Checkpoint,
        Lane,
        Layout,
        Request,
        SamplingParameters,
        Scheduler,
        Step,
    )
    from ignition.direct.server import _sampling

    unsteered_defaults = {"seed": 0, "temperature": 1.0, "top_p": 1.0, "min_p": 0.0,
                          "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "top_k": -1}
    steered_defaults = {**unsteered_defaults, "steering_scale": 0.0,
                        "ranges": {"steering_scale": {"min": 0.0, "max": 1.0}}}

    # Absent range clamps steering_scale to default 0.0; declared range clamps to [0, 1], including huge ints.
    assert _sampling({"steering_scale": 0.8}, unsteered_defaults).steering_scale == 0.0
    assert _sampling({"steering_scale": 0.75}, steered_defaults).steering_scale == 0.75
    assert _sampling({"chat_template_kwargs": {"steering_scale": 0.5}}, steered_defaults).steering_scale == 0.5
    with pytest.raises(AdmissionError, match="finite"):
        _sampling({"steering_scale": 10**400}, steered_defaults)
    with pytest.raises(AdmissionError, match="finite"):
        _sampling({"chat_template_kwargs": {"steering_scale": 10**400}}, steered_defaults)
    with pytest.raises(AdmissionError, match="must match"):
        _sampling({"steering_scale": 1e301, "chat_template_kwargs": {"steering_scale": 2e301}}, steered_defaults)
    assert _sampling({"steering_scale": -0.0}, steered_defaults).steering_bits() == 0
    with pytest.raises(AdmissionError, match="must match"):
        _sampling({"steering_scale": 1.0, "chat_template_kwargs": {"steering_scale": 0.25}}, steered_defaults)
    with pytest.raises(AdmissionError, match="must match"):
        _sampling(
            {"steering_scale": 9007199254740992, "chat_template_kwargs": {"steering_scale": 9007199254740993}},
            steered_defaults,
        )

    assert SamplingParameters().values(6) == (1.0, 1.0, 0.0, 1.0, 0.0, 0.0)
    assert SamplingParameters(steering_scale=0.5).values(7) == (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5)
    with pytest.raises(AdmissionError, match="7-column"):
        SamplingParameters(steering_scale=0.5).values(6)

    layout = Layout(
        capacity=2, max_active=2, context=128, page_tokens=2, table_width=4, pages=16, snapshots=4,
        prefill_chunk=64, vocabulary=32, eos=(1,),
        graphs={
            ("init", 1, 1): "init_1",
            ("restore", 1, 1): "restore_1",
            ("snapshot", 1, 1): "snapshot_1",
            ("prefill", 1, 8): "prefill_8",
            ("resume", 1, 1): "resume_1",
            ("decode", 1, 1): "decode_1",
        },
        widths=(), fixed_width=None,
        bindings={"init_1": [{"buffer": "admission_parameters", "source": wire.SAMPLING_PARAMETERS,
                              "indexing": wire.SUBMISSION_ROW, "value_offset": 0, "padding": 0}]},
        buffer_specs={"admission_parameters": (np.dtype("float32"), (2, 7))},
        page_views={}, state_rows_per_request=1, draft_pages=0, record_fields=24,
        confidence_widths={}, completion_cursors={},
    )
    assert layout.parameter_width == 7

    req = Request(1, [4, 5, 6], 1, SamplingParameters(steering_scale=0.75))
    req.slot, req.generation = 0, 1
    step = Step("init", "init_1", 1, 1, [Lane(req, 0, 1)], 1)
    uploads = pack(layout, step)
    assert len(uploads) == 1
    packed = np.frombuffer(uploads[0][2], dtype=np.float32).reshape(2, 7)
    assert float(packed[0, 6]) == 0.75
    assert float(packed[1, 6]) == 0.0

    # End-to-end Scheduler checkpoint isolation across distinct scales and reuse within matching scales.
    scheduler = Scheduler(layout, prefix_cache=True)

    def run_to_done(r: Request) -> None:
        scheduler.submit(r)
        while not r.done:
            s = scheduler.plan()
            assert s is not None
            rows = np.zeros((layout.capacity, layout.record_fields), dtype=np.int64)
            rows[:, 4] = -1
            rows[:, 16:] = -1
            lane = s.lanes[0]
            lr = lane.request
            row = rows[lr.slot]
            if s.kind in ("init", "restore"):
                row[:8] = (lr.generation, s.sequence, lr.consumed, 0, -1, 1, 0, 0)
            elif s.kind == "snapshot":
                row[:8] = (lr.generation, s.sequence, lr.consumed, len(lr.generated), -1, 2 if lr.finished_early else 1, 0, 0)
            else:
                emitted = [9] if (lr.consumed + lane.tokens) >= len(lr.prompt) else []
                status = 2 if emitted else 1
                consumed = lr.consumed + lane.tokens
                row[:8] = (lr.generation, s.sequence, consumed, len(lr.generated) + len(emitted),
                           emitted[-1] if emitted else -1, status, 0, 0)
                row[8] = len(emitted)
                row[12], row[13], row[15] = -1, max(consumed - 1, 0), s.width
                if emitted:
                    row[16] = emitted[0]
            scheduler.complete(s, rows, 100)

    r0_a = Request(10, [4, 5, 6, 7], 1, SamplingParameters(steering_scale=0.0))
    r1_a = Request(11, [4, 5, 6, 7], 1, SamplingParameters(steering_scale=1.0))
    r0_b = Request(12, [4, 5, 6, 7], 1, SamplingParameters(steering_scale=-0.0))
    r1_b = Request(13, [4, 5, 6, 7], 1, SamplingParameters(steering_scale=1.0))
    run_to_done(r0_a)
    run_to_done(r1_a)
    run_to_done(r0_b)
    run_to_done(r1_b)
    assert r0_a.reused == 0
    assert r1_a.reused == 0
    assert r0_b.reused == 4
    assert r1_b.reused == 4
    assert isinstance(scheduler.cache[(4, 5, 6, 7)], Checkpoint)
