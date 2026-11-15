# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DSpark numerical boundaries, committed context visibility, and checkpoint precision."""

import json
from types import SimpleNamespace
import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.components.recurrent_proposal import PreparedRecurrentProposalHead, corrected_logits, recurrent_gate
from ignition.recipes.inkling2.dspark_state import (block_inputs, context_slots, mix_features, rotary, rotary_table,
                                                    rotate, stage, tail_plan, window_plan)
from ignition.recipes.inkling2.dspark_weights import DraftWeights, fold_recurrent_head, mtp_quantization
from ignition.recipes.inkling2.dspark import DraftScratch, PreparedContext, PreparedDraft, SpeculativePreparation


@pytest.mark.parametrize("draft_tokens", [0, 1, 7])
def test_speculative_preparation_retains_context_at_zero_draft_depth(draft_tokens):
    config = SimpleNamespace(text={"hidden_size": 128}, vocabulary=128,
        draft=SimpleNamespace(block_size=8, heads=4, kv_heads=4, head_dim=128,
            text={"intermediate_size": 256}))
    weights = SimpleNamespace(config=config, shard=TensorParallelShard(0, 4))
    pool = SimpleNamespace(backing=torch.zeros((2, 16), dtype=torch.uint8))
    requests = SimpleNamespace(capacity=1, generation=torch.zeros(1, dtype=torch.int64))
    preparation = SpeculativePreparation(weights, pool, requests, None, None, decoder=object(),
        completion=None, frame=None, sequence=None, session=None, workspace=None, collectives={},
        draft_tokens=draft_tokens)
    assert preparation.block_size == 8 and preparation.logit_offset == 1
    assert preparation.forwards.context is PreparedContext
    assert preparation.forwards.draft is PreparedDraft
    assert preparation.draft_scratch.hidden.shape == (8, 128)
    assert preparation.draft_scratch.anchor.shape == (1, 128)
    assert not preparation.offset.count_nonzero()
    assert not preparation.generation.count_nonzero()


def test_recurrent_gate_uses_each_bf16_rounding_boundary():
    generator = torch.Generator().manual_seed(122)
    raw = torch.randn(3, 768, generator=generator) * 2
    state = torch.randn(3, 256, generator=generator).bfloat16()
    gate, candidate, output = raw.bfloat16().chunk(3, -1)
    gate = gate.sigmoid()
    expected = gate * state + (1 - gate) * candidate.tanh()
    actual, latent = recurrent_gate(raw, state)
    assert torch.equal(actual, expected)
    assert torch.equal(latent, output.tanh())
    assert not torch.equal(actual, (raw[:, :256].sigmoid() * state.float()
        + (1 - raw[:, :256].sigmoid()) * raw[:, 256:512].tanh()).bfloat16())


def test_logits_round_addition_before_fp32_softcap():
    base = torch.tensor([[1., 2., 30., -16.]], dtype=torch.bfloat16)
    correction = torch.tensor([[.007, .009, .09, -.06]], dtype=torch.bfloat16)
    expected = (30 * ((base + correction).float() / 30).tanh()).bfloat16()
    assert torch.equal(corrected_logits(base, correction, 30), expected)
    assert not torch.equal(expected, (30 * ((base.float() + correction.float()) / 30).tanh()).bfloat16())


def test_fold_products_use_fp64_before_bf16_storage():
    generator = torch.Generator().manual_seed(2)
    head = torch.randn(129, 32, generator=generator).bfloat16()
    weight = torch.randn(32, 8, generator=generator).bfloat16()
    bias = torch.randn(32, generator=generator).bfloat16()
    folded, offset = fold_recurrent_head(head, weight, bias, chunk=17)
    assert torch.equal(folded, (head.double() @ weight.double()).bfloat16())
    assert torch.equal(offset, (head.double() @ bias.double()).bfloat16())


def test_feature_mixtures_keep_raw_taps_until_fp32_normalization():
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(3, 8, 32, generator=generator).bfloat16()
    alpha = torch.randn(6, 8, generator=generator).softmax(-1).bfloat16()
    weight = torch.randn(32, generator=generator).bfloat16()
    mixed = sum(features[:, tap].float() * alpha[2, tap] for tap in range(8))
    expected = (mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).bfloat16()
    assert torch.equal(mix_features(features, alpha[2], weight, 1e-6), expected)


@pytest.mark.parametrize("start", [0, 3, 1023, 1024, 1027])
def test_noncausal_staging_retains_one_prefix_for_all_eight_positions(start):
    window, block = 1024, 8
    history = torch.zeros(3, window, 1, 2, dtype=torch.bfloat16)
    for position in range(max(0, start - window), start):
        history[1, position % window].fill_(position)
    values = torch.arange(start, start + block).bfloat16()[:, None, None].expand(block, 1, 2)
    m = Metadata.pack(torch.tensor([start]), torch.tensor([8]), torch.tensor([1]),
        torch.tensor([[0] * 16], dtype=torch.int32), torch.zeros(1, block, dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32), token_capacity=8, sample_count=8)
    plan = window_plan(m, window, 9)
    staged = stage(history, values, plan).flatten(0, 1)
    assert plan.lengths.tolist() == [min(start, window) + block]
    prefix = min(start, window)
    expected = torch.arange(start - prefix, start + block).bfloat16()
    assert torch.equal(staged[:prefix + block, 0, 0], expected)
    assert not staged[prefix + block:].count_nonzero()
    assert torch.equal(staged[prefix:prefix + block], values)


def test_context_writes_only_accepted_target_rows_and_preserves_inactive_rows():
    m = Metadata.pack(torch.tensor([1022, 127, 19]), torch.tensor([8, 8, 0]), torch.tensor([1, 2, 0]),
        torch.tensor([[2] * 16, [5] * 16, [-1] * 16], dtype=torch.int32),
        torch.zeros(3, 8, dtype=torch.int64), torch.tensor([3], dtype=torch.int32),
        token_capacity=24, sample_count=24)
    counts = torch.tensor([3, 1, 0])
    local, full = context_slots(m, counts, window=1024), context_slots(m, counts)
    assert local[:8].tolist() == [2046, 2047, 1024, -1, -1, -1, -1, -1]
    assert full[:8].tolist() == [382, 383, 256, -1, -1, -1, -1, -1]
    assert local[8:16].tolist() == [2175, -1, -1, -1, -1, -1, -1, -1]
    assert (local[16:] == -1).all() and (full[16:] == -1).all()


def test_paged_tail_preserves_committed_prefix_with_one_target_page():
    history = torch.arange(4 * 128).view(4, 128, 1, 1).bfloat16()
    before = history.clone()
    proposal = torch.arange(1000, 1008).view(8, 1, 1).bfloat16()
    m = Metadata.pack(torch.tensor([127]), torch.tensor([8]), torch.tensor([1]),
        torch.tensor([[2]], dtype=torch.int32), torch.zeros(1, 8, dtype=torch.int64),
        torch.tensor([1], dtype=torch.int32), token_capacity=8, sample_count=8)
    plan = tail_plan(m, 128, 2)
    staged = stage(history, proposal, plan).flatten(0, 1)
    assert plan.lengths.tolist() == [135]
    assert torch.equal(staged[:127], history[2, :127])
    assert torch.equal(staged[127:135], proposal)
    assert not staged[135:].count_nonzero()
    assert torch.equal(history, before)
    assert context_slots(m, torch.tensor([8])).tolist() == [383, -1, -1, -1, -1, -1, -1, -1]


def test_rotary_uses_full_head_fp32_angles_and_output_rounding():
    values = torch.arange(128).view(1, 1, 128).bfloat16()
    inverse = 1 / (1e6 ** (torch.arange(0, 128, 2).float() / 128))
    angles = 1025 * inverse
    left, right = values.float().chunk(2, -1)
    expected = torch.cat((left * angles.cos() - right * angles.sin(),
                          right * angles.cos() + left * angles.sin()), -1).bfloat16()
    assert torch.equal(rotary(values, torch.tensor([1025]), inverse), expected)


def test_mtp_mxfp8_metadata_requires_activation_quantization_and_complete_coverage():
    name = "layers.0.mlp.gate_proj.weight"
    group = dict(compute_dtype="mxfp8", output_dtype="float8_e4m3fn", weight_transform="mxfp8_nearest",
        weight_block_shape=[1, 32], required_activation_quantization={
            "block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}, tensors=[name])
    config = {"mtp": {"parameter_groups": [group]}}
    names = [name, name.removesuffix(".weight") + ".weight_scale"]
    assert mtp_quantization(config, names) == {"layers.0.mlp.gate_proj": {"quant_algo": "MXFP8", "group_size": 32}}
    changed = json.loads(json.dumps(config))
    changed["mtp"]["parameter_groups"][0]["required_activation_quantization"] = None
    with pytest.raises(ValueError, match="activations"):
        mtp_quantization(changed, names)
    with pytest.raises(ValueError, match="cover exactly"):
        mtp_quantization(config, names + ["undeclared.weight"])


def test_one_staging_plan_serves_key_and_value_histories_of_every_layer():
    generator = torch.Generator().manual_seed(9)
    window, block, pages = 1024, 8, 9
    m = Metadata.pack(torch.tensor([1027, 5, 300]), torch.tensor([8, 8, 0]), torch.tensor([2, 1, 0]),
        torch.tensor([[0] * 16, [1] * 16, [-1] * 16], dtype=torch.int32), torch.zeros(3, block, dtype=torch.int64),
        torch.tensor([3], dtype=torch.int32), token_capacity=24, sample_count=24)
    plan = window_plan(m, window, pages)
    offsets = torch.arange(pages * 128)[None]
    prefix = m.starts.clamp(max=window)[:, None]
    for _ in range(2):
        history = torch.randn(3, window, 2, 4, generator=generator).bfloat16()
        values = torch.randn(24, 2, 4, generator=generator).bfloat16()
        previous = history[m.state_rows[:, None], (m.starts[:, None] - prefix + offsets).remainder(window)]
        current = values[(m.query_start_loc[:-1, None] + offsets - prefix).clamp(0, 23)]
        valid = (m.lengths[:, None] > 0) & (offsets < prefix + m.lengths[:, None])
        expected = torch.where(valid[:, :, None, None],
            torch.where((offsets < prefix)[:, :, None, None], previous, current), 0).view(3 * pages, 128, 2, 4)
        assert torch.equal(stage(history, values, plan), expected)
    assert plan.lengths.tolist() == [1032, 13, 0]
    assert not stage(history, values, plan)[2 * pages:].count_nonzero()


def test_shared_rotary_table_matches_per_call_rotation():
    generator = torch.Generator().manual_seed(12)
    positions = torch.tensor([0, 7, 1025, -1, 1048575])
    inverse = 1 / (1e6 ** (torch.arange(0, 128, 2).float() / 128))
    table = rotary_table(positions, inverse)
    for heads in (2, 12):
        values = torch.randn(5, heads, 128, generator=generator).bfloat16()
        assert torch.equal(rotate(values, table), rotary(values, positions, inverse))


def test_block_inputs_embed_only_each_request_first_position():
    generator = torch.Generator().manual_seed(15)
    anchor = torch.randn(3, 32, generator=generator).bfloat16()
    mask = torch.randn(32, generator=generator).bfloat16()
    positions = torch.arange(24) % 8
    expected = torch.where((positions == 0)[:, None], anchor.repeat_interleave(8, 0), mask)
    assert torch.equal(block_inputs(anchor, mask, 8), expected)


def test_draft_scratch_keeps_one_anchor_row_per_block():
    config = SimpleNamespace(text={"hidden_size": 128}, vocabulary=256,
        draft=SimpleNamespace(heads=4, kv_heads=4, head_dim=128, text={"intermediate_size": 256}))
    scratch = DraftScratch.allocate(config, TensorParallelShard(0, 4), 32, 8, torch.device("cpu"))
    assert scratch.anchor.shape == (4, 128) and scratch.hidden.shape == (32, 128)
    rows = scratch.rows(16, 16)
    assert rows.anchor.shape == (2, 128) and rows.hidden.shape == (16, 128)
    assert rows.anchor.data_ptr() == scratch.anchor.data_ptr()
    with pytest.raises(ValueError, match="whole blocks"):
        scratch.rows(12, 12)
    with pytest.raises(ValueError, match="whole blocks"):
        DraftScratch.allocate(config, TensorParallelShard(0, 4), 12, 8, torch.device("cpu"))


def _recurrent_head(weight, bias, head, head_bias, positions, *, row_scaled):
    """Bind FP64 reference projections to the recurrent head's arithmetic."""
    rank = head.shape[1]
    operation = object.__new__(PreparedRecurrentProposalHead)
    operation.rank, operation.softcap, operation.positions = rank, 30., positions
    operation.context_bias = bias
    operation.context = lambda source: (source.double() @ weight[:, 2 * rank:].double().T).float()
    operation.dynamic = lambda source: (source.double() @ weight[:, :2 * rank].double().T).float()
    operation.embed = lambda token_ids: torch.zeros(token_ids.numel(), rank, dtype=torch.bfloat16)
    if row_scaled:
        operation.output = lambda latent: (latent.double() @ head.double().T).float()
        operation.output_bias = head_bias
    else:
        operation.output = lambda latent: (latent.double() @ head.double().T + head_bias.double()).bfloat16()
        operation.output_bias = None
    return operation


def test_block_context_projection_matches_one_projection_per_position():
    generator = torch.Generator().manual_seed(18)
    rows, positions, hidden, rank, vocabulary = 3, 7, 16, 8, 24
    weight = (torch.randn(3 * rank, 2 * rank + 2 * hidden, generator=generator) / 8).bfloat16()
    bias = torch.randn(3 * rank, generator=generator) / 8
    head = (torch.randn(vocabulary, rank, generator=generator) / 4).bfloat16()
    head_bias = (torch.randn(vocabulary, generator=generator) / 4).bfloat16()
    states = torch.randn(rows, positions, hidden, generator=generator).bfloat16()
    context = torch.randn(rows, hidden, generator=generator).bfloat16()
    operation = _recurrent_head(weight, bias, head, head_bias, positions, row_scaled=False)
    fixed = operation.project_context(states, context)
    assert fixed.shape == (rows, positions, 3 * rank) and fixed.dtype == torch.float32
    for index in range(positions):
        source = torch.cat((states[:, index], context), -1)
        assert torch.equal(fixed[:, index], operation.context(source) + bias)
    with pytest.raises(ValueError, match="every prepared block position"):
        operation.project_context(states[:, :3], context)


def test_row_scaled_correction_adds_the_bf16_bias_in_fp32_before_bf16_rounding():
    generator = torch.Generator().manual_seed(21)
    rows, hidden, rank, vocabulary = 2, 16, 8, 24
    weight = (torch.randn(3 * rank, 2 * rank + 2 * hidden, generator=generator) / 8).bfloat16()
    bias = torch.randn(3 * rank, generator=generator) / 8
    head = (torch.randn(vocabulary, rank, generator=generator) / 4).bfloat16()
    head_bias = (torch.randn(vocabulary, generator=generator) / 4).bfloat16()
    base = torch.randn(rows, vocabulary, generator=generator).bfloat16()
    fixed = torch.randn(rows, 3 * rank, generator=generator)
    previous = torch.zeros(rows, dtype=torch.int64)
    outputs = []
    for row_scaled in (False, True):
        operation = _recurrent_head(weight, bias, head, head_bias, 1, row_scaled=row_scaled)
        state, logits = operation.step(operation.begin(base), previous, fixed, base)
        outputs.append((state, logits))
    state, latent = recurrent_gate(fixed, torch.zeros(rows, rank, dtype=torch.bfloat16))
    correction = ((latent.double() @ head.double().T).float() + head_bias.float()).bfloat16()
    assert torch.equal(outputs[1][0], state)
    assert torch.equal(outputs[1][1], corrected_logits(base, correction, 30.))
    # With identical weights the two bias placements differ only by one FP32 addition before BF16 rounding.
    assert torch.equal(outputs[0][0], outputs[1][0])
    torch.testing.assert_close(outputs[0][1].float(), outputs[1][1].float(), rtol=2**-7, atol=2**-9)


def test_folded_correction_head_uses_the_supplied_allocator():
    generator = torch.Generator().manual_seed(24)
    head = torch.randn(64, 32, generator=generator).bfloat16()
    weight = torch.randn(32, 8, generator=generator).bfloat16()
    bias = torch.randn(32, generator=generator).bfloat16()
    calls = []

    def transient(shape, dtype, device):
        calls.append(shape)
        return torch.empty(shape, dtype=dtype, device=device)

    folded, offset = fold_recurrent_head(head, weight, bias, allocate=transient)
    assert calls == [(64, 8)]
    persistent, persistent_offset = fold_recurrent_head(head, weight, bias)
    assert torch.equal(folded, persistent) and torch.equal(offset, persistent_offset)


@pytest.mark.parametrize("arguments,message", [
    (dict(draft_head_quantization="fp8"), "draft head quantization"),
    (dict(markov_head_quantization="nvfp4"), "correction head quantization"),
])
def test_draft_weights_reject_unsupported_head_formats_before_reading(tmp_path, arguments, message):
    with pytest.raises(ValueError, match=message):
        DraftWeights.load(tmp_path, None, torch.device("cpu"), TensorParallelShard(0, 4), target_head=None, **arguments)
