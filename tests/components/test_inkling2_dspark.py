# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DSpark numerical boundaries, committed context visibility, and checkpoint precision."""

import json
from types import SimpleNamespace
import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.components.recurrent_proposal import corrected_logits, recurrent_gate
from ignition.recipes.inkling2.dspark_state import context_slots, mix_features, rotary, stage_paged_tail, stage_window
from ignition.recipes.inkling2.dspark_weights import fold_recurrent_head, mtp_quantization
from ignition.recipes.inkling2.dspark import PreparedContext, PreparedDraft, SpeculativePreparation


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
    staged = stage_window(history, values, m, 9).flatten(0, 1)
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
    staged = stage_paged_tail(history, proposal, m, 2).flatten(0, 1)
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
