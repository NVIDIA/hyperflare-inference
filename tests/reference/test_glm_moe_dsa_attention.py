# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hand-computable sparse attention and precision boundaries for independent references."""

import pytest
import torch

from ignition.reference.glm_moe_dsa_attention import fp32_dot_allowance, sparse_attention_arithmetic
from ignition.reference.glm_moe_dsa_attention_bounds import sparse_attention_allowance, two_term_e4m3_allowance
from ignition.reference.glm_moe_dsa_cache import encode_latent


@pytest.mark.parametrize("chunks_per_split", [1, 2])
def test_constant_values_survive_split_storage(chunks_per_split):
    latent = torch.full((128, 512), 1.75, dtype=torch.bfloat16)
    records = encode_latent(latent, torch.zeros((128, 64), dtype=torch.bfloat16))
    query = torch.zeros((8, 576), dtype=torch.bfloat16)
    result = sparse_attention_arithmetic(query, records, torch.arange(128),
                                         chunks_per_split=chunks_per_split, softmax_scale=1/16)
    assert torch.equal(result.output, torch.full_like(result.output, 1.75))
    torch.testing.assert_close(result.split_probabilities,
                               torch.full_like(result.split_probabilities, chunks_per_split / 2), atol=0, rtol=0)


def test_padding_does_not_read_its_clamped_record():
    latent = torch.full((65, 512), 3.5, dtype=torch.bfloat16)
    latent[0].fill_(-224)
    records = encode_latent(latent, torch.zeros((65,64), dtype=torch.bfloat16))
    selected = torch.cat((torch.arange(1,65), torch.full((64,), -1)))
    result = sparse_attention_arithmetic(torch.zeros((8,576), dtype=torch.bfloat16), records, selected,
                                         chunks_per_split=1, softmax_scale=1/16)
    assert torch.equal(result.output, torch.full_like(result.output, 3.5))
    assert torch.equal(result.split_probabilities, torch.tensor([[1.,0.]]).expand(8,-1))
    empty = sparse_attention_arithmetic(torch.zeros((8,576), dtype=torch.bfloat16), records,
                                        torch.full((128,), -1), chunks_per_split=1, softmax_scale=1/16)
    assert not empty.output.any() and not empty.split_probabilities.any()


def test_rotary_key_controls_attention_without_becoming_a_value():
    latent = torch.stack((torch.ones(512), -torch.ones(512))).bfloat16()
    rotary = torch.zeros((2,64), dtype=torch.bfloat16)
    rotary[:,0] = torch.tensor([16.,-16.])
    records = encode_latent(latent, rotary)
    query = torch.zeros((8,576), dtype=torch.bfloat16)
    query[:,512] = 1
    result = sparse_attention_arithmetic(query, records, torch.arange(2), chunks_per_split=1, softmax_scale=1/16)
    expected = torch.tanh(torch.tensor(1., dtype=torch.float64))
    torch.testing.assert_close(result.dense, expected.expand_as(result.dense), atol=1e-7, rtol=0)
    reversed_records = encode_latent(latent, -rotary)
    control = sparse_attention_arithmetic(query, reversed_records, torch.arange(2), chunks_per_split=1, softmax_scale=1/16)
    assert (result.output > 0).all() and (control.output < 0).all()


def test_fp32_projection_bound_rejects_a_bf16_roundtrip():
    # Two exactly representable products sum to a BF16 midpoint. FP32 retains
    # the midpoint. Rounding to BF16 loses 1/256 even in a 6144-term dot.
    exact = torch.tensor([1 + 1/256], dtype=torch.float64)
    bound = fp32_dot_allowance(exact, exact, 6144, bf16_output=False)
    assert (exact.float().double() - exact).abs().le(bound).all()
    assert (exact.bfloat16().double() - exact).abs().gt(bound).all()
    bf16_bound = fp32_dot_allowance(exact, exact, 6144, bf16_output=True)
    assert (exact.bfloat16().double() - exact).abs().le(bf16_bound).all()


def test_two_term_fp8_bound_covers_midpoints_and_subnormals():
    codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    codes = codes[torch.isfinite(codes)].unique(sorted=True)
    midpoints = (codes[:-1] + codes[1:]) / 2
    values = torch.cat((codes, midpoints, midpoints + 2**-18, midpoints - 2**-18)).float()
    high = values.to(torch.float8_e4m3fn).float()
    low = (values - high).to(torch.float8_e4m3fn).float()
    error = (high.double() + low.double() - values.double()).abs()
    assert (error <= two_term_e4m3_allowance(values.double(), torch.ones_like(error))).all()
    # A single FP8 term cannot satisfy the two-term arithmetic contract.
    assert ((high.double() - values.double()).abs()
            > two_term_e4m3_allowance(values.double(), torch.ones_like(error))).any()


@pytest.mark.parametrize("chunks_per_split", [1, 2])
def test_attention_allowance_rejects_wrong_scale_and_output(chunks_per_split):
    generator = torch.Generator().manual_seed(5300)
    latent = (torch.randn((128, 512), generator=generator) / 8 + 0.5).bfloat16()
    records = encode_latent(latent, torch.zeros((128, 64), dtype=torch.bfloat16))
    query = (torch.randn((8, 576), generator=generator) / 8).bfloat16()
    selected = torch.arange(128)
    options = dict(chunks_per_split=chunks_per_split, softmax_scale=1/16)
    bound = sparse_attention_allowance(query, records, selected, **options)
    result = sparse_attention_arithmetic(query, records, selected, **options).output.double()
    assert ((result - bound.expected).abs() <= bound.absolute).all()
    assert ((1.2 * result - bound.expected).abs() > bound.absolute).any()
    corrupted = records.clone()
    scales = corrupted[:, 512:528].contiguous().view(torch.float32) * 1.2
    corrupted[:, 512:528] = scales.view(torch.uint8)
    wrong = sparse_attention_arithmetic(query, corrupted, selected, **options).output.double()
    assert ((wrong - bound.expected).abs() > bound.absolute).any()
