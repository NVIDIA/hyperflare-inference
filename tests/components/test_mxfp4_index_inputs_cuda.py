# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Index producers preserve encoded values, scales, inactive slots and page padding."""

import os

import pytest
import torch

from ignition.components.mxfp4_index_inputs import quantize_index_queries, write_index_keys
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA GPU")


def encode(values):
    blocks = values.float().reshape(-1, 4, 32)
    maximum = blocks.abs().amax(-1).clamp_min(6 * 2. ** -126)
    exponent = torch.ceil(torch.log2(maximum / 6)).clamp(-127, 127)
    scaled = blocks * torch.exp2(-exponent[..., None])
    magnitude = scaled.abs()
    code = torch.zeros_like(scaled, dtype=torch.uint8)
    for edge, up in zip((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.),
                        (False, True, False, True, False, True, False)):
        code += (magnitude >= edge if up else magnitude > edge).to(torch.uint8)
    code |= torch.signbit(scaled).to(torch.uint8) << 3
    code = code.reshape(-1, 128)
    return code[:, ::2] | (code[:, 1::2] << 4), (exponent + 127).to(torch.uint8)


def rotate(values, positions, table, *, identity_prefix=False):
    result = values.float().clone()
    shape = (values.shape[0],) + (1,) * (values.ndim - 2) + (32,)
    c = table[positions, :32].view(shape)
    s = table[positions, 32:].view(shape)
    even, odd = values[..., 64::2].float(), values[..., 65::2].float()
    result[..., 64::2] = even * c - odd * s
    result[..., 65::2] = odd * c + even * s
    if identity_prefix:
        # The key kernel applies cos=1, sin=0 to the non-rotary pairs too.
        # Those operations affect signed zero, which FP4 preserves in its bytes.
        even, odd = values[..., :64:2].float(), values[..., 1:64:2].float()
        result[..., :64:2] = even * 1. - odd * 0.
        result[..., 1:64:2] = odd * 1. + even * 0.
    return result.to(torch.bfloat16)


def table():
    angles = torch.arange(4096, device="cuda", dtype=torch.float32)[:, None] * torch.linspace(.001, .2, 32, device="cuda")
    return torch.cat((angles.cos(), angles.sin()), -1)


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_query_quantization_matches_independent_encoding_under_replay(weight_dtype):
    torch.manual_seed(6201)
    rows = 257
    query = torch.randn((rows, 32, 128), device="cuda", dtype=torch.bfloat16)
    query[0].zero_()
    weights = torch.randn((rows, 32), device="cuda", dtype=torch.float32)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    cs = table()
    outputs = (torch.empty((rows, 32, 64), device="cuda", dtype=torch.int8),
               torch.empty((rows, 32), device="cuda", dtype=torch.int32), torch.empty_like(weights, dtype=weight_dtype))

    def forward():
        result = quantize_index_queries(query, positions, cs, weights, weight_dtype=weight_dtype)
        for out, value in zip(outputs, result):
            out.copy_(value)

    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for step in range(2):
            positions.add_(137)
            query.neg_()
            expected, sf = encode(rotate(query, positions, cs))
            folded = (weights * (128 ** -.5) * (32 ** -.5)).to(weight_dtype)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(outputs[0].view(torch.uint8).reshape(-1, 64), expected, rtol=0, atol=0)
            torch.testing.assert_close(outputs[1].view(torch.uint8).reshape(-1, 4), sf, rtol=0, atol=0)
            torch.testing.assert_close(outputs[2], folded, rtol=0, atol=0)
    finally:
        graph.reset()


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("eps", [1e-6, 1e-20])
def test_key_cache_normalization_rotation_and_masked_writes(ratio, eps):
    torch.manual_seed(7173)
    rows, page = 257, 128 // ratio
    width = (page * 68 + 511) // 512 * 512
    backing = torch.full((7, width), 0xAD, device="cuda", dtype=torch.uint8)
    cache = backing.as_strided((7, page, 68), (width, 68, 1))
    values = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16)
    values[0].zero_()
    values[1].mul_(1e-12)
    norm = torch.rand((128,), device="cuda", dtype=torch.float32) + .5
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    slots[::7] = -1
    cs = table()

    def forward():
        write_index_keys(values, positions, cs, norm, cache, slots, eps=eps, ratio=ratio)

    compiled = compile_forward(forward)
    compiled()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for step in range(2):
            backing.fill_(0xAD)
            positions.add_(131)
            slots.copy_(slots.flip(0))
            values.neg_()
            normalized = (values.float() * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + eps) * norm).to(torch.bfloat16)
            packed, scales = encode(rotate(normalized, positions // ratio * ratio, cs, identity_prefix=True))
            expected = backing.clone()
            live = (slots >= 0) & ((positions + 1) % ratio == 0)
            physical, offset = slots[live] // page, slots[live] % page
            expected[physical[:, None], offset[:, None] * 64 + torch.arange(64, device="cuda")] = packed[live]
            expected[physical[:, None], page * 64 + offset[:, None] * 4 + torch.arange(4, device="cuda")] = scales[live]
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            torch.testing.assert_close(backing, expected, rtol=0, atol=0)
    finally:
        graph.reset()
