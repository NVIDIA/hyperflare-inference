# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The unchanged ds41 decode kernels must match FlashMLA under changing replay inputs.

FLARE_FLASHMLA_LIBRARY names a source build of vLLM-project/FlashMLA commit
0eee43b12f034b657133cf2afca6a72ebb6efccf's _flashmla_C shared library.
"""

import os

import numpy as np
import pytest
import torch

from ignition.components.mega_attention_kernel import prepare


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned SM103 GPU",
)


@pytest.fixture(scope="module")
def attention(tmp_path_factory):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    library = os.environ.get("FLARE_FLASHMLA_LIBRARY")
    if not library:
        pytest.skip("requires the pinned FlashMLA reference library")
    torch.ops.load_library(library)
    cache = tmp_path_factory.mktemp("mega-attention")
    return prepare(cache), cache


def packed_cache(source, page, kind):
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference

    width, payload = (528, 512) if kind == "swa" else (288, 256)
    records = pack_deepseek_v41_cache_reference(source, page_size=page, cache_kind=kind)
    records = records.reshape(-1, page, width)
    planes = torch.cat((records[:, :, :payload].flatten(1), records[:, :, payload:].flatten(1)), 1)
    # FlashMLA pages contain payload rows followed by scale rows, with padding
    # after both planes to satisfy the TMA page-stride alignment.
    stride = ((page * width + 511) // 512) * 512
    storage = torch.empty((records.shape[0], stride), dtype=torch.uint8, device="cuda")
    result = storage[:, :page * width].view(-1, page, width)
    result.copy_(planes.view_as(result))
    return result


@pytest.mark.parametrize("rows, window_page, compressed_page", [
    (1, 128, 64), (8, 128, 128), (16, 136, 64), (24, 128, 64),
    (64, 128, 128), (1, 128, 0), (8, 136, 0), (24, 128, 0),
])
def test_decode_matches_flashmla_with_changing_graph_inputs(attention, rows, window_page, compressed_page):
    attention, _ = attention
    gen = torch.Generator(device="cuda").manual_seed(711)

    def randn(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen) * .2

    window = packed_cache(randn((4 * window_page, 512)), window_page, "swa")
    comp = None if not compressed_page else packed_cache(
        randn((8 * compressed_page, 512)), compressed_page, "indexed")
    q = randn((rows, 64, 512))
    q = q.reshape(rows, 64, 32, 16).transpose(1, 2).contiguous().reshape_as(q)
    pos = torch.arange(2048, 2048 + rows, device="cuda", dtype=torch.int32)
    angles = torch.arange(8192, device="cuda").float()[:, None] / (
        10000 ** (torch.arange(32, device="cuda").float() / 32))[None, :]
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1)
    sink = randn((64,)).float()
    wi = torch.arange(128, 256, device="cuda", dtype=torch.int32).repeat(rows, 1)
    wl = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
    ci = None if comp is None else torch.stack([
        torch.randperm(512, generator=gen, device="cuda") for _ in range(rows)]).int()
    cl = None if comp is None else torch.full((rows,), 512, device="cuda", dtype=torch.int32)
    out = torch.empty((rows, 8, 4096), dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty((8, 32, (rows + 3) // 4 * 4), dtype=torch.int32, device="cuda").permute(2, 0, 1)[:rows]
    expected = torch.empty_like(out)
    expected_sf = torch.empty_strided(sf.shape, sf.stride(), dtype=sf.dtype, device="cuda")
    # Tagged-word scratch and epoch persist across sequential launches. The
    # exact profile uses parallel, cluster, and window-only modes respectively.
    words = rows * 10 * (128 + 128 + 2048 + 1)
    scratch = torch.zeros(2 * words, device="cuda")
    part_ml = torch.zeros(1, device="cuda")
    flags = torch.zeros(1, device="cuda", dtype=torch.int32)
    epoch = torch.zeros(2, device="cuda", dtype=torch.int32)
    mode = -2 if comp is None else (-1 if rows <= 16 else -3)
    nsub = (1 if 8 * rows <= 152 else (2 if 4 * rows <= 152 else 4)) if comp is None else 0
    scale = 512 ** -.5
    log2_scale = float(np.float32(scale) * np.float32(1.4426950408889634))

    def call():
        attention.w1_attn_decode(q, pos, cos_sin, sink, log2_scale, window, wi, wl,
                                 comp, ci, cl, out, sf, scratch, part_ml, flags, epoch,
                                 None, mode, nsub, True, 0, None)

    def reference():
        torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
            q, window.unsqueeze(-2), wi, scale, 512, sink, wl,
            None if comp is None else comp.unsqueeze(-2), ci, cl, False, 0., pos,
            False, 64, cos_sin, 8, 32, True, True, True, expected, expected_sf)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for step in range(4):
        q.copy_(randn(q.shape))
        pos.add_(1)
        wi.copy_(torch.roll(wi, 17, -1))
        wl.fill_(128)
        wl[0] = 17 if step % 2 else 128
        if ci is not None:
            ci.copy_(torch.roll(ci, 23, -1))
            cl.fill_(512)
            cl[0] = 61 if step % 2 else 512
        if step == 2:
            wl.zero_()
            if cl is not None:
                cl.zero_()
        reference()
        torch.cuda.synchronize()
        for _ in range(2):
            out.view(torch.uint8).fill_(255)
            sf.fill_(-999)
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
            assert torch.isfinite(out.float()).all()
            torch.testing.assert_close(out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(sf, expected_sf, rtol=0, atol=0)


def test_preparation_rejects_capture_even_with_cached_module(attention):
    _, cache = attention
    value = torch.zeros(1, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        value.add_(1)
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            prepare(cache)
