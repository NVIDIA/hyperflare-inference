# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference prefill gathering follows physical pages and preserves encoded precision."""

import os

import pytest
import torch

from ignition.components.fused_mla import gather_cache
from ignition.components.mega_attention_cache import MegaCacheLayout
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA GPU")


@pytest.mark.parametrize("prefix,suffix", [(0, 0), (3, 11)], ids=["contiguous", "shared-buffer"])
@pytest.mark.parametrize("page,record", [(136, 528), (64, 288), (128, 288)])
def test_gathered_prefixes_match_independent_cache_decoder(page, record, prefix, suffix):
    from b12x.attention._shared.mla.compressed_reference import (
        pack_deepseek_v41_cache_reference, unpack_deepseek_v41_cache_reference,
    )
    layout = MegaCacheLayout(page, record)
    generator = torch.Generator(device="cuda").manual_seed(1129 + record)
    values = torch.randn((6 * page, 512), device="cuda", dtype=torch.bfloat16, generator=generator) * .2
    kind = "swa" if record == 528 else "indexed"
    interleaved = pack_deepseek_v41_cache_reference(values, page_size=page, cache_kind=kind)
    decoded = unpack_deepseek_v41_cache_reference(interleaved, page_size=page, cache_kind=kind).view(6, page, 512)
    records = interleaved.view(6, page, record)
    backing = torch.full((6, layout.page_bytes), 0xAD, device="cuda", dtype=torch.uint8)
    backing[:, :page * layout.payload_bytes].copy_(records[:, :, :layout.payload_bytes].flatten(1))
    backing[:, page * layout.payload_bytes:page * record].copy_(records[:, :, layout.payload_bytes:].flatten(1))
    cache = layout.view(backing)
    pages = torch.tensor([[4, 1], [3, 5], [2, 0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([page + 17, 1, 0], device="cuda", dtype=torch.int32)
    destination = torch.full((3, prefix + 2 * page + suffix, 512), -123,
                             device="cuda", dtype=torch.bfloat16)
    output = destination[:, prefix:prefix + 2 * page]

    def forward():
        gather_cache(cache, lengths, pages, output=output)

    function = compile_forward(forward)
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    try:
        original = backing.clone()
        for sizes in ([page + 17, 1, 0], [2 * page, page - 1, page + 1]):
            lengths.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
            pages.copy_(pages.flip(0))
            expected = decoded[pages.long()].flatten(1, 2).to(torch.bfloat16)
            destination.fill_(-123)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            for request, count in enumerate(sizes):
                torch.testing.assert_close(output[request, :count], expected[request, :count], rtol=0, atol=0)
                assert torch.all(output[request, count:] == -123)
            assert torch.all(destination[:, :prefix] == -123)
            assert torch.all(destination[:, prefix + 2 * page:] == -123)
            assert torch.equal(backing, original)
    finally:
        graph.reset()


@pytest.mark.parametrize("shape,strides,dtype", [
    ((2, 128, 512), (128 * 512, 512, 1), torch.float32),
    ((2, 128, 256), (128 * 256, 256, 1), torch.bfloat16),
    ((2, 128, 512), (128 * 1024, 1024, 2), torch.bfloat16),
    ((2, 128, 512), (0, 512, 1), torch.bfloat16),
    ((2, 128, 512), (2**31, 512, 1), torch.bfloat16),
], ids=["dtype", "channels", "strided-channels", "overlapping-requests", "offset-overflow"])
def test_gather_rejects_unsafe_destination_geometry(shape, strides, dtype):
    layout = MegaCacheLayout(128, 288)
    cache = layout.view(torch.empty((2, layout.page_bytes), device="meta", dtype=torch.uint8))
    pages = torch.empty((2, 1), device="meta", dtype=torch.int32)
    lengths = torch.empty(2, device="meta", dtype=torch.int32)
    output = torch.empty_strided(shape, strides, device="meta", dtype=dtype)
    with pytest.raises(ValueError, match="cache gather output requires"):
        gather_cache(cache, lengths, pages, output=output)
