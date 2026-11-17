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


@pytest.mark.parametrize("page,record", [(136, 528), (64, 288), (128, 288)])
def test_gathered_prefixes_match_independent_cache_decoder(page, record):
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
    output = torch.empty((3, 2 * page, 512), device="cuda", dtype=torch.bfloat16)

    def forward():
        output.copy_(gather_cache(cache, lengths, pages, capacity=2 * page))

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
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            for request, count in enumerate(sizes):
                torch.testing.assert_close(output[request, :count], expected[request, :count], rtol=0, atol=0)
            assert torch.equal(backing, original)
    finally:
        graph.reset()
