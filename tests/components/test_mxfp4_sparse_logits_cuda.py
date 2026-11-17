# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Candidate scoring preserves reference arithmetic and request pairing under replay."""

import os

import pytest
import torch

from ignition.components.mxfp4_logits import PreparedSparsePagedMXFP4Logits
from ignition.components.megamoe_kernel import initialize
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def candidates(rows, count, block_tokens, lengths, step):
    raw = torch.arange(count, device="cuda", dtype=torch.int32)[None, :] * (step + 1)
    raw = raw + torch.arange(rows, device="cuda", dtype=torch.int32)[:, None] % 3
    valid = raw * block_tokens < lengths[:, None]
    last = torch.where(valid, raw, 0).amax(1)
    blocks = torch.where(valid, raw, last[:, None]).contiguous()
    n = valid.sum(1)
    end = torch.where(n > 0, (n - 1) * block_tokens + (lengths - last * block_tokens).clamp(0, block_tokens), 0)
    return blocks, end


@pytest.mark.parametrize("page,block_tokens,count", [(64, 8, 128), (128, 16, 128), (64, 8, 2048)])
def test_sparse_logits_reference_and_captured_transitions(page, block_tokens, count, tmp_path):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    reference = initialize(tmp_path)
    generator = torch.Generator(device="cuda").manual_seed(899 + page)
    columns, physical_pages = count * block_tokens * 2 + 3, 701
    page_bytes = (page * 68 + 511) // 512 * 512
    backing = torch.zeros((physical_pages, page_bytes), device="cuda", dtype=torch.uint8)
    backing[:, :page * 64].random_(0, 256, generator=generator)
    backing[:, page * 64:page * 68].random_(122, 129, generator=generator)
    cache = backing.as_strided((physical_pages, page, 68), (page_bytes, 68, 1))
    calls, data, outputs, graphs = {}, {}, {}, []
    with RecordedPreparationSession(device="cuda:0", checkpoint="generated-sparse-mxfp4-index",
            workload_schema="paged-sparse-mqa", numerical_policy="reference-bitwise",
            compile_workers=2, autotune_effort="off") as session:
        for rows in (1, 17, 257):
            query = torch.randint(-128, 128, (rows, 32, 64), device="cuda", dtype=torch.int8, generator=generator)
            scales = torch.randint(122, 129, (rows, 32, 4), device="cuda", dtype=torch.uint8, generator=generator)
            scales = scales.view(torch.int32).squeeze(-1)
            weights = torch.randn((rows, 32), device="cuda", dtype=torch.bfloat16, generator=generator)
            lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
            requests = torch.arange(rows, device="cuda", dtype=torch.int32) // 3
            request_pages = torch.randint(0, physical_pages, ((rows + 2) // 3, (columns + page - 1) // page),
                                         device="cuda", dtype=torch.int32, generator=generator)
            pages = request_pages[requests.long()].contiguous()
            blocks, _ = candidates(rows, count, block_tokens, lengths, 0)
            data[rows] = query, scales, cache, weights, lengths, pages, requests, blocks
            calls[rows] = PreparedSparsePagedMXFP4Logits(*data[rows], block_tokens=block_tokens,
                                                       kernel_cache=tmp_path, session=session, name=f"sparse-{rows}")
            outputs[rows] = torch.empty((rows, count * block_tokens), device="cuda", dtype=torch.bfloat16)
        session.tune()
        query, scales, cache, weights, lengths, pages, requests, blocks = data[1]
        zero_stride = pages.expand(2, -1)[:1]
        assert zero_stride.is_contiguous() and zero_stride.stride(0) == 0
        with pytest.raises(ValueError, match="matching int32 metadata"):
            PreparedSparsePagedMXFP4Logits(query, scales, cache, weights, lengths,
                zero_stride, requests, blocks, block_tokens=block_tokens,
                kernel_cache=tmp_path, session=session, name="invalid-singleton")
        for rows in data:
            def forward(rows=rows):
                outputs[rows].copy_(calls[rows](*data[rows]))
            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs.append((rows, graph))
        session.freeze()
        original = backing.clone()
        try:
            for step in range(3):
                for rows, graph in reversed(graphs):
                    query, scales, cache, weights, lengths, pages, requests, blocks = data[rows]
                    query.bitwise_xor_(31)
                    pages.copy_(pages.flip(1))
                    lengths.copy_((torch.arange(rows, device="cuda", dtype=torch.int32) * 131 + step * 23) % columns)
                    if step == 0:
                        lengths.fill_(columns)
                    elif step == 2:
                        lengths.zero_()
                    next_blocks, end = candidates(rows, count, block_tokens, lengths, step)
                    blocks.copy_(next_blocks)
                    metadata = reference.get_paged_sparse_mqa_logits_metadata(
                        lengths, pages, requests, page, blocks, query.dtype, block_tokens)
                    expected = reference.fp8_fp4_paged_sparse_mqa_logits(
                        (query[:, None], scales[:, None]), cache.unsqueeze(2), weights,
                        metadata, count, block_tokens)
                    outputs[rows].fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                    mask = torch.arange(count * block_tokens, device="cuda")[None, :] < end[:, None]
                    torch.testing.assert_close(outputs[rows][mask], expected[mask], rtol=0, atol=0)
            assert torch.equal(backing, original)
            with pytest.raises(ValueError, match="prepared geometry"):
                calls[257](*data[1])
        finally:
            for _, graph in graphs:
                graph.reset()
