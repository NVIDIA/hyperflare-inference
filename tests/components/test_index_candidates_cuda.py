# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Candidate blocks preserve causal prefixes, newest keys and logical addresses."""

import os

import pytest
import torch

from ignition.components.index_candidates import select_blocks, expand_blocks, remap_columns, mask_blocks
from ignition.preparation.compilation import compile_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA GPU")


@pytest.mark.parametrize("with_starts", [False, True])
def test_contiguous_candidate_mask_replay(with_starts):
    rows, width, count, block_tokens = 7, 2049, 2048, 8
    source = torch.randn((rows, width), device="cuda")
    output = torch.empty_like(source)
    candidates = torch.full((rows, count), -1, device="cuda", dtype=torch.int32)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, width)

    def forward():
        output.copy_(source)
        mask_blocks(output, ends, candidates, block_tokens=block_tokens,
                    starts=starts if with_starts else None)

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for step in range(3):
            source.neg_()
            starts.copy_(torch.tensor([0, 1, 7, 9, 17, 1024, 2048], device="cuda", dtype=torch.int32))
            if not with_starts:
                starts.zero_()
            ends.copy_(torch.tensor([0, 1, 17, 1025, 2048, 2049, 2049], device="cuda", dtype=torch.int32))
            candidates.fill_(-1)
            candidates[:, :4] = torch.tensor([0, 2, 17, 10000], device="cuda", dtype=torch.int32)
            if step == 1:
                candidates[:, :4].add_(1)
            elif step == 2:
                ends.copy_(starts)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                columns = torch.arange(width, device="cuda")
                ids = candidates[row, :4]
                keep = torch.isin((columns - start) // block_tokens, ids)
                if torch.any(start + ids * block_tokens >= width):
                    keep[-1] = True
                keep &= (columns >= start) & (columns < end)
                expected = source[row].masked_fill(~keep, -float("inf"))
                torch.testing.assert_close(output[row], expected, rtol=0, atol=0)
    finally:
        graph.reset()


@pytest.mark.parametrize("width,count,block_tokens,sparse_tokens", [
    (17, 2048, 8, 8), (20003, 2048, 8, 8), (40007, 2048, 16, 8), (40007, 2048, 16, 16),
])
def test_candidate_selection_expansion_and_replay(width, count, block_tokens, sparse_tokens):
    torch.manual_seed(1721)
    rows = 7
    logits = torch.randn((rows, width), device="cuda")
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    candidates = torch.empty((rows, count), dtype=torch.int32, device="cuda")
    blocks = torch.empty((rows, count * block_tokens // sparse_tokens), dtype=torch.int32, device="cuda")
    ends = torch.empty_like(lengths)
    columns = torch.empty((rows, 512), dtype=torch.int32, device="cuda")
    mapped = torch.empty_like(columns)

    def forward():
        selected = select_blocks(logits, lengths, count=count, block_tokens=block_tokens)
        expanded, end = expand_blocks(selected, lengths, block_tokens=block_tokens, sparse_tokens=sparse_tokens)
        col = torch.arange(512, device="cuda", dtype=torch.int32)[None, :].expand(rows, -1)
        col = torch.where(col < end[:, None], col, -1)
        candidates.copy_(selected)
        blocks.copy_(expanded)
        ends.copy_(end)
        columns.copy_(col)
        mapped.copy_(remap_columns(col, expanded, sparse_tokens=sparse_tokens))

    compiled = compile_forward(forward)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    try:
        for step in range(3):
            logits.neg_()
            lengths.copy_(torch.tensor([0, 1, 7, 9, width // 2, width - 1, width], device="cuda", dtype=torch.int32))
            if step == 2:
                lengths.zero_()
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            for row, length in enumerate(lengths.tolist()):
                scores = logits[row].clone()
                scores[length:] = -float("inf")
                padded = torch.nn.functional.pad(scores, (0, (-width) % block_tokens), value=-float("inf"))
                reduced = padded.view(-1, block_tokens).amax(-1)
                if length:
                    reduced[(length - 1) // block_tokens] = float("inf")
                top = reduced.topk(min(count, reduced.numel()))
                expected = top.indices[top.values != -float("inf")].int()
                actual = candidates[row][candidates[row] >= 0]
                assert torch.equal(actual.sort().values, expected.sort().values)
                assert actual.unique().numel() == actual.numel()
                if length:
                    assert (length - 1) // block_tokens in actual.tolist()
                tokens = sorted(token for block in actual.tolist()
                                for token in range(block * block_tokens, min(length, (block + 1) * block_tokens)))
                assert ends[row].item() == len(tokens)
                nblocks = (len(tokens) + sparse_tokens - 1) // sparse_tokens
                expected_blocks = sorted({token // sparse_tokens for token in tokens})
                assert blocks[row, :nblocks].tolist() == expected_blocks
                assert torch.all(blocks[row, nblocks:] == (expected_blocks[-1] if expected_blocks else 0))
                wanted = tokens[:512] + [-1] * max(0, 512 - len(tokens))
                assert mapped[row].tolist() == wanted
        invalid = torch.full_like(columns, blocks.shape[1] * sparse_tokens)
        assert torch.all(remap_columns(invalid, blocks, sparse_tokens=sparse_tokens) == -1)
    finally:
        graph.reset()
