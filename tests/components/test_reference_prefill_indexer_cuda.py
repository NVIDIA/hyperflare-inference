# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiled prefill composes lossless gathering, FP32 scoring and local selection."""

import os

import pytest
import torch

from ignition.components.megamoe_kernel import initialize
from ignition.components.mxfp4_index_gather import PreparedMXFP4IndexGather
from ignition.components.mxfp4_index_inputs import index_cache_view, quantize_index_queries
from ignition.components.reference_indexer import PreparedReferenceIndexer
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


class PrefillIndexerForward:
    def __init__(self, backing, pages, offsets, query, weights, lengths, requests, mode, *, session, kernel_cache):
        rows, capacity = query.shape[0], pages.numel() * 128
        self.backing, self.pages, self.offsets = backing, pages, offsets
        self.query, self.weights, self.lengths, self.requests = query, weights, lengths, requests
        self.positions = torch.zeros(rows, dtype=torch.int64, device=query.device)
        self.cs = torch.cat((torch.ones((1, 32), device=query.device), torch.zeros((1, 32), device=query.device)), -1)
        self.mode = mode
        cache = index_cache_view(backing, 128)
        self.gather = PreparedMXFP4IndexGather(cache, pages, offsets, capacity=capacity,
            kernel_cache=kernel_cache, session=session, name="gather")
        keys = torch.empty((capacity, 64), dtype=torch.int8, device=query.device)
        scales = torch.empty((capacity,), dtype=torch.int32, device=query.device)
        self.selector = PreparedReferenceIndexer(query, backing, pages[requests], lengths,
            page_tokens=128, cos_sin=self.cs, candidate_mode=mode, candidate_count=2048,
            block_tokens=8, prefill=True, prefill_keys=(keys, scales), kernel_cache=kernel_cache,
            library=None, session=session, name="prefill-indexer")
        self.candidates = torch.full((rows, 2048), -1, dtype=torch.int32, device=query.device)
        self.output = torch.empty((rows, 512), dtype=torch.int32, device=query.device)
        self.blocks = torch.empty_like(self.candidates)
        self.counts = torch.empty_like(lengths)

    def __call__(self):
        keys = self.gather(index_cache_view(self.backing, 128), self.pages, self.offsets)
        starts = self.offsets[self.requests]
        result = self.selector(self.query, self.weights, self.pages[self.requests],
            starts + self.lengths, self.positions, self.requests, prefill_keys=keys,
            starts=starts, candidates=self.candidates if self.mode == "restrict" else None)
        if self.mode == "publish":
            result, blocks, counts = result
            self.blocks.copy_(blocks)
            self.counts.copy_(counts)
        self.output.copy_(result)


@pytest.mark.parametrize("rows,requests,page_count,mode", [
    (257, 3, 17, "none"), (257, 1, 157, "publish"),
    (257, 3, 17, "restrict"), (8192, 1, 17, "restrict"),
])
def test_prefill_indexer_reference_selection_and_replay(tmp_path, rows, requests, page_count, mode):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    reference = initialize(tmp_path)
    torch.manual_seed(1173)
    page_bytes = (128 * 68 + 511) // 512 * 512
    backing = torch.randint(0, 256, (page_count + 3, page_bytes), dtype=torch.uint8, device="cuda")
    backing[:, 128 * 64:128 * 68].random_(122, 129)
    pages = torch.randint(0, page_count + 3, (requests, page_count), dtype=torch.int32, device="cuda")
    length = page_count * 128 - 3
    offsets = torch.arange(requests + 1, device="cuda", dtype=torch.int32) * length
    query = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
    weights = torch.randn((rows, 32), device="cuda")
    lengths = torch.full((rows,), length, dtype=torch.int32, device="cuda")
    request_ids = (torch.arange(rows, device="cuda") * requests // rows).long()
    with RecordedPreparationSession(device="cuda:0", checkpoint="synthetic-prefill-index",
            workload_schema="reference-prefill-index", numerical_policy="exact-selection",
            compile_workers=2, autotune_effort="off") as session:
        forward = PrefillIndexerForward(backing, pages, offsets, query, weights, lengths, request_ids,
                                        mode, session=session, kernel_cache=tmp_path)
        session.tune()
        graphs, _ = capture_forwards({"prefill": forward},
            {"prefill": GraphSignature("prefill", requests, rows)}, session=session,
            device=torch.device("cuda:0"), stream=torch.cuda.Stream(),
            pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path)
        graph = graphs["prefill"]
        session.freeze()
        try:
            for step in range(3):
                query.neg_()
                weights.neg_()
                pages.copy_(pages.roll(1, 1))
                lengths.copy_((torch.arange(rows, device="cuda", dtype=torch.int32) * 131 + 713) % length)
                if step == 2:
                    lengths.zero_()
                    offsets.zero_()
                nblocks = (length + 7) // 8
                ids = torch.arange(2048, device="cuda", dtype=torch.int32) * 2
                forward.candidates.copy_(torch.where(ids < nblocks, ids, -1)[None, :].expand(rows, -1))
                # Independently decode page addresses and concatenate exact encoded bytes.
                payloads = backing[:, :128 * 64].reshape(-1, 128, 64)
                scale_bytes = backing[:, 128 * 64:128 * 68].reshape(-1, 128, 4)
                packed_keys, packed_scales = [], []
                for request in range(requests):
                    packed_keys.append(payloads[pages[request].long()].reshape(-1, 64)[:length])
                    packed_scales.append(scale_bytes[pages[request].long()].reshape(-1, 4)[:length])
                keys = torch.cat(packed_keys).contiguous().view(torch.int8)
                key_scales = torch.cat(packed_scales).contiguous().view(torch.int32).squeeze(-1)
                q, qs, folded = quantize_index_queries(query, forward.positions, forward.cs, weights,
                                                       weight_dtype=torch.float32)
                starts = offsets[request_ids]
                ends = starts + lengths
                scores = reference.fp8_fp4_mqa_logits((q, qs), (keys, key_scales), folded,
                                                      starts, ends, clean_logits=False)
                columns = torch.arange(length, device="cuda")[None, :].expand(rows, -1)
                local_scores = scores.gather(1, (columns + starts[:, None]).long())
                local_scores.masked_fill_(columns >= lengths[:, None], -float("inf"))
                if mode == "restrict":
                    local_scores.masked_fill_((columns // 8) % 2 != 0, -float("inf"))
                forward.output.fill_(-99)
                original = backing.clone()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                assert torch.equal(backing, original)
                valid = torch.arange(512, device="cuda")[None, :] < lengths[:, None].clamp(max=512)
                assert torch.all(forward.output[~valid] == -1)
                assert torch.all(forward.output[valid] >= 0)
                assert torch.all(forward.output[valid] < lengths[:, None].expand(-1, 512)[valid])
                found = local_scores.gather(1, forward.output.clamp_min(0).long()).masked_fill(~valid, -float("inf"))
                wanted = local_scores.topk(512, dim=-1).values
                torch.testing.assert_close(found.sort(-1, descending=True).values, wanted, rtol=0, atol=0)
                if mode == "publish":
                    padded = torch.nn.functional.pad(local_scores, (0, (-length) % 8), value=-float("inf"))
                    block_scores = padded.view(rows, -1, 8).amax(-1)
                    latest = ((lengths - 1).clamp_min(0) // 8).long()
                    block_scores[torch.arange(rows, device="cuda"), latest] = torch.where(
                        lengths > 0, float("inf"), -float("inf"))
                    top = block_scores.topk(min(2048, block_scores.shape[1]), dim=-1)
                    expected = torch.where(top.values != -float("inf"), top.indices.int(), -1)
                    expected = torch.nn.functional.pad(expected, (0, 2048 - expected.shape[1]), value=-1)
                    assert torch.equal(forward.blocks.sort(-1).values, expected.sort(-1).values)
                    assert torch.equal(forward.counts, (expected >= 0).sum(-1, dtype=torch.int32))
        finally:
            graph.reset()
