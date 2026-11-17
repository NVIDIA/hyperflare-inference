# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed index gathering preserves bytes, page order and empty-request bounds."""

import os

import pytest
import torch

from ignition.components.mxfp4_index_gather import PreparedMXFP4IndexGather
from ignition.components.mxfp4_index_inputs import index_cache_view
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


class GatherForward:
    def __init__(self, cache, pages, offsets, *, capacity, session, kernel_cache):
        self.cache, self.pages, self.offsets = cache, pages, offsets
        self.gather = PreparedMXFP4IndexGather(cache, pages, offsets, capacity=capacity,
            kernel_cache=kernel_cache, session=session, name="index-gather")
        self.keys = torch.empty((capacity, 64), dtype=torch.int8, device=cache.device)
        self.scales = torch.empty((capacity,), dtype=torch.int32, device=cache.device)

    def __call__(self):
        keys, scales = self.gather(self.cache, self.pages, self.offsets)
        self.keys.copy_(keys)
        self.scales.copy_(scales)


@pytest.mark.parametrize("page_tokens", [64, 128])
def test_index_gather_worker_capture_and_page_changes(tmp_path, page_tokens):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    torch.manual_seed(1127)
    page_bytes = (page_tokens * 68 + 511) // 512 * 512
    backing = torch.randint(0, 256, (70, page_bytes), dtype=torch.uint8, device="cuda")
    cache = index_cache_view(backing, page_tokens)
    table = torch.randint(0, 70, (3, 70), dtype=torch.int32, device="cuda")
    offsets = torch.tensor([0, 1, 1, 8191], dtype=torch.int32, device="cuda")
    offsets[3] = min(8191, page_tokens * 70)
    with RecordedPreparationSession(device="cuda:0", checkpoint="synthetic-mxfp4-pages",
            workload_schema="index-cache-gather", numerical_policy="byte-exact",
            compile_workers=1, autotune_effort="off") as session:
        forward = GatherForward(cache, table, offsets, capacity=8192,
                                session=session, kernel_cache=tmp_path)
        session.tune()
        graphs, _ = capture_forwards({"gather": forward},
            {"gather": GraphSignature("prefill", 3, 8192)}, session=session,
            device=torch.device("cuda:0"), stream=torch.cuda.Stream(),
            pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path)
        graph = graphs["gather"]
        session.freeze()
        try:
            for lengths in ([1, 0, page_tokens * 31 + 7], [0, page_tokens + 1, 17], [0, 0, 0]):
                table.copy_(table.roll(1, 0).roll(3, 1))
                backing.bitwise_xor_(47)
                offsets[0] = 0
                offsets[1:].copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32).cumsum(0))
                original = backing.clone()
                forward.keys.fill_(0)
                forward.scales.fill_(0)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                assert torch.equal(backing, original)
                first = 0
                for request, length in enumerate(lengths):
                    index = torch.arange(length, device="cuda")
                    pages = table[request, index // page_tokens].long()
                    values = backing[:, :page_tokens * 64].view(70, page_tokens, 64)
                    scales = backing[:, page_tokens * 64:page_tokens * 68].view(70, page_tokens, 4)
                    expected_keys = values[pages, index % page_tokens].view(torch.int8)
                    expected_scales = scales[pages, index % page_tokens].contiguous().view(torch.int32).squeeze(-1)
                    assert torch.equal(forward.keys[first:first + length], expected_keys)
                    assert torch.equal(forward.scales[first:first + length], expected_scales)
                    first += length
            with pytest.raises(ValueError, match="prepared geometry"):
                forward.gather(cache, table.to(torch.int64), offsets)
        finally:
            graph.reset()
