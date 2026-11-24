# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured graphs bind their cuBLASLt workspace from recorded storage.

Warmups and pre-capture callbacks run on a dedicated stream because
PyTorch caches one cuBLASLt workspace per (handle, stream). Keeping
their workspaces off the capture stream leaves the capture stream's
cache empty, so the first capture allocates its workspace inside the
recording pool and the cache retains it for every later call into the
pool. The test uses plain MemPools for transient storage and the
shared graph pool handle so it needs no package-export resources.
"""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


def _active_pool_blocks(pool):
    return {block["address"]: block
            for segment in torch.cuda.memory_snapshot(pool, include_traces=False)
            for block in segment["blocks"]
            if block["state"] == "active_allocated"}


def test_first_capture_binds_workspace_from_the_recording_pool():
    from ignition.preparation import capture as _capture
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.execution import GraphSignature

    torch.cuda.set_device(0)
    previous_blas = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        source = torch.randn((512, 512), device="cuda").bfloat16()
        weight = torch.randn((512, 512), device="cuda").bfloat16()
        expected = torch.mm(source, weight)
        output = torch.empty_like(source)

        def forward():
            torch.mm(source, weight, out=output)

        pool = torch.cuda.graph_pool_handle()
        session = SimpleNamespace(capture=nullcontext)

        transients = []

        def capture(stream, before_run=None):
            # Each call owns a separate transient pool, matching the
            # per-preparation lifetime of production transient storage.
            transient = torch.cuda.MemPool()
            transients.append(transient)
            storage = SimpleNamespace(
                transient=lambda: torch.cuda.use_mem_pool(transient))
            return capture_forwards(
                {"decode": forward}, {"decode": GraphSignature("decode", 1, 1)},
                session=session, device=0, stream=stream, pool=pool,
                warmup_storage=storage, compile_decoder=False, before_run=before_run)

        graphs = []
        stream = torch.cuda.Stream()
        first_call, _ = capture(stream)
        graphs.append(first_call)
        try:
            # The forward allocates nothing itself, so an active block in the
            # shared graph pool can only be the captured cuBLASLt workspace.
            # If warmups ran on the capture stream, the graph would keep the
            # transient warmup workspace's address and the pool would stay
            # empty.
            first = _active_pool_blocks(pool)
            assert first, ("no cuBLASLt workspace was recorded in "
                           "the captured graph pool")

            # Later calls into the same pool must not free the workspace
            # earlier graphs still reference, whether they capture on the
            # same or a different stream, and a before_run callback that
            # invokes BLAS must not contaminate the capture stream.
            same_stream, _ = capture(stream)
            graphs.append(same_stream)
            other_stream, _ = capture(torch.cuda.Stream(),
                                      before_run=lambda _: torch.mm(source, weight, out=output))
            graphs.append(other_stream)
            # Warmup workspaces are seeded in a dedicated pool owned by
            # the capture module, so no transient pool may hold a live
            # workspace allocation, and the seed must be verifiably
            # resident there rather than in any caller's storage.
            for transient in transients:
                assert not _active_pool_blocks(transient.id), (
                    "a warmup workspace remained in transient storage")
            seeded = _active_pool_blocks(_capture._warmup_pools[0].id)
            assert seeded, "no seeded workspace was recorded in the dedicated warmup pool"
            # The warmup stream must be outside torch.cuda.Stream's
            # round-robin pool: a later pool stream can alias an earlier
            # one, and a capture stream aliasing the warmup stream would
            # reintroduce the defect.
            warm = _capture._warmup_stream(0)
            pool_streams = [torch.cuda.Stream() for _ in range(40)]
            assert warm.cuda_stream not in {
                stream.cuda_stream for stream in pool_streams}, (
                "the warmup stream aliases a torch.cuda.Stream pool entry")
            remaining = _active_pool_blocks(pool)
            assert set(first) <= set(remaining), (
                "a later capture freed the workspace of an earlier graph")
            output.zero_()
            first_call["decode"].replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected)
        finally:
            for call in graphs:
                for graph in call.values():
                    graph.reset()
    finally:
        torch.backends.cuda.preferred_blas_library(previous_blas)
