# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native ring commits preserve accepted bytes and compiler-visible lifetimes."""

import os

import pytest
import torch

from ignition.components.ring_commit import PreparedRingCommit
from ignition.preparation.compilation import compile_forward
from ignition.preparation.graphs import PreparedGraphBundle


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("batch,width,strided", [(1, 4, False), (4, 8, True), (64, 8, False), (1, 4096, False)])
@pytest.mark.parametrize("planes,alignment", [(None, 1), ((512, 16), 512)])
@torch.no_grad()
def test_compiled_ring_commits_share_state_and_preserve_producer_consumer_lifetimes(batch, width, strided, planes, alignment):
    device = torch.device("cuda", 0)
    torch.manual_seed(4190 + batch + width)
    regions, window, record = 40, 128, 528
    offsets = tuple(256 + layer * (window * record + 256) for layer in range(regions))
    backing = torch.full((batch + 3, offsets[-1] + window * record + 256), 0xA5,
                         dtype=torch.uint8, device=device)
    rows = torch.arange(1, batch + 1, dtype=torch.int64, device=device).flip(0)
    starts = (127 + 83 * torch.arange(batch, device=device)).long()
    counts = torch.zeros(batch, dtype=torch.int32, device=device)
    if strided:
        vectors = []
        for value, stride in ((rows, 2), (starts, 3), (counts, 2)):
            storage = torch.empty(batch * stride, dtype=value.dtype, device=device)
            view = storage[::stride]
            view.copy_(value)
            vectors.append(view)
        rows, starts, counts = vectors
    operations, sources, targets, functions, graphs = {}, {}, {}, {}, {}
    staged_bytes = ((window + width) * record + alignment - 1) // alignment * alignment
    staging = torch.empty((regions, batch, staged_bytes), dtype=torch.uint8, device=device)
    worker = None
    try:
        for capacity in (max(1, width // 2), width):
            operation = PreparedRingCommit(backing, offsets, rows, starts, token_capacity=capacity,
                window_tokens=window, record_bytes=record, staging=staging,
                record_planes=planes, staging_alignment=alignment)
            assert operation.staging.data_ptr() == staging.data_ptr()
            source = torch.randint(0, 256, operation.staging.shape, dtype=torch.uint8, device=device)
            target = operation.staging
            operation.functional()
            assert operation.staging is None and operation.rows is None and operation.starts is None

            def forward(operation=operation, source=source, target=target):
                target.copy_(source)
                operation(counts, staging=target, rows=rows, starts=starts)
                # Recycling follows the commit's last read in the same compiled forward.
                target.fill_(0xDE)

            function = compile_forward(forward)
            function()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                function()
            operations[capacity], sources[capacity] = operation, source
            targets[capacity] = target
            functions[capacity], graphs[str(capacity)] = function, graph
        bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"backing": backing},
            resources=(operations, sources, functions, rows, starts, counts))
        worker = bundle.start_worker()
        for iteration, capacity in enumerate((*operations, *reversed(operations))):
            source, operation = sources[capacity], operations[capacity]
            source.bitwise_xor_(37 + iteration)
            accepted = [([0, 1, min(3, capacity), capacity][(request + iteration) % 4])
                        for request in range(batch)]
            if batch == 1:
                accepted = [capacity if iteration != 2 else 1]
            counts.copy_(torch.tensor(accepted, dtype=torch.int32, device=device))
            rows.copy_(torch.arange(1, batch + 1, dtype=torch.int64, device=device).flip(0))
            if iteration == 3:
                rows[-1] = 0
            expected = backing.cpu()
            staged = source.cpu()
            row_ids, positions = rows.tolist(), starts.tolist()
            for region, destination in enumerate(offsets):
                for request, (row, start, length) in enumerate(zip(row_ids, positions, accepted)):
                    if row <= 0:
                        continue
                    for token in range(max(0, length - window), length):
                        prefix = 0
                        for plane in planes or (record,):
                            offset = destination + prefix * window + ((start + token) % window) * plane
                            origin = prefix * (window + capacity) + (window + token) * plane
                            expected[row, offset:offset + plane] = staged[region, request, origin:origin + plane]
                            prefix += plane
            originals = [value.clone() for value in (source, rows, starts, counts, operation.offsets)]
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            ticket = worker.submit(bundle.graph_indices[str(capacity)], repetitions=2)
            completion = worker.poll(timeout_ms=120000)
            assert completion is not None and completion.ticket == ticket and completion.error is None
            after = torch.cuda.memory_stats()
            for key in ("allocation.all.allocated", "allocation.all.freed",
                        "segment.all.allocated", "segment.all.freed"):
                assert before[key] == after[key], key
            assert torch.equal(backing.cpu(), expected)
            assert targets[capacity].eq(0xDE).all()
            for value, original in zip((source, rows, starts, counts, operation.offsets), originals, strict=True):
                assert torch.equal(value, original)
    finally:
        if worker is not None:
            worker.close()
        for graph in graphs.values():
            graph.reset()


@pytest.mark.parametrize("separated", [False, True])
@torch.no_grad()
def test_cache_writers_feed_batched_commits_before_staging_reuse(tmp_path, separated):
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
    from b12x.preparation import PreparationSession
    from ignition.components.compressed_mla import PreparedCompressedCacheWriter
    from ignition.components.mega_attention_cache import MegaCacheLayout, PreparedMegaCacheWriter

    if separated and torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("the separated cache writer requires SM103")

    device = torch.device("cuda", 0)
    torch.manual_seed(4148)
    regions, batch, width, window, record = 3, 4, 8, 128, 528
    offsets = tuple(256 + layer * (window * record + 256) for layer in range(regions))
    backing = torch.full((7, offsets[-1] + window * record + 256), 0xA5,
                         dtype=torch.uint8, device=device)
    rows = torch.tensor([3, 1, 5, 0], dtype=torch.int64, device=device)
    starts = torch.tensor([127, 255, 4095, 88], dtype=torch.int64, device=device)
    counts = torch.zeros(batch, dtype=torch.int32, device=device)
    operation = PreparedRingCommit(backing, offsets, rows, starts, token_capacity=width,
        window_tokens=window, record_bytes=record, record_planes=(512, 16) if separated else None,
        staging_alignment=512 if separated else 1)
    values = torch.randn((regions, batch * width, 512), dtype=torch.bfloat16, device=device) * .2
    slots = (torch.arange(batch, device=device)[:, None] * (window + width)
             + window + torch.arange(width, device=device)).reshape(-1).long()
    positions = torch.zeros(batch * width, dtype=torch.int64, device=device)
    cs = torch.cat((torch.ones((1, 32), device=device), torch.zeros((1, 32), device=device)), -1)
    graph = worker = None
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
            if separated:
                layout = MegaCacheLayout(window + width, record)
                caches = tuple(layout.view(operation.staging[layer]) for layer in range(regions))
                writers = tuple(PreparedMegaCacheWriter(values[layer], positions, cs, caches[layer], slots,
                    ratio=1, kernel_cache=tmp_path, session=session, name=f"window{layer}")
                    for layer in range(regions))
            else:
                writers = tuple(PreparedCompressedCacheWriter(values[layer], operation.staging[layer], slots,
                    page_size=window + width, kind="swa", session=session, name=f"window{layer}")
                    for layer in range(regions))

            def forward():
                for layer, writer in enumerate(writers):
                    if separated:
                        writer(values[layer], positions, cs, caches[layer], slots)
                    else:
                        writer(values[layer], slots)
                operation(counts, staging=operation.staging, rows=rows, starts=starts)
                operation.staging.fill_(0xDE)

            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            bundle = PreparedGraphBundle(device=0, graphs={"commit": graph}, buffers={"backing": backing},
                resources=(operation, values, slots, writers, compiled, session, rows, starts, counts))
            worker = bundle.start_worker()
            for accepted in ([0, 1, 3, 8], [8, 0, 1, 4]):
                values.neg_()
                counts.copy_(torch.tensor(accepted, dtype=torch.int32, device=device))
                expected = backing.cpu()
                for layer, destination in enumerate(offsets):
                    source = torch.zeros((batch, window + width, 512), dtype=torch.bfloat16, device=device)
                    source[:, window:].copy_(values[layer].view(batch, width, 512))
                    packed = pack_deepseek_v41_cache_reference(source.flatten(0, 1),
                        page_size=window + width, cache_kind="swa").cpu().reshape(batch, window + width, record)
                    for request, (row, start, count) in enumerate(zip(rows.tolist(), starts.tolist(), accepted)):
                        if row <= 0:
                            continue
                        for token in range(count):
                            prefix = 0
                            for plane in (512, 16) if separated else (record,):
                                offset = destination + prefix * window + ((start + token) % window) * plane
                                expected[row, offset:offset + plane] = packed[request, window + token, prefix:prefix + plane]
                                prefix += plane
                original = values.clone()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(bundle.graph_indices["commit"], repetitions=2)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key], key
                assert torch.equal(backing.cpu(), expected)
                assert operation.staging.eq(0xDE).all()
                assert torch.equal(values, original)
    finally:
        if worker is not None:
            worker.close()
        if graph is not None:
            graph.reset()
