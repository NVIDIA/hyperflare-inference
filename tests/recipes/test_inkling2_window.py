# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 verification stages sliding-window rings and commits accepted rows byte for byte.

The references gather each request's whole ring, select in BF16 and rewrite the ring,
which states the staged pages and committed ring bytes independently of the kernels.
"""

import os

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.recipes.inkling2.attention import accepted_records, staged_lengths

KV, PAGE, WINDOW, RING, HEADS, DIM, WIDTH = torch.float8_e4m3fn, 128, 1024, 1152, 2, 128, 8
PAGES = -(-(WINDOW - 1 + WIDTH) // PAGE)
RECORD = HEADS * DIM
cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned SM120 GPU")


def reference_stage(rings, values, m):
    """FP8 pages `[requests * PAGES, PAGE, heads, dim]` of each request's window and rows, and key counts."""
    rows, capacity = m.starts.numel(), PAGES * PAGE
    ring = rings[m.state_rows].view(KV).view(rows, RING, *values.shape[1:])
    first = (m.starts - WINDOW + 1).clamp_min(0)
    positions = first[:, None] + torch.arange(capacity, device=values.device)
    previous = ring[torch.arange(rows, device=values.device)[:, None], positions.remainder(RING)].to(values.dtype)
    index = (m.query_start_loc[:-1, None] + positions - m.starts[:, None]).clamp(0, values.shape[0] - 1)
    staged = torch.where((positions < m.starts[:, None])[:, :, None, None], previous, values[index])
    lengths = torch.where(m.lengths > 0, m.starts - first + m.lengths, 0).int()
    staged = torch.where((positions < (m.starts + m.lengths)[:, None])[:, :, None, None], staged, 0)
    return staged.clamp(-448, 448).to(KV).view(rows * PAGES, PAGE, *values.shape[1:]), lengths


def reference_retain(rings, values, counts, m):
    """Rewrite the ring of every request that accepted rows, in place."""
    rows = m.starts.numel()
    ring = rings[m.state_rows].view(KV).view(rows, RING, *values.shape[1:])
    first = m.starts + counts - RING
    positions = first[:, None] + (torch.arange(RING, device=values.device) - first[:, None]).remainder(RING)
    indices = (m.query_start_loc[:-1, None] + positions - m.starts[:, None]).clamp(0, values.shape[0] - 1)
    current = values[indices].clamp(-448, 448)
    keep = ((positions < m.starts[:, None]) | (positions < 0))[:, :, None, None]
    retained = torch.where(keep, ring.to(values.dtype), current).to(KV).view(torch.uint8).view(rows, -1)
    selected = counts > 0
    rings[m.state_rows[selected]] = retained[selected]


def metadata(lengths, starts, rows, device="cpu"):
    m = Metadata.allocate(len(lengths) * WIDTH, len(lengths), 1, device, max_query_tokens=WIDTH)
    update(m, lengths, starts, rows)
    return m


def update(m, lengths, starts, rows):
    device, count = m.starts.device, m.starts.numel()
    m.set(torch.as_tensor(starts, device=device), torch.as_tensor(lengths, dtype=torch.int32, device=device),
          torch.as_tensor(rows, device=device), torch.zeros(count, 1, dtype=torch.int32, device=device),
          torch.zeros(count, WIDTH, dtype=torch.int64, device=device), torch.tensor([count], device=device))


def candidates(tokens, generator, device="cpu"):
    values = torch.randn(tokens, HEADS, DIM, generator=generator) * 4
    values.view(-1)[:6].copy_(torch.tensor([449., -449., 0., -0., 448., 1.0625]))
    return values.bfloat16().to(device)


def test_staged_lengths_count_the_preceding_window_and_the_request_rows():
    m = metadata([8, 3, 0, 8, 1], [0, 5, 77, 1023, 5000], [1, 2, 0, 3, 4])
    assert staged_lengths(m, WINDOW).tolist() == [8, 8, 0, 1031, 1024]
    values = candidates(m.positions.numel(), torch.Generator().manual_seed(1))
    rings = torch.zeros(5, RING * RECORD, dtype=torch.uint8)
    assert torch.equal(staged_lengths(m, WINDOW), reference_stage(rings, values, m)[1])


@pytest.mark.parametrize("starts", [[0, 1, 1144], [1151, 1152, 1153], [1016, 1023, 1024], [3499, 2297, 1048568]])
def test_accepted_records_commit_to_the_bytes_of_a_rewritten_ring(starts):
    generator = torch.Generator().manual_seed(sum(starts))
    lengths, rows = [8, 5, 8], [3, 1, 2]
    m = metadata(lengths, starts, rows)
    values = (candidates(m.positions.numel(), generator), candidates(m.positions.numel(), generator))
    initial = torch.randn(2, 5, RING, HEADS, DIM, generator=generator).to(KV).view(torch.uint8).view(2, 5, -1)
    records = accepted_records(*values, m)
    assert records.shape == (2, 3, WIDTH * RECORD) and records.dtype == torch.uint8
    for accepted in range(9):
        counts = torch.tensor([min(accepted, 8), min((accepted + 1) % 9, 5), 8 - accepted], dtype=torch.int32)
        for region in range(2):
            expected = initial[region].clone()
            reference_retain(expected, values[region], counts, m)
            actual = initial[region].clone().view(5, RING, RECORD)
            # The commit stores record `t` of a request at slot `(start + t) mod ring` for `t` below its count.
            for request, count in enumerate(counts.tolist()):
                for token in range(count):
                    actual[rows[request], (starts[request] + token) % RING] = records[region, request].view(
                        WIDTH, RECORD)[token]
            assert torch.equal(actual.view(5, -1), expected)


def _layer(batch, device, generator):
    """Continuation rows with K and V rings at distinct offsets, and the layer's prepared operations."""
    from ignition.components.ring_commit import PreparedRingCommit
    from ignition.components.window_staging import PreparedWindowStaging

    offsets = (256, 256 + RING * RECORD + 768)
    row_bytes = offsets[1] + RING * RECORD + 256
    backing = torch.randint(0, 256, (batch + 2, row_bytes), dtype=torch.uint8, generator=generator).to(device)
    for offset in offsets:
        ring = torch.randn(batch + 2, RING, HEADS, DIM, generator=generator).to(KV).view(torch.uint8)
        backing[:, offset:offset + RING * RECORD].copy_(ring.view(batch + 2, -1).to(device))
    rings = tuple(backing[:, offset:offset + RING * RECORD] for offset in offsets)
    m = metadata([WIDTH] * batch, [0] * batch, list(range(1, batch + 1)), device)
    tokens = batch * WIDTH
    key, value = (torch.zeros(tokens, HEADS, DIM, dtype=torch.bfloat16, device=device) for _ in range(2))
    staged = tuple(torch.zeros(batch * PAGES, PAGE, HEADS, DIM, dtype=KV, device=device) for _ in range(2))

    def build(session):
        shape = (-1, RING, HEADS, DIM)
        staging = PreparedWindowStaging(rings[0].view(KV).view(shape), rings[1].view(KV).view(shape), key, value,
            *staged, m, lookback=WINDOW - 1, session=session, name="staging").functional(session=session)
        commit = PreparedRingCommit(backing, offsets, m.state_rows, m.starts, token_capacity=WIDTH,
            window_tokens=RING, record_bytes=RECORD, candidate_offset=0).functional()
        return staging, commit

    return backing, rings, m, key, value, build


def _cases(batch, generator):
    """Request starts, row counts and state rows at sequence start, window and ring boundaries."""
    boundaries = [0, 1, 7, 1015, 1016, 1023, 1024, 1025, 1143, 1144, 1151, 1152, 1153, 2297, 3499, 1048568]
    for shift in range(4):
        starts = [boundaries[(shift * 5 + 3 * lane) % len(boundaries)] for lane in range(batch)]
        lengths = [int(torch.randint(0 if shift else 1, WIDTH + 1, (1,), generator=generator)) for _ in range(batch)]
        if shift == 0:
            lengths = [WIDTH] * batch
        rows = torch.randperm(batch, generator=generator).add(1).tolist()
        yield starts, lengths, rows


@cuda
@pytest.mark.parametrize("batch", [1, 8, 32])
@torch.no_grad()
def test_verification_staging_and_commit_match_the_whole_ring_references(batch):
    from ignition.preparation.tuning import RecordedPreparationSession

    device, generator = torch.device("cuda", 0), torch.Generator().manual_seed(600 + batch)
    backing, rings, m, key, value, build = _layer(batch, device, generator)
    counts = torch.zeros(batch, dtype=torch.int32, device=device)
    observed = tuple(torch.zeros(batch * PAGES, PAGE, HEADS, DIM, dtype=KV, device=device) for _ in range(2))
    with RecordedPreparationSession(device=device, checkpoint="inkling2-window-verification",
            workload_schema="window-verification", numerical_policy="exact", compile_workers=1) as session:
        staging, commit = build(session)
        session.tune(show_progress=False)

        def forward():
            staged = staging(key, value, metadata=m)
            for output, pages in zip(observed, staged):
                output.copy_(pages)
            commit(counts, staging=accepted_records(key, value, m), rows=m.state_rows, starts=m.starts)

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        for starts, lengths, rows in _cases(batch, generator):
            for accepted in ("none", "all", "depth", "random"):
                update(m, lengths, starts, rows)
                key.copy_(candidates(batch * WIDTH, generator, device))
                value.copy_(candidates(batch * WIDTH, generator, device))
                limit = torch.tensor(lengths, dtype=torch.int32)
                selected = {"none": torch.zeros_like(limit), "all": limit,
                            "depth": torch.minimum(limit, torch.arange(batch, dtype=torch.int32) % (WIDTH + 1)),
                            "random": (torch.rand(batch, generator=generator) * (limit + 1)).int()}[accepted]
                counts.copy_(selected.to(device))
                before, expected = backing.clone(), backing.clone()
                references = []
                for offset, values in zip((256, 256 + RING * RECORD + 768), (key, value)):
                    region = expected[:, offset:offset + RING * RECORD]
                    references.append(reference_stage(region, values, m))
                    reference_retain(region, values, counts, m)
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                live = (m.lengths > 0).repeat_interleave(PAGES)
                for output, (pages, counted) in zip(observed, references):
                    assert torch.equal(output.view(torch.uint8)[live], pages.view(torch.uint8)[live])
                    # A request without rows stages no keys; attention reads none of its pages.
                    assert not output.view(torch.uint8)[~live].any()
                    assert torch.equal(staged_lengths(m, WINDOW), counted)
                assert torch.equal(backing, expected)
                if accepted == "none":
                    assert torch.equal(backing, before)
        graph.reset()


@cuda
@torch.no_grad()
def test_successive_rounds_keep_rings_identical_across_ring_wraps():
    from ignition.preparation.tuning import RecordedPreparationSession

    batch = 4
    device, generator = torch.device("cuda", 0), torch.Generator().manual_seed(77)
    backing, rings, m, key, value, build = _layer(batch, device, generator)
    counts = torch.zeros(batch, dtype=torch.int32, device=device)
    with RecordedPreparationSession(device=device, checkpoint="inkling2-window-rounds",
            workload_schema="window-verification", numerical_policy="exact", compile_workers=1) as session:
        staging, commit = build(session)
        session.tune(show_progress=False)
        expected = backing.clone()
        starts, rows = [0, 1000, 1140, 2300], [4, 2, 1, 3]
        for _ in range(500):
            lengths = torch.randint(1, WIDTH + 1, (batch,), generator=generator).tolist()
            update(m, lengths, starts, rows)
            key.copy_(candidates(batch * WIDTH, generator, device))
            value.copy_(candidates(batch * WIDTH, generator, device))
            accepted = [int(torch.randint(0, length + 1, (1,), generator=generator)) for length in lengths]
            counts.copy_(torch.tensor(accepted, dtype=torch.int32))
            staged = staging(key, value, metadata=m)
            for offset, values, pages in zip((256, 256 + RING * RECORD + 768), (key, value), staged):
                region = expected[:, offset:offset + RING * RECORD]
                assert torch.equal(pages.view(torch.uint8), reference_stage(region, values, m)[0].view(torch.uint8))
                reference_retain(region, values, counts, m)
            commit(counts, staging=accepted_records(key, value, m), rows=m.state_rows, starts=m.starts)
            assert torch.equal(backing, expected)
            starts = [start + count for start, count in zip(starts, accepted)]
        # Every request has crossed its window; three have wrapped their rings.
        assert starts[0] > WINDOW and min(starts[1:]) > RING
