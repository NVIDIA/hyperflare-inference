# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Selected sparse records retain a causal token order independent of top-k enumeration."""

import torch

from ignition.recipes.glm53_full.indexer import selected_records


def test_selection_enumeration_cannot_change_attention_partitions():
    # Logical pages use nonmonotonic physical addresses. Ordering by physical
    # record would make the arithmetic depend on the cache allocator.
    pages = torch.tensor([[7, 2, 9], [4, -1, -1]], dtype=torch.int32)
    positions = torch.tensor([130, -1])
    logical = torch.tensor([[129, 3, 64, 65, -1, 180], [0, 2, -1, 1, 4, -1]], dtype=torch.int32)
    expected = torch.tensor([[7 * 64 + 3, 2 * 64, 2 * 64 + 1, 9 * 64 + 1, -1, -1],
                             [-1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    for permutation in (torch.arange(6), torch.arange(5, -1, -1), torch.tensor([3, 5, 0, 4, 1, 2])):
        selection = selected_records(logical[:, permutation], positions, pages)
        assert torch.equal(selection.records, expected)
        assert torch.equal(selection.lengths, torch.tensor([4, 0], dtype=torch.int32))


@torch.inference_mode()
def test_four_paged_verification_rows_match_independent_logical_order():
    import os
    import pytest
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1")
    generator = torch.Generator().manual_seed(53)
    rows, width, page_count = 4, 2048, 48
    positions = torch.tensor([2317, 2318, -1, 2320], dtype=torch.int64)
    logical = torch.stack([torch.randperm(2800, generator=generator)[:width] for _ in range(rows)]).int()
    logical[:, :6] = torch.tensor([0, 63, 64, 65, -1, 2799])
    pages = torch.stack([torch.randperm(page_count, generator=generator) for _ in range(rows)]).int()
    pages[:, 4] = -1
    expected = []
    for row in range(rows):
        keep = sorted(i for i in logical[row].tolist() if 0 <= i <= positions[row] and pages[row, i // 64] >= 0)
        expected.append([int(pages[row, i // 64]) * 64 + i % 64 for i in keep])
    def verify(result):
        records, length = (x.cpu() for x in result)
        for row, wanted in enumerate(expected):
            assert length[row].item() == len(wanted)
            assert records[row, :len(wanted)].tolist() == wanted
            assert records[row, len(wanted):].eq(-1).all()
    actual = selected_records(logical.cuda(), positions.cuda(), pages.cuda())
    verify(actual)
    stale = pages.roll(1, dims=1)
    with pytest.raises(AssertionError):
        verify(selected_records(logical.cuda(), positions.cuda(), stale.cuda()))
    wrong = logical.clone()
    wrong[0, 2] = 62
    with pytest.raises(AssertionError):
        verify(selected_records(wrong.cuda(), positions.cuda(), pages.cuda()))


@torch.inference_mode()
def test_prefill_rows_sharing_one_page_table_match_host_ordering():
    """The fused ordering kernel reads the stride-0 table view that the rows of a prefill forward share."""
    import os
    import pytest
    from ignition.components.sparse_record_order import ordered_sparse_records
    from ignition.recipes.glm53_full.attention import PAGE_RECORDS
    from ignition.recipes.glm53_full.indexer import index_tables

    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1")
    generator = torch.Generator().manual_seed(5371)
    rows, width, model_pages = 96, 2048, 12
    tokens = model_pages * 256
    block_table = torch.randperm(40, generator=generator)[:model_pages].int()[None]
    block_table[0, [3, 7]] = -1
    logical = torch.stack([torch.randperm(tokens, generator=generator)[:width] for _ in range(rows)]).int()
    logical[:, ::97] = -1
    positions = torch.randint(0, tokens, (rows,), generator=generator)
    positions[::11] = -1
    requests = torch.zeros(rows, dtype=torch.int64)
    # The host path of `selected_records` is the reference; it shares no code with the kernel.
    host = selected_records(logical, positions, index_tables(block_table, requests, mode="prefill")[1])
    causal = ((logical >= 0) & (logical <= positions[:, None])).sum(1)
    assert host.lengths[::11].eq(0).all() and host.lengths.max() > 1024
    assert (host.lengths < causal).any(), "some selected positions must lie on unassigned pages"

    device = torch.device("cuda")
    def shared_pages(table):
        tables, pages = index_tables(table.to(device), requests.to(device), mode="prefill")
        assert pages.stride(0) == 0 and pages.untyped_storage().data_ptr() == tables.untyped_storage().data_ptr()
        return pages

    def verify(pages):
        records, lengths = ordered_sparse_records(logical.to(device), positions.to(device), pages, PAGE_RECORDS)
        assert torch.equal(lengths.cpu(), host.lengths)
        assert torch.equal(records.cpu(), host.records)

    verify(shared_pages(block_table))
    dispatched = selected_records(logical.to(device), positions.to(device), shared_pages(block_table))
    assert torch.equal(dispatched.records.cpu(), host.records) and torch.equal(dispatched.lengths.cpu(), host.lengths)
    with pytest.raises(AssertionError):
        verify(shared_pages(block_table.roll(1, dims=1)))


@torch.inference_mode()
def test_captured_index_tables_follow_reassigned_request_pages():
    import os
    import pytest
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.glm53_full.decoder import attention_metadata

    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        pytest.skip("requires FLARE_RUN_CUDA_TESTS=1")
    device = torch.device("cuda")
    packed = Metadata.allocate(4, 2, 12, device)
    packed.requests.copy_(torch.tensor([1, 0, 1, 0], device=device))
    packed.positions.copy_(torch.tensor([2456, 2560, -1, 2721], device=device))
    logical = torch.tensor([0, 63, 64, 255, 256, 2048, 2311, 2456, 2705, -1], device=device,
                           dtype=torch.int32).expand(4, -1).contiguous()
    tables = torch.arange(24, dtype=torch.int32, device=device).reshape(2, 12)
    packed.page_table.copy_(tables)

    def select():
        placement = attention_metadata(packed)
        return selected_records(logical, placement.positions, placement.index_tables[1])

    select()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = select()
    torch.cuda.current_stream().wait_stream(stream)

    def expected(table):
        records = torch.full((4, 10), -1, dtype=torch.int32)
        lengths = torch.zeros(4, dtype=torch.int32)
        for row, (request, limit) in enumerate(zip(packed.requests.tolist(), packed.positions.tolist())):
            positions = sorted(i for i in logical[row].tolist() if 0 <= i <= limit)
            lengths[row] = len(positions)
            records[row, :len(positions)] = torch.tensor(
                [int(table[request, i // 256]) * 256 + i % 256 for i in positions], dtype=torch.int32)
        return records, lengths

    previous = None
    for table in (tables, tables.roll(3, dims=1).flip(0), tables):
        packed.page_table.copy_(table)
        graph.replay()
        actual = tuple(t.cpu() for t in result)
        wanted = expected(table.cpu())
        assert all(torch.equal(x, y) for x, y in zip(actual, wanted, strict=True))
        if previous is not None:
            assert not torch.equal(previous, wanted[0])
        previous = actual[0]
