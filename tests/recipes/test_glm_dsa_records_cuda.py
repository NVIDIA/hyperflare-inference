# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM MoE DSA record kernels on one CUDA device: the candidate merge, the context-parallel pulls,
the record writer, and selection compaction."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA device")

RECORD = 656


@torch.no_grad()
def test_record_writer_matches_the_reference_packing():
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference, unpack_mla_kv_cache_reference

    from ignition.recipes.glm_dsa.records import write_records

    device = torch.device("cuda", 0)
    torch.manual_seed(5)
    rows = 300
    latent = (torch.randn(rows, 512, device=device) * torch.rand(rows, 1, device=device) * 20).to(torch.bfloat16)
    # One all-zero group and one all-zero row exercise the unit scale of an empty group.
    latent[3, 128:256] = 0
    latent[7] = 0
    rotary = torch.randn(rows, 64, device=device).to(torch.bfloat16)
    slots = torch.randperm(4 * rows, device=device)[:rows]
    slots[11] = -1
    cache = torch.full((4 * rows // 64 + 1, 64, RECORD), 0x5A, dtype=torch.uint8, device=device)
    before = cache.clone()
    write_records(latent, rotary, slots, cache)
    torch.cuda.synchronize(device)
    expected = pack_mla_kv_cache_reference(latent.float(), rotary).squeeze(1)
    records = cache.view(-1, RECORD)
    live = slots >= 0
    assert torch.equal(records[slots[live]], expected[live])
    # A skipped row and every unaddressed record keep their bytes.
    untouched = torch.ones(records.shape[0], dtype=torch.bool, device=device)
    untouched[slots[live]] = False
    assert torch.equal(records[untouched], before.view(-1, RECORD)[untouched])
    restored = unpack_mla_kv_cache_reference(records[slots[live]]).squeeze(1)
    assert torch.equal(restored[:, 512:], rotary[live].float())
    error = (restored[:, :512] - latent[live].float()).abs().amax() / latent.float().abs().amax()
    assert error < 0.07


@torch.no_grad()
def test_pull_copies_marked_earlier_positions_from_their_owners():
    from ignition.recipes.glm_dsa.config import PAGE_RECORDS
    from ignition.recipes.glm_dsa.dcp import ContextShards, global_records, pull_marked
    from ignition.recipes.glm_dsa.records import write_records

    device = torch.device("cuda", 0)
    torch.manual_seed(21)
    world, pages, physical = 8, 3, 7
    positions = pages * PAGE_RECORDS * world
    table = torch.full((pages + 2,), -1, dtype=torch.int32, device=device)
    table[:pages] = torch.randperm(physical, device=device)[:pages].to(torch.int32)
    latent = torch.randn(positions, 512, device=device) * torch.rand(positions, 1, device=device) * 20
    latent, rotary = latent.to(torch.bfloat16), torch.randn(positions, 64, device=device).to(torch.bfloat16)
    index = torch.arange(positions, device=device)
    # Each context rank's page pool, on one device here, written as the rank writes the records it owns.
    pools = [torch.full((physical * PAGE_RECORDS, RECORD), 0x3C, dtype=torch.uint8, device=device)
             for _ in range(world)]
    for rank, pool in enumerate(pools):
        slots = ContextShards.on(rank, world, device).record_slots(index, table.expand(positions, -1))
        write_records(latent, rotary, slots, pool)
    addresses = torch.tensor([pool.data_ptr() for pool in pools], dtype=torch.int64, device=device)
    # The records every position would hold in a pool indexed by global position.
    expected = torch.full((positions, RECORD), 0xA5, dtype=torch.uint8, device=device)
    write_records(latent, rotary, index, expected)
    # Four rows at the last position select about a sixth of the context between them.
    selected = torch.randint(0, positions, (4, 2048), device=device, dtype=torch.int32)
    selected[:, 300:] = -1
    marks = torch.empty(positions + 1, dtype=torch.uint8, device=device)
    global_records(selected, torch.full((4,), positions - 1, device=device), marks=marks)
    slab = torch.full_like(expected, 0xA5)
    begin = positions - 700
    pull_marked(addresses, table, marks, torch.tensor([begin], dtype=torch.int64, device=device), slab,
                ContextShards.on(1, world, device))
    torch.cuda.synchronize(device)
    pulled = (marks[:positions] != 0) & (index < begin)
    # Some earlier positions carry a mark and some do not.
    assert 0 < pulled.sum() < begin
    assert torch.equal(slab[pulled], expected[pulled])
    assert (slab[~pulled] == 0xA5).all()
    # Nothing is read once every position belongs to the chunk.
    slab.fill_(0xA5)
    pull_marked(addresses, table, marks, torch.zeros(1, dtype=torch.int64, device=device), slab,
                ContextShards.on(0, world, device))
    torch.cuda.synchronize(device)
    assert (slab == 0xA5).all()


def _selection(rows, context, device, generator):
    """Sorted distinct positions padded with -1, as the indexer returns them, and each row's own position."""
    selected = torch.full((rows, 2048), -1, dtype=torch.int32)
    positions = torch.randint(0, context, (rows,), generator=generator)
    positions[0], positions[1] = 0, -1
    for row in range(rows):
        count = int(min(positions[row] + 1, torch.randint(0, 2049, (1,), generator=generator)))
        if count > 0:
            chosen = torch.randperm(int(positions[row]) + 1, generator=generator)[:count].sort().values
            selected[row, :count] = chosen.to(torch.int32)
    # One row lists positions past its own, which a draft row reusing another row's selection can hold.
    selected[2, :5] = torch.tensor([0, 1, int(positions[2]) + 1, int(positions[2]) + 2, context - 1], dtype=torch.int32)
    return selected.to(device), positions.to(device)


@pytest.mark.parametrize("first", [0, 1])
@torch.no_grad()
def test_single_rank_compaction_matches_the_position_by_position_reference(first):
    from ignition.recipes.glm_dsa.config import PAGE_RECORDS
    from ignition.recipes.glm_dsa.dcp import ContextShards
    from ignition.reference import glm_dsa as reference

    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(30 + first)
    rows, pages = 70, 6
    selected, positions = _selection(rows, pages * PAGE_RECORDS, device, generator)
    tables = torch.full((rows, pages + 2), -1, dtype=torch.int32)
    for row in range(rows):
        mapped = pages - int(row % 5 == 0)
        tables[row, :mapped] = torch.randperm(4 * pages, generator=generator)[:mapped].to(torch.int32)
    # Production builds the single-rank shards without a rank tensor, and the kernel then reads no rank.
    records, counts = ContextShards().local_records(selected, positions, tables.to(device), first=first)
    torch.cuda.synchronize(device)
    expected, expected_counts = reference.local_records(selected.cpu(), positions.cpu(), tables, first=first)
    assert torch.equal(records.cpu(), expected) and torch.equal(counts.cpu(), expected_counts)
    assert 0 < int(counts.sum()) and int((counts == 0).sum()) > 0


@pytest.mark.parametrize("world", [8])
@pytest.mark.parametrize("first", [0, 1])
@torch.no_grad()
def test_selection_compaction_matches_the_position_by_position_reference(world, first):
    from ignition.recipes.glm_dsa.config import PAGE_RECORDS
    from ignition.recipes.glm_dsa.dcp import ContextShards, global_records
    from ignition.reference import glm_dsa as reference

    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(29 + world + first)
    rows, pages = 70, 6
    context = pages * PAGE_RECORDS * world
    selected, positions = _selection(rows, context, device, generator)
    tables = torch.full((rows, pages + 2), -1, dtype=torch.int32)
    for row in range(rows):
        # The last page of some rows is unmapped, which drops the selections it covers.
        mapped = pages - int(row % 5 == 0)
        tables[row, :mapped] = torch.randperm(4 * pages, generator=generator)[:mapped].to(torch.int32)
    for rank in sorted({0, world - 1}):
        records, counts = ContextShards.on(rank, world, device).local_records(
            selected, positions, tables.to(device), first=first)
        torch.cuda.synchronize(device)
        expected, expected_counts = reference.local_records(
            selected.cpu(), positions.cpu(), tables, rank=rank, world=world, first=first)
        assert torch.equal(records.cpu(), expected) and torch.equal(counts.cpu(), expected_counts)
        assert 0 < int(counts.sum()) and int((counts == 0).sum()) > 0
    # One request's rows share a page-table row, as prefill metadata expands it.
    shared = tables[3:4].to(device).expand(rows, -1)
    records, counts = ContextShards.on(0, world, device).local_records(selected, positions, shared, first=first)
    expected, expected_counts = reference.local_records(
        selected.cpu(), positions.cpu(), tables[3:4].expand(rows, -1), world=world, first=first)
    assert torch.equal(records.cpu(), expected) and torch.equal(counts.cpu(), expected_counts)

    marks = torch.full((context + 1,), 7, dtype=torch.uint8, device=device)
    records, counts = global_records(selected, positions, first=first, marks=marks)
    torch.cuda.synchronize(device)
    expected_marks = torch.full((context + 1,), 7, dtype=torch.uint8)
    expected, expected_counts = reference.global_records(selected.cpu(), positions.cpu(), first=first,
                                                         marks=expected_marks)
    assert torch.equal(records.cpu(), expected) and torch.equal(counts.cpu(), expected_counts)
    # The discard entry belongs to the reference's scatter. Only positions carry marks.
    assert torch.equal(marks.cpu()[:context], expected_marks[:context])
    unmarked, _ = global_records(selected, positions, first=first)
    assert torch.equal(unmarked.cpu(), expected)


@pytest.mark.parametrize("world", [8])
@torch.no_grad()
def test_key_pull_copies_every_earlier_position_from_its_owner(world):
    from ignition.components.fp8_indexer import write_fp8_index_keys
    from ignition.recipes.glm_dsa.config import PAGE_RECORDS
    from ignition.recipes.glm_dsa.dcp import ContextShards, pull_keys

    device = torch.device("cuda", 0)
    torch.manual_seed(17 + world)
    pages, physical, page_bytes = 3, 7, 64 * 132
    positions = pages * PAGE_RECORDS * world
    table = torch.full((pages + 2,), -1, dtype=torch.int32, device=device)
    table[:pages] = torch.randperm(physical, device=device)[:pages].to(torch.int32)
    keys = (torch.randn(positions, 128, device=device) * torch.rand(positions, 1, device=device) * 9).to(torch.bfloat16)
    index = torch.arange(positions, device=device)
    # Each context rank's key pool, on one device here, written as the rank writes the keys it owns.
    pools = [torch.full((physical * PAGE_RECORDS // 64, page_bytes), 0x3C, dtype=torch.uint8, device=device)
             for _ in range(world)]
    for rank, pool in enumerate(pools):
        slots = ContextShards.on(rank, world, device).record_slots(index, table.expand(positions, -1))
        write_fp8_index_keys(keys, slots, pool, hadamard=False, power_of_two=False)
    addresses = torch.tensor([pool.data_ptr() for pool in pools], dtype=torch.int64, device=device)
    # The keys every position would hold in a pool indexed by global position.
    expected = torch.full((positions // 64, page_bytes), 0xA5, dtype=torch.uint8, device=device)
    write_fp8_index_keys(keys, index, expected, hadamard=False, power_of_two=False)
    slab = torch.full_like(expected, 0xA5)
    begin = positions - 700
    pull_keys(addresses, table, torch.tensor([begin], dtype=torch.int64, device=device), slab,
              ContextShards.on(1, world, device))
    torch.cuda.synchronize(device)

    def keys_of(pool):
        """Each key's values and scale, [positions, 132]."""
        return torch.cat((pool[:, :64 * 128].reshape(-1, 128), pool[:, 64 * 128:].reshape(-1, 4)), -1)

    assert torch.equal(keys_of(slab)[:begin], keys_of(expected)[:begin])
    assert (keys_of(slab)[begin:] == 0xA5).all()
    # Nothing is read once every position belongs to the chunk, and an unmapped page is skipped.
    slab.fill_(0xA5)
    pull_keys(addresses, table, torch.zeros(1, dtype=torch.int64, device=device), slab,
              ContextShards.on(0, world, device))
    table[1] = -1
    pull_keys(addresses, table, torch.tensor([positions], dtype=torch.int64, device=device), slab,
              ContextShards.on(0, world, device))
    torch.cuda.synchronize(device)
    unmapped = (index // world // PAGE_RECORDS) == 1
    assert torch.equal(keys_of(slab)[~unmapped], keys_of(expected)[~unmapped])
    assert (keys_of(slab)[unmapped] == 0xA5).all()


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("first", [0, 1])
@torch.no_grad()
def test_candidate_kernels_match_the_reference_merge(world, first):
    from ignition.recipes.glm_dsa.config import PAGE_RECORDS
    from ignition.recipes.glm_dsa.dcp import ContextShards
    from ignition.reference.glm_dsa import merged_records, pack_candidates

    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(41 + world + first)
    rows, pages = 37, 9
    context = pages * PAGE_RECORDS * world
    positions = torch.randint(0, context, (rows,), generator=generator)
    positions[0], positions[1], positions[2] = context - 1, -1, 5
    tables = torch.full((rows, pages + 2), -1, dtype=torch.int32)
    for row in range(rows):
        mapped = pages - int(row % 5 == 0)
        tables[row, :mapped] = torch.randperm(4 * pages, generator=generator)[:mapped].to(torch.int32)
    packed, expected_packed = [], []
    for rank in range(world):
        # A rank's selection: distinct local keys in arbitrary order with tied scores, padded with -1.
        selected = torch.full((rows, 2048), -1, dtype=torch.int32)
        for row in range(rows):
            owned = len(range(rank, int(positions[row]) + 1, world))
            count = min(owned, 2048)
            selected[row, :count] = torch.randperm(owned, generator=generator)[:count].to(torch.int32)
        scores = torch.randint(-300, 300, (rows, 2048), generator=generator).float() / 16
        packed.append(ContextShards.on(rank, world, device).pack_candidates(selected.to(device), scores.to(device)))
        expected_packed.append(pack_candidates(selected, scores, rank=rank, world=world))
    torch.cuda.synchronize(device)
    for kernel, reference in zip(packed, expected_packed, strict=True):
        assert torch.equal(kernel.cpu(), reference)
    gathered = torch.stack(packed, 1).contiguous()
    kept = 0
    for rank in range(world):
        records, counts = ContextShards.on(rank, world, device).merged_records(
            gathered, positions.to(device), tables.to(device), first=first)
        torch.cuda.synchronize(device)
        expected, expected_counts = merged_records(gathered.cpu(), positions, tables, rank=rank, first=first)
        assert torch.equal(records.cpu(), expected) and torch.equal(counts.cpu(), expected_counts)
        kept += int(counts[0])
    # Row 0 sees more than 2048 positions on every rank: the ranks keep 2048 between them, less slot
    # zero when it is among them and excluded, and less the candidates of its unmapped last page.
    assert 0 < kept <= 2048


@torch.no_grad()
def test_wrapped_kernels_declare_their_writes():
    """torch analyzes each wrapped kernel's reads and writes when a forward compiles."""
    from torch._higher_order_ops.triton_kernel_wrap import identify_accessed_tensors

    from ignition.recipes.glm_dsa import dcp

    device = torch.device("cuda", 0)

    def tensor(dtype, *shape):
        return torch.zeros(shape, dtype=dtype, device=device)

    world, rows, topk = 4, 3, 2048
    rank = tensor(torch.int64, 1)
    tables, positions = tensor(torch.int32, rows, 6), tensor(torch.int32, rows)
    cases = (
        (dcp._pack_candidates, dict(Selected=tensor(torch.int32, rows, topk), Scores=tensor(torch.float32, rows, topk),
                                    Packed=tensor(torch.int64, rows, topk), Rank=rank, WORLD=world, TOPK=topk),
         ["Packed"]),
        (dcp._merge_records, dict(Candidates=tensor(torch.int64, rows, world, topk), Positions=positions,
                                  Tables=tables, Output=tensor(torch.int32, rows, topk),
                                  Counts=tensor(torch.int32, rows), Rank=rank, table_stride=6, columns=6,
                                  WORLD=world, PAGE=256, FIRST=0, TOPK=topk, STEPS=11, LISTS=4),
         ["Output", "Counts"]),
        (dcp._pull_keys, dict(Addresses=tensor(torch.int64, world), Table=tables[0], Begin=rank,
                              Slab=tensor(torch.int32, 8, 2112), columns=6, WORLD=world, PAGE=256, TOKENS=16,
                              KERNEL_PAGE=64, KEY_WORDS=32, PAGE_WORDS=2112),
         ["Slab"]),
    )
    for kernel, arguments, written in cases:
        writes = identify_accessed_tensors(kernel, arguments, {}).read_writes.writes
        assert sorted(dependency.name for dependency in writes) == sorted(written)
