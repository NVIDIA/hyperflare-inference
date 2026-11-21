# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel placement of GLM MoE DSA records against position-by-position arithmetic."""

import pytest
import torch

from ignition.recipes.glm_dsa.config import PAGE_RECORDS
from ignition.recipes.glm_dsa.dcp import ContextShards
from ignition.reference import glm_dsa as reference

TOP_K = 2048


def shards(rank, world):
    return ContextShards.on(rank, world, torch.device("cpu"))


def tables(pages, rows, *, seed=0):
    """Distinct physical pages for the first `pages` table columns of every row. Later columns are unmapped."""
    generator = torch.Generator().manual_seed(seed)
    table = torch.full((rows, pages + 3), -1, dtype=torch.int32)
    for row in range(rows):
        table[row, :pages] = torch.randperm(4 * pages, generator=generator)[:pages].to(torch.int32)
    return table


def test_every_position_has_one_owner_and_a_distinct_local_record():
    positions = torch.tensor([0, 1, 510, 511, 512, 513])
    # Rank r of two holds position 2l + r as local record l, in page 5 below 256 and in page 2 from 256 on.
    table = torch.tensor([[5, 2]], dtype=torch.int32).expand(positions.numel(), -1)
    assert shards(0, 2).record_slots(positions, table).tolist() == [1280, -1, 1535, -1, 512, -1]
    assert shards(1, 2).record_slots(positions, table).tolist() == [-1, 1280, -1, 1535, -1, 512]


@pytest.mark.parametrize("owner,lengths,expected", [
    # Rank 1 of four owns positions 1 and 5 among the first six.
    (shards(1, 4), [0, 1, 2, 5, 6], [0, 0, 1, 1, 2]),
    (ContextShards(), [0, 3], [0, 3]),
], ids=["rank-1-of-4", "one-rank"])
def test_local_tokens_count_owned_positions(owner, lengths, expected):
    assert owner.local_tokens(torch.tensor(lengths)).tolist() == expected


def test_ranks_partition_each_rows_visible_selection():
    selected = torch.tensor([[0, 1, 4, 511, 512, 513, 1024, 1025, 1601, -1]], dtype=torch.int32)
    positions = torch.tensor([1600])
    # Rank r of two holds position 2l + r as local record l: page 3 holds l below 256 and page 1 the next 256.
    # Positions 1024 to 1535 lie in the unmapped third page, and 1601 lies in page 6 after the row's own.
    table = torch.tensor([[3, 1, -1, 6]], dtype=torch.int32)
    records, counts = reference.local_records(selected, positions, table, rank=0, world=2)
    assert records.tolist() == [[768, 770, 256, -1, -1, -1, -1, -1, -1, -1]] and counts.tolist() == [3]
    records, counts = reference.local_records(selected, positions, table, rank=1, world=2)
    assert records.tolist() == [[768, 1023, 256, -1, -1, -1, -1, -1, -1, -1]] and counts.tolist() == [3]
    records, counts = reference.local_records(selected, positions, table, rank=0, world=2, first=1)
    assert records.tolist() == [[770, 256, -1, -1, -1, -1, -1, -1, -1, -1]] and counts.tolist() == [2]


def scored_keys(rows, context, *, seed=2):
    """Every position's score for each row, with ties, and each row's own position."""
    generator = torch.Generator().manual_seed(seed)
    # Few distinct values, some negative: many positions tie, and ties resolve to the lower position.
    scores = torch.randint(-40, 40, (rows, context), generator=generator).float() / 8
    positions = torch.randint(0, context, (rows,), generator=generator)
    positions[0], positions[1] = context - 1, TOP_K // 2
    return scores, positions


def local_selection(scores, positions, rank, world, *, generator):
    """One rank's selection among the keys it owns: its highest 2048 visible local keys in arbitrary order."""
    rows = scores.shape[0]
    selected = torch.full((rows, TOP_K), -1, dtype=torch.int32)
    values = torch.zeros((rows, TOP_K), dtype=torch.float32)
    for row in range(rows):
        owned = range(rank, int(positions[row]) + 1, world)
        order = sorted(owned, key=lambda p: (-float(scores[row, p]), p))[:TOP_K]
        order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
        selected[row, :len(order)] = torch.tensor(order, dtype=torch.int32) // world
        values[row, :len(order)] = scores[row, order]
    return selected, values


def test_merged_candidates_are_the_highest_scoring_positions():
    # Rank r of two selects local keys l, positions 2l + r. In row 0 the four highest are positions 3, 0, 1 and
    # 2, and position 4 loses the tie at 2.0 to the lower positions. In row 1 the four highest are positions 3,
    # 4, 0 and 1, so the cut falls between -2.0 at position 1 and -3.0 at position 2.
    selected = torch.tensor([[0, 1, 2, -1], [0, 1, 2, -1]], dtype=torch.int32)
    even = reference.pack_candidates(selected, torch.tensor([[3.0, 2.0, 2.0, 0.0], [-1.0, -3.0, 1.0, 0.0]]),
                                     rank=0, world=2)
    odd = reference.pack_candidates(selected, torch.tensor([[2.0, 5.0, -1.0, 0.0], [-2.0, 4.0, -4.0, 0.0]]),
                                    rank=1, world=2)
    candidates = torch.stack((even, odd), 1)
    positions, table = torch.tensor([5, 5]), torch.tensor([[7], [7]], dtype=torch.int32)
    records, counts = reference.merged_records(candidates, positions, table, rank=0)
    assert records.tolist() == [[1792, 1793, -1, -1], [1794, 1792, -1, -1]] and counts.tolist() == [2, 2]
    records, counts = reference.merged_records(candidates, positions, table, rank=1)
    assert records.tolist() == [[1793, 1792, -1, -1], [1793, 1792, -1, -1]] and counts.tolist() == [2, 2]
    records, counts = reference.merged_records(candidates, positions, table, rank=0, first=1)
    assert records.tolist() == [[1793, -1, -1, -1], [1794, -1, -1, -1]] and counts.tolist() == [1, 1]


def test_merged_candidates_drop_unmapped_pages_and_inactive_rows():
    world, pages = 4, 2
    context = pages * PAGE_RECORDS * world
    scores, positions = scored_keys(3, context, seed=11)
    positions[1], positions[2] = context - 3, -1
    table = tables(pages, 3)
    table[1, 1] = -1
    generator = torch.Generator().manual_seed(3)
    gathered = torch.stack([reference.pack_candidates(
        *local_selection(scores, positions, rank, world, generator=generator), rank=rank, world=world)
        for rank in range(world)], 1)
    for rank in range(world):
        records, counts = reference.merged_records(gathered, positions, table, rank=rank)
        mapped = int((records[1] >= 0).sum())
        # Row 1's candidates in its unmapped second page have no record.
        assert 0 < int(counts[1]) == mapped and (records[1, :mapped] // PAGE_RECORDS == table[1, 0]).all()
        assert int(counts[2]) == 0 and (records[2] == -1).all()


def test_global_records_keep_visible_positions_and_mark_them():
    selected = torch.tensor([[3, 7, 9, -1], [0, 5, -1, -1]], dtype=torch.int32)
    positions = torch.tensor([7, 0])
    # Every mark starts set, so a position that no row keeps must have its mark cleared.
    marks = torch.ones(11, dtype=torch.uint8)
    records, counts = reference.global_records(selected, positions, marks=marks)
    assert records.tolist() == [[3, 7, -1, -1], [0, -1, -1, -1]] and counts.tolist() == [2, 1]
    assert marks[:10].tolist() == [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    records, counts = reference.global_records(selected, positions, first=1)
    assert records.tolist() == [[3, 7, -1, -1], [-1, -1, -1, -1]] and counts.tolist() == [2, 0]


def test_context_shards_reject_a_missing_rank_tensor_and_a_rank_outside_the_world():
    with pytest.raises(ValueError, match="one-element int64 device tensor"):
        ContextShards(rank=1, world=8)
    with pytest.raises(ValueError, match="rank within a positive world"):
        ContextShards(rank=8, world=8, index=torch.tensor([8]))
