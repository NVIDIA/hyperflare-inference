# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row shards of a prefill chunk: ownership, per-rank metadata, convolution halos and history tails.

The convolution and commit references are the recipe's Torch arithmetic. Each rank computes them on
its halo-extended rows with its view of the metadata, exchanging edge rows as the column gather
does; the results must equal the whole chunk's bit for bit.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import collectives, row_shards
from ignition.components.paged_batch import Metadata
from ignition.components.row_shards import PreparedRouteGather, RowShards, row_sharding_fits
from ignition.recipes.inkling2.arithmetic import commit_history, short_convolution, stored_history

HALO, CHANNELS = 3, 8


def _metadata(capacity, lengths, starts):
    rows = len(lengths)
    m = Metadata.allocate(capacity, rows, 1, "cpu", max_query_tokens=capacity)
    m.set(torch.tensor(starts), torch.tensor(lengths, dtype=torch.int32), torch.arange(1, rows + 1),
          torch.zeros(rows, 1, dtype=torch.int32), torch.zeros(rows, capacity, dtype=torch.int64),
          torch.tensor([rows]))
    return m


def _shards(world, capacity):
    return [RowShards.of(rank, world, capacity, "cpu", halo=HALO) for rank in range(world)]


@pytest.mark.parametrize("world,capacity", [(2, 16), (4, 16), (4, 2048), (3, 9)])
def test_shards_partition_the_capacity_and_the_live_rows(world, capacity):
    shards = _shards(world, capacity)
    assert torch.equal(torch.cat([s.own_rows for s in shards]), torch.arange(capacity))
    assert [int(s.offset) for s in shards] == [rank * capacity // world for rank in range(world)]
    assert [int(s.predecessor) for s in shards] == [max(rank - 1, 0) for rank in range(world)]
    for live in (0, 1, capacity // world, capacity // world + 1, capacity - 1, capacity):
        m = _metadata(capacity, [live], [5])
        counts = [s.metadata(m).live_count for s in shards]
        assert all(count.dtype == m.live_counts.dtype and count.shape == (1,) for count in counts)
        assert [int(count) for count in counts] == [min(max(live - int(s.offset), 0), s.rows) for s in shards]
        assert sum(int(count) for count in counts) == live


@pytest.mark.parametrize("arguments", [(0, 1, 16), (4, 4, 16), (0, 4, 18), (0, 4, 8), (-1, 4, 16)])
def test_shards_reject_geometry_without_equal_shards_holding_a_halo(arguments):
    with pytest.raises(ValueError, match="row shards require"):
        RowShards.of(*arguments, "cpu", halo=HALO)


def _exchange(shards, extended, views):
    """Every rank's edge rows as the column gather delivers them, and each rank's placed history tail."""
    exchanged = torch.cat([s.edges(e, v) for s, e, v in zip(shards, extended, views, strict=True)], dim=1)
    return [s.place_edges(e, exchanged.clone(), v) for s, e, v in zip(shards, extended, views, strict=True)]


# One request per chunk, as prefill graphs hold, and packed requests, whose boundaries fall inside shards.
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("lengths,starts", [
    ([16], [0]), ([16], [40]), ([13], [7]), ([5], [0]), ([4], [3]), ([9], [0]), ([2], [0]), ([1], [6]), ([0], [0]),
    ([5, 9], [0, 12]), ([3, 2, 7], [4, 0, 1]), ([8, 8], [0, 0]), ([0, 6], [0, 2]), ([7, 0, 2], [1, 0, 9])])
def test_sharded_convolution_and_history_commit_match_the_whole_chunk(world, lengths, starts):
    capacity = 16
    generator = torch.Generator().manual_seed(1115 + world + sum(lengths))
    m = _metadata(capacity, lengths, starts)
    x = (torch.randint(-64, 65, (capacity, CHANNELS), generator=generator).float() / 16).bfloat16()
    weight = torch.randint(-8, 9, (CHANNELS, 1, HALO + 1), generator=generator).float() / 8
    history = (torch.randint(-64, 65, (len(lengths) + 1, HALO, CHANNELS), generator=generator).float() / 16).bfloat16()
    previous = stored_history(history, m)
    expected = short_convolution(x, weight, previous, m)
    expected_history = history.clone()
    commit_history(expected_history, previous, x, m)

    shards = _shards(world, capacity)
    views = [s.metadata(m) for s in shards]
    # The reduce-scatter delivers each rank its rows after the unwritten halo rows.
    extended = [torch.cat((torch.full((HALO, CHANNELS), float("nan"), dtype=x.dtype), x[s.own_rows])) for s in shards]
    tails = _exchange(shards, extended, views)
    for rank, (s, e, view, tail) in enumerate(zip(shards, extended, views, tails, strict=True)):
        if rank:
            assert torch.equal(e[:HALO], x[int(s.offset) - HALO:int(s.offset)])
        actual = short_convolution(e, weight, previous, view.convolution)
        assert torch.equal(actual[:HALO], torch.zeros_like(actual[:HALO]))
        assert torch.equal(actual[HALO:], expected[s.own_rows])
        assert tail.shape == (len(lengths) * HALO, CHANNELS) and tail.is_contiguous()
        committed = history.clone()
        commit_history(committed, previous, tail, view.tail)
        assert torch.equal(committed, expected_history)
    # Padding rows stay zero and requests without rows keep their history bytes.
    assert torch.equal(expected[sum(lengths):], torch.zeros_like(expected[sum(lengths):]))


def test_convolution_metadata_keeps_the_chunk_metadata_types():
    m = _metadata(16, [11], [3])
    view = RowShards.of(2, 4, 16, "cpu", halo=HALO).metadata(m)
    assert view.convolution.requests.dtype == m.requests.dtype and view.convolution.positions.dtype == m.positions.dtype
    assert view.convolution.query_start_loc.dtype == m.query_start_loc.dtype
    assert view.tail.query_start_loc.dtype == m.query_start_loc.dtype and view.tail.lengths.dtype == m.lengths.dtype
    assert view.convolution.positions.tolist() == [-1, -1, -1, 11, 12, 13, -1]
    assert view.convolution.query_start_loc.tolist() == [-5, 6]
    assert view.tail.lengths.tolist() == [3] and view.tail_rows.tolist() == [3, 4, 5]
    # Rows 8, 9 and 10 of the chunk belong to rank 2.
    assert view.tail_select.tolist() == [2, 6, 10]


class _Group:
    """Four ranks' gathers on one process: peers contribute a marked copy of this rank's shard."""

    world_size = 4

    def __init__(self, rank):
        self.rank = rank

    def gather_shards(self, source, *, out, plan):
        rows = source.shape[0]
        for peer in range(self.world_size):
            out[peer * rows:(peer + 1) * rows].copy_(source if peer == self.rank else source.flip(0))
        return out


class _ShardGather(collectives.PreparedShardGather):
    def __init__(self, source, output, *, runtime, session, coordinator, name):
        assert output.shape == (source.shape[0] * runtime.world_size, source.shape[1])
        assert source.dtype == output.dtype == torch.bfloat16 and source.shape[1] * 2 % 16 == 0
        self.runtime, self.output, self.single, self.plan = runtime, output, False, None
        self.handle = next(collectives._HANDLES)
        collectives._OWNERS[self.handle] = self


def test_route_gather_carries_ids_and_weights_bit_for_bit(monkeypatch):
    monkeypatch.setattr(row_shards, "PreparedShardGather", _ShardGather)
    shards = RowShards.of(1, 4, 32, "cpu", halo=HALO)
    gather = PreparedRouteGather(shards, 8, runtime=_Group(1), session=None, coordinator=None, name="routes")
    generator = torch.Generator().manual_seed(5)
    ids = torch.randint(-1, 384, (8, 8), generator=generator, dtype=torch.int32)
    weights = torch.rand((8, 8), generator=generator)
    weights[3, 2] = float("nan")
    gathered_ids, gathered_weights = gather(ids, weights)
    assert gathered_ids.shape == gathered_weights.shape == (32, 8)
    assert gathered_ids.dtype == torch.int32 and gathered_weights.dtype == torch.float32
    assert gathered_ids.is_contiguous() and gathered_weights.is_contiguous()
    assert torch.equal(gathered_ids[8:16], ids) and torch.equal(gathered_ids[:8], ids.flip(0))
    assert torch.equal(gathered_weights[8:16].view(torch.int32), weights.view(torch.int32))
    with pytest.raises(ValueError, match="even expert count"):
        PreparedRouteGather(shards, 7, runtime=_Group(1), session=None, coordinator=None, name="routes")


def test_single_rank_row_collectives_copy_their_source():
    source = torch.arange(24, dtype=torch.float32).view(6, 4).bfloat16()
    for component in (collectives.PreparedReduceScatter, collectives.PreparedShardGather):
        prepared = component(source, torch.zeros_like(source), runtime=None, session=None, coordinator=None, name="rows")
        assert torch.equal(prepared.functional(session=None)(source), source)
        with pytest.raises(ValueError, match="single-rank"):
            component(source, torch.zeros_like(source[:3]), runtime=None, session=None, coordinator=None, name="rows")


def test_sharding_requires_a_pipelined_roce_message_of_enough_rows(monkeypatch):
    accepted = []
    runtime = SimpleNamespace(world_size=4,
        should_reduce_scatter=lambda source: accepted.append(tuple(source.shape)) or True,
        should_gather_shards=lambda source: accepted.append(tuple(source.shape)) or True)
    partial = torch.zeros((2048, 16), dtype=torch.bfloat16)
    def fits(value, **changes):
        return row_sharding_fits(runtime, value, **{"halo": HALO, "minimum_rows": 2048, **changes})

    monkeypatch.setattr(row_shards, "_roce", lambda runtime: False)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)
    assert not fits(partial) and not row_sharding_fits(None, partial, halo=HALO, minimum_rows=2048)
    monkeypatch.setattr(row_shards, "_roce", lambda runtime: True)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: True)
    assert fits(partial) and accepted == [(2048, 16), (512, 16)]
    assert not fits(partial[:1024]) and not fits(partial[:2046], minimum_rows=1024)
    assert not fits(partial[:8], minimum_rows=8)
    runtime.should_reduce_scatter = lambda source: False
    assert not fits(partial)
