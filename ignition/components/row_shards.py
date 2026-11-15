# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row shards of a prefill chunk: each rank runs row-local stages on the rows it owns.

Between two cross-rank reductions of a `[tokens, columns]` chunk, a tensor-parallel rank computes
stages whose result for a row depends on that row alone (normalization, mixtures, routing, residual
additions) or on a few preceding rows (causal convolutions). Replicated, every rank computes them for
every row. Sharded, rank `r` owns rows `[r * tokens / world, (r + 1) * tokens / world)`: a
reduce-scatter delivers it those rows of the sum, it computes the row-local stages for them, and a
shard gather reassembles all rows where a consumer needs them on every rank. The reduce-scatter and
the shard gather are the two halves of the pipelined all-reduce, whose bits they reproduce, and the
gathers copy bytes, so every rank's rows equal the replicated computation's.

A causal convolution reads `halo` rows before each row and commits each request's last `halo` raw
rows to replicated history. Ranks exchange those rows by one small column gather after each
reduce-scatter: every rank contributes the last `halo` rows it owns and its copy of each request's
last rows, takes its predecessor's rows as the halo before its own, and takes each request's last
rows from their owners.

Row ownership follows the chunk's capacity, not its live rows: a rank whose rows are padding
computes inactive rows, which stay zero and mutate no state. A rank's row offset reaches traced
forwards only through device tensors, so every rank compiles the same forward.
"""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllGather, PreparedReduceScatter, PreparedShardGather, _roce
from ignition.preparation.examples import declaration_tensor


@dataclass(frozen=True)
class ConvolutionRows:
    """Batch metadata of a causal convolution over the halo rows followed by this rank's rows."""

    requests: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    state_rows: torch.Tensor


@dataclass(frozen=True)
class TailRows:
    """Batch metadata of a history commit from `[requests * halo, columns]` rows: each request's last rows."""

    starts: torch.Tensor
    lengths: torch.Tensor
    query_start_loc: torch.Tensor
    state_rows: torch.Tensor


@dataclass(frozen=True)
class ShardMetadata:
    """One rank's view of a chunk's batch metadata.

    `tail_rows` indexes the halo-extended rows this rank contributes for each request's last rows;
    `tail_select` picks each of those rows from its owner in the exchanged `[requests * halo * world]` rows.
    """

    convolution: ConvolutionRows
    tail: TailRows
    live_count: torch.Tensor
    tail_rows: torch.Tensor
    tail_select: torch.Tensor


@dataclass(frozen=True)
class RowShards:
    """Equal contiguous row shards of a `tokens`-row chunk across `world` ranks, and this rank's rows."""

    world: int
    tokens: int
    halo: int
    own_rows: torch.Tensor
    offset: torch.Tensor
    predecessor: torch.Tensor

    @classmethod
    def of(cls, rank, world, tokens, device, *, halo):
        if (any(type(value) is not int for value in (rank, world, tokens, halo)) or world < 2
                or not 0 <= rank < world or halo < 1 or tokens % world or tokens // world < halo):
            raise ValueError("row shards require at least two ranks, a rank among them, and a chunk that divides "
                             "into shards of at least the halo's rows")
        rows = tokens // world
        return cls(world, tokens, halo,
            own_rows=torch.arange(rank * rows, (rank + 1) * rows, dtype=torch.int64, device=device),
            offset=torch.full((1,), rank * rows, dtype=torch.int64, device=device),
            # Rank zero has no predecessor and names itself: no row reads a halo before the chunk's first row.
            predecessor=torch.full((1,), max(rank - 1, 0), dtype=torch.int64, device=device))

    @property
    def rows(self):
        return self.tokens // self.world

    def metadata(self, m):
        """This rank's view of the batch metadata `m` of the whole chunk."""
        halo, rows = self.halo, self.rows
        device = self.own_rows.device
        offset = self.offset

        def extended(values):
            # Halo rows carry no position: the convolution reads them and computes nothing for them.
            return torch.cat((torch.full((halo,), -1, dtype=values.dtype, device=device),
                              values.index_select(0, self.own_rows)))

        starts = m.query_start_loc
        convolution = ConvolutionRows(extended(m.requests).clamp_min(0), extended(m.positions),
            (starts - offset + halo).to(starts.dtype), m.state_rows)
        requests = m.lengths.shape[0]
        steps = torch.arange(halo, dtype=torch.int64, device=device)
        # A request's last rows, or its first `halo` rows when it has fewer: the rows a commit reads.
        first = starts[:-1].long() + (m.lengths.long() - halo).clamp_min(0)
        last = (first[:, None] + steps).reshape(-1).clamp(0, self.tokens - 1)
        tail = TailRows(m.starts, m.lengths.clamp(max=halo),
            (torch.arange(requests + 1, dtype=torch.int64, device=device) * halo).to(starts.dtype), m.state_rows)
        return ShardMetadata(convolution, tail,
            live_count=(m.live_counts[:1] - offset).clamp(0, rows).to(m.live_counts.dtype),
            tail_rows=(last - offset).clamp(0, rows - 1) + halo,
            tail_select=torch.arange(requests * halo, dtype=torch.int64, device=device) * self.world
                        + torch.div(last, rows, rounding_mode="floor"))

    def edges(self, extended, shard):
        """The rows this rank contributes to the exchange: its last `halo` rows, then each request's last rows."""
        return torch.cat((extended[self.rows:], extended.index_select(0, shard.tail_rows)))

    def place_edges(self, extended, exchanged, shard):
        """Write the predecessor's rows before this rank's rows; return each request's last rows.

        `exchanged` holds every rank's `edges` as `[rows, world * columns]` in rank order.
        """
        columns = extended.shape[1]
        exchanged = exchanged.view(-1, self.world, columns)
        extended[:self.halo] = exchanged[:self.halo].index_select(1, self.predecessor).squeeze(1)
        return exchanged[self.halo:].reshape(-1, columns).index_select(0, shard.tail_select)


def row_sharding_fits(runtime, partial, *, halo, minimum_rows):
    """True when every rank can shard `partial`'s rows through `runtime` (rank-uniform)."""
    if runtime is None or not _roce(runtime):
        return False
    rows, world = partial.shape[0], runtime.world_size
    return (rows >= minimum_rows and rows % world == 0 and rows // world >= halo
            and PreparedReduceScatter.fits(runtime, partial)
            and PreparedShardGather.fits(runtime, partial[:rows // world]))


class RowShardCollectives:
    """The reduce-scatter, shard gather and edge exchange of one chunk capacity's `[tokens, columns]` BF16 rows."""

    def __init__(self, shards, columns, *, requests, runtime, session, coordinator, name, examples=None):
        device = shards.own_rows.device

        def example(role, rows, width=columns):
            return declaration_tensor(examples, "row_shards." + role, (rows, width), torch.bfloat16, device)

        self.shards, self.columns = shards, columns
        collectives = dict(runtime=runtime, session=session, coordinator=coordinator)
        self.reduce_scatter = PreparedReduceScatter(example("partial", shards.tokens), example("reduced", shards.rows),
            name=name + ".reduce_scatter", **collectives).functional(session=session)
        self.shard_gather = PreparedShardGather(example("shard", shards.rows), example("gathered", shards.tokens),
            name=name + ".shard_gather", **collectives).functional(session=session)
        edges = (1 + requests) * shards.halo
        self.exchange = PreparedAllGather(example("edges", edges), example("exchanged", edges, shards.world * columns),
            name=name + ".edge_exchange", **collectives).functional(session=session)

    def reduce(self, partial, shard):
        """This rank's rows of the reduced `partial` after their halo rows, and each request's last reduced rows."""
        shards = self.shards
        extended = torch.empty((shards.halo + shards.rows, self.columns), dtype=partial.dtype, device=partial.device)
        self.reduce_scatter(partial, extended[shards.halo:])
        return extended, shards.place_edges(extended, self.exchange(shards.edges(extended, shard)), shard)

    def gather(self, rows):
        """Every rank's `rows` in rank order."""
        return self.shard_gather(rows)


class PreparedRouteGather:
    """Gather every rank's top-k expert IDs and FP32 route weights of its rows as one message.

    Each row travels as the bytes of its int32 IDs followed by its FP32 weights, in BF16 lanes.
    """

    def __init__(self, shards, top_k, *, runtime, session, coordinator, name, examples=None):
        if type(top_k) is not int or top_k < 1 or top_k % 2:
            raise ValueError("gathered routes require an even expert count per row: whole 16-byte packs")
        device = shards.own_rows.device
        self.lanes = 2 * top_k

        def example(role, rows):
            return declaration_tensor(examples, "row_shards." + role, (rows, 2 * self.lanes), torch.bfloat16, device)

        self.shard_gather = PreparedShardGather(example("routes", shards.rows), example("gathered_routes", shards.tokens),
            runtime=runtime, session=session, coordinator=coordinator, name=name).functional(session=session)

    def __call__(self, ids, weights):
        if ids.dtype != torch.int32 or weights.dtype != torch.float32 or ids.shape != weights.shape:
            raise ValueError("gathered routes are matching int32 expert IDs and FP32 weights")
        gathered = self.shard_gather(torch.cat((ids.view(torch.bfloat16), weights.view(torch.bfloat16)), dim=1))
        return (gathered[:, :self.lanes].contiguous().view(torch.int32),
                gathered[:, self.lanes:].contiguous().view(torch.float32))
