# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Context-parallel placement of GLM MoE DSA cache records.

Context ranks own single interleaved positions: with W context ranks, global
position p belongs to context rank p % W as local record p // W. Every rank
addresses its local records through the shared page tables: a page holds
PAGE_RECORDS local latent records of each layer and PAGE_RECORDS index keys of
each layer that owns an indexer, which cover PAGE_RECORDS * W global positions.

Decode selects among the keys a rank owns. Each rank packs its selection as
candidates, the ranks gather every rank's candidates, and each keeps its own
candidates among the 2048 highest. It then gathers every context rank's
absorbed query heads, attends the records it kept, and the ranks combine the
partial results by their log-sum-exp.

Prefill selects and attends with a rank's own heads over two slabs indexed by
global position. The key slab holds one layer's index keys: the chunk's own,
and every earlier position's, copied from the ranks that own them. The record
slab holds the chunk's latent records and the earlier positions its rows
selected, copied the same way.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from .config import INDEX_RECORD_BYTES, KERNEL_PAGE_RECORDS, LATENT_RECORD_BYTES, PAGE_RECORDS

# Positions one pull program scans. A program copies the marked ones in turn.
_PULL_TOKENS = 16
_RECORD_WORDS = LATENT_RECORD_BYTES // 4


@dataclass(frozen=True)
class ContextShards:
    """One rank's share of the context.

    Compiled forwards are shared by every rank, so the rank they use is a
    device tensor bound per rank rather than a traced constant. A world of one
    keeps the whole context on every rank.
    """

    rank: int = 0
    world: int = 1
    index: torch.Tensor | None = None

    def __post_init__(self):
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError("GLM MoE DSA context shards require a rank within a positive world")
        if self.world > 1 and (self.index is None or self.index.shape != (1,) or self.index.dtype != torch.int64):
            raise ValueError("GLM MoE DSA context shards require the rank as a one-element int64 device tensor")

    @classmethod
    def on(cls, rank, world, device):
        return cls(rank, world, torch.tensor([rank], dtype=torch.int64, device=device))

    @property
    def owner(self):
        return 0 if self.index is None else self.index

    def record_slots(self, positions, tables):
        """Local physical record of each global position, -1 where inactive, unmapped, or another rank's.

        A position's latent record and index key have the same slot in their layer's pools. `tables`
        holds the page-table row of every position.
        """
        local = positions.clamp_min(0) // self.world
        column = (local // PAGE_RECORDS).clamp(max=tables.shape[-1] - 1)
        page = tables.gather(-1, column[..., None]).squeeze(-1).long()
        live = (positions >= 0) & (positions % self.world == self.owner) & (page >= 0)
        return torch.where(live, page * PAGE_RECORDS + local % PAGE_RECORDS, -1)

    def index_pages(self, tables):
        """Map each page-table entry to its 64-key index pages, int32 [rows, width * PAGE_RECORDS / 64]."""
        per_page = PAGE_RECORDS // KERNEL_PAGE_RECORDS
        offsets = torch.arange(per_page, dtype=tables.dtype, device=tables.device)
        return torch.where(tables[..., None] >= 0, tables[..., None] * per_page + offsets, -1).flatten(-2).to(torch.int32)

    def local_tokens(self, lengths):
        """This rank's share of the first `lengths` global positions."""
        return lengths // self.world + (lengths % self.world > self.owner).to(lengths.dtype)

    def local_records(self, selected, positions, tables, *, first=0):
        """Local physical latent records among selected positions, compacted to the front, with each row's count.

        `selected` holds sorted global positions padded with -1, int32 [rows, 2048]. A selection before
        `first`, after the row's own position, in an unmapped page, or owned by another rank is dropped.
        """
        records = torch.empty_like(selected)
        counts = torch.empty((selected.shape[0],), dtype=torch.int32, device=selected.device)
        # One rank owns every position, and the kernel then reads no rank.
        _launch(_local_records, (selected.shape[0],),
                (selected, positions, tables, records, counts, positions if self.index is None else self.index,
                 tables.stride(0), tables.shape[1]),
                WORLD=self.world, PAGE=PAGE_RECORDS, FIRST=first, TOPK=selected.shape[1], num_warps=8)
        return records, counts

    def pack_candidates(self, selected, scores):
        """This rank's selection as candidates in descending order, int64 [rows, 2048].

        `selected` holds local key indices padded with -1, int32 [rows, 2048], and `scores` their FP32
        scores. A candidate orders by score, then by the lower global position, so no two are equal.
        An empty selection becomes a value below every candidate.
        """
        packed = torch.empty(selected.shape, dtype=torch.int64, device=selected.device)
        _launch(_pack_candidates, (selected.shape[0],), (selected, scores, packed, self.index),
                WORLD=self.world, TOPK=selected.shape[1], num_warps=16)
        return packed

    def merged_records(self, candidates, positions, tables, *, first=0):
        """Local physical latent records of this rank's candidates among the 2048 highest of every rank's.

        `candidates` holds every context rank's packed candidates, int64 [rows, world, 2048]. Returns the
        records compacted to the front in descending candidate order, and each row's count. A candidate
        before `first`, after the row's own position, or in an unmapped page is dropped after the merge.
        """
        rows, world, topk = candidates.shape
        if world != self.world or topk & (topk - 1) or world & (world - 1):
            raise ValueError("candidates require one descending list for each of a power of two context ranks, "
                             "a power of two long")
        records = torch.empty((rows, topk), dtype=torch.int32, device=candidates.device)
        counts = torch.empty((rows,), dtype=torch.int32, device=candidates.device)
        _launch(_merge_records, (rows,),
                (candidates, positions, tables, records, counts, self.index, tables.stride(0), tables.shape[1]),
                WORLD=world, PAGE=PAGE_RECORDS, FIRST=first, TOPK=topk, STEPS=topk.bit_length() - 1,
                LISTS=min(world, 4), num_warps=16)
        return records, counts


def global_records(selected, positions, *, first=0, marks=None):
    """Selected global positions from `first` through each row's own, compacted to the front, with each row's count.

    The positions address a slab that holds one record per global position. With `marks`, uint8
    [positions + 1] whose last entry is a discard, also set the mark of every position kept and clear
    every other mark.
    """
    records = torch.empty_like(selected)
    counts = torch.empty((selected.shape[0],), dtype=torch.int32, device=selected.device)
    if marks is not None:
        marks.zero_()
    # Rows that keep the same position store the same mark.
    _launch(_global_records, (selected.shape[0],),
            (selected, positions, records, counts, records if marks is None else marks),
            FIRST=first, TOPK=selected.shape[1], MARK=marks is not None, num_warps=8)
    return records, counts


@triton.jit
def _local_records(Selected, Positions, Tables, Output, Counts, Rank, table_stride, columns,
                   WORLD: tl.constexpr, PAGE: tl.constexpr, FIRST: tl.constexpr, TOPK: tl.constexpr):
    # One row's selected global positions as this rank's physical records, compacted to the front in
    # their selected order.
    row = tl.program_id(0)
    column = tl.arange(0, TOPK)
    selected = tl.load(Selected + row * TOPK + column).to(tl.int64)
    live = (selected >= FIRST) & (selected <= tl.load(Positions + row).to(tl.int64))
    if WORLD > 1:
        live &= selected % WORLD == tl.load(Rank)
    local = tl.maximum(selected, 0) // WORLD
    block = tl.minimum(local // PAGE, columns - 1)
    page = tl.load(Tables + row * table_stride + block, mask=live, other=-1).to(tl.int64)
    live &= page >= 0
    position = tl.cumsum(live.to(tl.int32), 0) - 1
    tl.store(Output + row * TOPK + column, tl.full((TOPK,), -1, tl.int32))
    tl.debug_barrier()
    tl.store(Output + row * TOPK + position, (page * PAGE + local % PAGE).to(tl.int32), mask=live)
    tl.store(Counts + row, tl.sum(live.to(tl.int32), 0))


@triton.jit
def _global_records(Selected, Positions, Output, Counts, Marks, FIRST: tl.constexpr, TOPK: tl.constexpr,
                    MARK: tl.constexpr):
    # One row's selected global positions from FIRST through its own, compacted to the front in their
    # selected order. MARK also flags each kept position.
    row = tl.program_id(0)
    column = tl.arange(0, TOPK)
    selected = tl.load(Selected + row * TOPK + column)
    live = (selected >= FIRST) & (selected.to(tl.int64) <= tl.load(Positions + row).to(tl.int64))
    position = tl.cumsum(live.to(tl.int32), 0) - 1
    tl.store(Output + row * TOPK + column, tl.full((TOPK,), -1, tl.int32))
    tl.debug_barrier()
    tl.store(Output + row * TOPK + position, selected, mask=live)
    tl.store(Counts + row, tl.sum(live.to(tl.int32), 0))
    if MARK:
        tl.store(Marks + selected, tl.full((TOPK,), 1, tl.uint8), mask=live)


@triton.jit
def _orderable(scores):
    # FP32 bits as int32 values that order as the scores do.
    bits = scores.to(tl.int32, bitcast=True)
    return tl.where(bits < 0, bits ^ 2147483647, bits)


@triton.jit
def _pack_candidates(Selected, Scores, Packed, Rank, WORLD: tl.constexpr, TOPK: tl.constexpr):
    # One row's selected local keys as candidates in descending order: by score, then by the lower
    # global position. An empty selection becomes the value below every candidate.
    row = tl.program_id(0)
    column = tl.arange(0, TOPK)
    local = tl.load(Selected + row * TOPK + column).to(tl.int64)
    score = tl.load(Scores + row * TOPK + column)
    position = local * WORLD + tl.load(Rank)
    keys = (_orderable(score).to(tl.int64) << 32) | (2147483647 - position)
    keys = tl.where(local >= 0, keys, -9223372036854775808)
    tl.store(Packed + row * TOPK + column, tl.sort(keys, descending=True))


@triton.jit
def _merge_records(Candidates, Positions, Tables, Output, Counts, Rank, table_stride, columns,
                   WORLD: tl.constexpr, PAGE: tl.constexpr, FIRST: tl.constexpr, TOPK: tl.constexpr,
                   STEPS: tl.constexpr, LISTS: tl.constexpr):
    # One row's own candidates among the TOPK highest of every rank's, as this rank's physical records
    # compacted to the front. Every rank's list is descending and no two candidates are equal, so a
    # candidate's place among all of them is the number above it in each list, found by bisection.
    row = tl.program_id(0)
    column = tl.arange(0, TOPK)
    lists = Candidates + row * WORLD * TOPK
    keys = tl.load(lists + tl.load(Rank) * TOPK + column)
    above = tl.zeros((TOPK,), tl.int32)
    for group in tl.static_range(WORLD // LISTS):
        # LISTS lists bisect side by side, so their reads overlap instead of following one another.
        entries = lists + (group * LISTS + tl.arange(0, LISTS))[:, None] * TOPK
        # A halving step adds its width to the count when the entry that far ahead is above the
        # candidate, and the last read tests the entry the steps leave. Each step refers to the count
        # twice: torch's access analysis of a wrapped kernel walks every dependency path of a read's
        # address, which a bisection over two bounds multiplies eightfold a step.
        count = tl.zeros((LISTS, TOPK), tl.int32)
        for step in tl.static_range(STEPS):
            width = TOPK >> (step + 1)
            count += tl.where(tl.load(entries + count + (width - 1)) > keys[None, :], width, 0)
        count += (tl.load(entries + count) > keys[None, :]).to(tl.int32)
        above += tl.sum(count, axis=0)
    selected = 2147483647 - (keys & 4294967295)
    live = ((keys != -9223372036854775808) & (above < TOPK) & (selected >= FIRST)
            & (selected <= tl.load(Positions + row).to(tl.int64)))
    local = selected // WORLD
    block = tl.minimum(local // PAGE, columns - 1)
    page = tl.load(Tables + row * table_stride + block, mask=live, other=-1).to(tl.int64)
    live &= page >= 0
    position = tl.cumsum(live.to(tl.int32), 0) - 1
    tl.store(Output + row * TOPK + column, tl.full((TOPK,), -1, tl.int32))
    tl.debug_barrier()
    tl.store(Output + row * TOPK + position, (page * PAGE + local % PAGE).to(tl.int32), mask=live)
    tl.store(Counts + row, tl.sum(live.to(tl.int32), 0))


@triton.jit
def _pull_keys(Addresses, Table, Begin, Slab, columns, WORLD: tl.constexpr, PAGE: tl.constexpr,
               TOKENS: tl.constexpr, KERNEL_PAGE: tl.constexpr, KEY_WORDS: tl.constexpr,
               PAGE_WORDS: tl.constexpr):
    # Copy the index keys of global positions before Begin into the slab at their positions, reading
    # each from its owner's pool: this rank's own memory or a peer's through CUDA IPC. A 64-key page
    # holds every key's values, then every key's scale.
    token = tl.program_id(0).to(tl.int64) * TOKENS + tl.arange(0, TOKENS).to(tl.int64)
    begin = tl.load(Begin).to(tl.int64)
    if tl.program_id(0).to(tl.int64) * TOKENS < begin:
        live = token < begin
        local = token // WORLD
        page = tl.load(Table + tl.minimum(local // PAGE, columns - 1), mask=live, other=-1).to(tl.int64)
        live &= page >= 0
        record = page * PAGE + local % PAGE
        source = tl.load(Addresses + token % WORLD).to(tl.pointer_type(tl.int32))
        source += record // KERNEL_PAGE * PAGE_WORDS
        target = Slab + token // KERNEL_PAGE * PAGE_WORDS
        word = tl.arange(0, KEY_WORDS)
        values = tl.load(source[:, None] + (record % KERNEL_PAGE)[:, None] * KEY_WORDS + word[None, :],
                         mask=live[:, None], other=0)
        tl.store(target[:, None] + (token % KERNEL_PAGE)[:, None] * KEY_WORDS + word[None, :], values,
                 mask=live[:, None])
        scales = tl.load(source + KERNEL_PAGE * KEY_WORDS + record % KERNEL_PAGE, mask=live, other=0)
        tl.store(target + KERNEL_PAGE * KEY_WORDS + token % KERNEL_PAGE, scales, mask=live)


@triton.jit
def _pull_marked(Addresses, Table, Marks, Begin, Slab, columns, WORLD: tl.constexpr, PAGE: tl.constexpr,
                 TOKENS: tl.constexpr, WORDS: tl.constexpr, WORD_BLOCK: tl.constexpr):
    # Copy the latent records of marked global positions before Begin into the slab at their positions, reading
    # each from its owner's cache: this rank's own memory or a peer's through CUDA IPC.
    first = tl.program_id(0) * TOKENS
    begin = tl.load(Begin).to(tl.int64)
    if first < begin:
        word = tl.arange(0, WORD_BLOCK)
        for offset in tl.range(0, TOKENS):
            token = first + offset
            if token < begin:
                if tl.load(Marks + token) != 0:
                    owner = token % WORLD
                    local = token // WORLD
                    page = tl.load(Table + tl.minimum(local // PAGE, columns - 1)).to(tl.int64)
                    if page >= 0:
                        source = tl.load(Addresses + owner).to(tl.pointer_type(tl.int32))
                        values = tl.load(source + (page * PAGE + local % PAGE) * WORDS + word, mask=word < WORDS)
                        tl.store(Slab + token * WORDS + word, values, mask=word < WORDS)


def _launch(kernel, grid, args, **meta):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **meta)
    else:
        kernel[grid](*args, **meta)


def pull_marked(addresses, table, marks, begin, slab, shards):
    """Copy the marked positions before `begin` from the ranks that own them into `slab` [positions, 656] uint8.

    `addresses` is every context rank's address of one layer's latent page pool, [world] int64, `table` the
    request's page-table row, and `begin` a one-element int64 tensor. Positions from `begin` on, the chunk's
    own, are written by the caller.
    """
    if slab.dtype != torch.uint8 or slab.shape[1:] != (LATENT_RECORD_BYTES,) or not slab.is_contiguous():
        raise ValueError("the context slab holds contiguous 656-byte records")
    if marks.dtype != torch.uint8 or marks.shape != (slab.shape[0] + 1,):
        raise ValueError("position marks cover the slab and one discard entry")
    _launch(_pull_marked, (triton.cdiv(slab.shape[0], _PULL_TOKENS),),
            (addresses, table, marks, begin, slab.view(torch.int32), table.shape[0]),
            WORLD=shards.world, PAGE=PAGE_RECORDS, TOKENS=_PULL_TOKENS, WORDS=_RECORD_WORDS,
            WORD_BLOCK=triton.next_power_of_2(_RECORD_WORDS), num_warps=4)


def pull_keys(addresses, table, begin, slab, shards):
    """Copy the index keys of every position before `begin` from the ranks that own them into `slab`.

    `slab` holds one layer's keys by global position as planar 64-key pages, [positions / 64, 64 * 132]
    uint8. `addresses` is every context rank's address of that layer's key pool, [world] int64, `table`
    the request's page-table row, and `begin` a one-element int64 tensor. Positions from `begin` on, the
    chunk's own, are written by the caller.
    """
    page_bytes = KERNEL_PAGE_RECORDS * INDEX_RECORD_BYTES
    if slab.dtype != torch.uint8 or slab.shape[1:] != (page_bytes,) or not slab.is_contiguous():
        raise ValueError("the key slab holds contiguous planar pages of 64 index keys")
    _launch(_pull_keys, (triton.cdiv(slab.shape[0] * KERNEL_PAGE_RECORDS, _PULL_TOKENS),),
            (addresses, table, begin, slab.view(torch.int32), table.shape[0]),
            WORLD=shards.world, PAGE=PAGE_RECORDS, TOKENS=_PULL_TOKENS, KERNEL_PAGE=KERNEL_PAGE_RECORDS,
            KEY_WORDS=(INDEX_RECORD_BYTES - 4) // 4, PAGE_WORDS=page_bytes // 4, num_warps=4)


class PrefillSlabs(NamedTuple):
    """One prefill forward's storage by global position, which every layer of the forward reuses.

    `records` holds one latent record per position, uint8 [positions, 656], and every layer rewrites
    the positions its rows attend. `marks` flags the positions the rows selected, uint8 [positions + 1]
    whose last entry is a discard. `keys` holds one index key per position as planar 64-key pages,
    uint8 [positions / 64, 64 * 132], and every layer that selects rewrites them.
    """

    records: torch.Tensor
    marks: torch.Tensor
    keys: torch.Tensor

    @classmethod
    def allocate(cls, positions, device):
        return cls(torch.empty((positions, LATENT_RECORD_BYTES), dtype=torch.uint8, device=device),
                   torch.empty((positions + 1,), dtype=torch.uint8, device=device),
                   torch.empty((positions // KERNEL_PAGE_RECORDS, KERNEL_PAGE_RECORDS * INDEX_RECORD_BYTES),
                               dtype=torch.uint8, device=device))


@dataclass(frozen=True)
class ContextParallel:
    """Context-parallel resources shared by every attention layer of a rank.

    `heads` is the b12x head exchange that gathers absorbed queries and combines partial attention, and
    `candidates` the gather of every context rank's packed selection. `peers` and `index_peers` hold
    every context rank's address of each layer's latent and index-key page pool ({layer: [world] int64}),
    and `slab_positions` is the global positions the prefill slabs hold, the prepared context.
    """

    shards: ContextShards
    heads: object
    candidates: object
    coordinator: object
    peers: dict
    index_peers: dict
    slab_positions: int

    def __post_init__(self):
        if self.slab_positions <= 0 or self.slab_positions % KERNEL_PAGE_RECORDS:
            raise ValueError("the context slab holds whole 64-record kernel pages")
        for addresses in (*self.peers.values(), *self.index_peers.values()):
            if addresses.shape != (self.shards.world,) or addresses.dtype != torch.int64:
                raise ValueError("context-parallel layers require every context rank's page-pool address")

