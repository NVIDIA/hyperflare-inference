# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-token sparse-attention index keys and top-2048 selection for one layer.

A layer with an indexer scores every earlier token for each query row and
keeps the 2048 best; the following layers without an indexer attend to the
same records. Index keys are stored one per token in 64-record planar FP8
pages, four to a 256-token model page, addressed by the token's latent
record slot.

The checkpoint rotates the first 64 of the 128 index dimensions by position.
Queries and keys here hold those 64 dimensions last, where the shared rotary
operator rotates in place; scores are dot products over all 128 dimensions,
so the common reordering leaves them unchanged.
"""

import torch

from ignition.components.fp8_indexer import PreparedFP8Indexer
from ignition.components.glm_mla_cache import store_glm_index_key
from ignition.components.linear import UnquantizedProjection
from ignition.components.rotary import PreparedRotaryAngles

from .attention import PAGE_RECORDS, QUERY_RANK, TOPK, AttentionMetadata, ExampleBuffers, Selection

INDEX_HEADS, INDEX_WIDTH, INDEX_RECORD_BYTES = 32, 128, 132
# Query rows scored by one prefill selection call.
SELECT_ROWS = 128
_K_NORM_EPS = 1e-6


def index_page_table(block_table):
    """The four 64-record index pages of each model page, in order; unassigned pages stay -1."""
    pages = block_table[:, :, None] * 4 + torch.arange(4, dtype=block_table.dtype, device=block_table.device)
    return torch.where(block_table[:, :, None] >= 0, pages, -1).flatten(1)


def selected_records(logical, positions, pages):
    """Map selected logical positions to physical records in ascending token order.

    `pages` holds one index-page table row per query row. A selection is kept
    when it names a position at or before the row's own position on an
    assigned page. Sorting logical positions makes attention partitions
    independent of top-k enumeration and physical cache-page placement.
    Returns the records padded with -1 and the kept counts.
    """
    live = (logical >= 0) & (logical <= positions[:, None])
    page = pages.gather(1, (logical.clamp(min=0) // PAGE_RECORDS).long().clamp(max=pages.shape[1] - 1))
    live &= page >= 0
    records = torch.where(live, page * PAGE_RECORDS + logical % PAGE_RECORDS, -1)
    order = logical.masked_fill(~live, torch.iinfo(logical.dtype).max).sort(dim=1).indices
    return Selection(records.gather(1, order).to(torch.int32), live.sum(1, dtype=torch.int32))


class PreparedIndexer:
    """FP8 index queries and keys, FP32 head weights, and top-2048 selection."""

    def __init__(self, weights, query_residual, index_key, index_weights, metadata: AttentionMetadata, cache,
                 angles: PreparedRotaryAngles, *, mode, session, name, workspace,
                 buffers: ExampleBuffers | None = None, cache_only=False):
        rows, device = query_residual.shape[0], query_residual.device
        buffers = ExampleBuffers(device) if buffers is None else buffers
        if (query_residual.shape != (rows, QUERY_RANK) or index_key.shape != (rows, INDEX_WIDTH)
                or index_weights.shape != (rows, INDEX_HEADS)):
            raise ValueError("GLM 5.3 indexing requires the normalized query rank, index key, and head weights")
        if index_weights.dtype != torch.float32:
            raise ValueError("GLM 5.3 index head weights must retain FP32 projection precision")
        if weights["indexer.wq_b"].shape != (INDEX_HEADS * INDEX_WIDTH, QUERY_RANK):
            raise ValueError("GLM 5.3 index query projection geometry mismatch")
        if cache.dtype != torch.uint8 or cache.ndim != 2 or cache.shape[1] != PAGE_RECORDS * INDEX_RECORD_BYTES:
            raise ValueError("GLM 5.3 index cache requires planar 64-record uint8 pages")
        if mode not in ("decode", "prefill"):
            raise ValueError("GLM 5.3 indexing prepares decode or prefill")
        self.weights, self.cache, self.angles, self.mode, self.rows = weights, cache, angles, mode, rows
        self.projection, self.bounds, self.selectors = None, (), {}
        if cache_only:
            return
        query = buffers("indexer.query", rows, INDEX_HEADS * INDEX_WIDTH)
        self.projection = UnquantizedProjection(query_residual, weights["indexer.wq_b"], query,
                                                session=session, name=f"{name}.wq_b")
        width = metadata.block_table.shape[1] * 4
        step = SELECT_ROWS if mode == "prefill" else rows
        self.bounds = tuple((start, min(start + step, rows)) for start in range(0, rows, step))
        self.selectors = {}
        for start, end in self.bounds:
            count = end - start
            if count in self.selectors:
                continue
            pages = buffers("indexer.pages", 1 if mode == "prefill" else count, width, dtype=torch.int32)
            self.selectors[count] = PreparedFP8Indexer(
                query[start:end].view(count, INDEX_HEADS, INDEX_WIDTH),
                buffers("indexer.weights", count, INDEX_HEADS, dtype=torch.float32), cache,
                pages.expand(count, -1) if mode == "prefill" else pages,
                buffers("indexer.lengths", count, dtype=torch.int32), None,
                buffers("indexer.selected", count, TOPK, dtype=torch.int32), mode=mode, session=session,
                name=f"{name}.select{count}", workspace=workspace, topk=TOPK, prefill_records_per_call=131072,
                head_weight_scale=1.0, hadamard=False, key_power_of_two=False)

    def rebind_pages(self, cache):
        """Write and select through a replacement index page pool."""
        self.cache = cache
        for selector in self.selectors.values():
            selector.cache = cache
            selector.selection.rebind_cache(cache)

    def functional(self, *, session):
        if self.projection is not None:
            self.projection.functional(session=session)
        for selector in self.selectors.values():
            selector.functional(session=session)
        return self

    def __call__(self, query_residual, index_key, index_weights, metadata, angles):
        """Store this forward's index keys and select each row's records."""
        rows, device = query_residual.shape[0], query_residual.device
        query = self.projection(query_residual).view(rows, INDEX_HEADS, INDEX_WIDTH)
        query = torch.cat((query[:, :, 64:], query[:, :, :64]), dim=-1)
        self.angles.apply(query, angles)
        self.store(index_key, metadata, angles)
        head_weights = index_weights * (INDEX_HEADS ** -0.5)
        positions = metadata.positions[:rows]
        visible = (positions + 1).clamp(min=0).to(torch.int32)
        tables = index_page_table(metadata.block_table)
        row_pages = tables[metadata.requests[:rows].long()]
        logical = torch.empty((rows, TOPK), dtype=torch.int32, device=device)
        for start, end in self.bounds:
            count = end - start
            pages = tables[:1].expand(count, -1) if self.mode == "prefill" else row_pages[start:end].contiguous()
            logical[start:end].copy_(self.selectors[count](query[start:end], head_weights[start:end], pages,
                                                           visible[start:end], None, cache=self.cache))
        return selected_records(logical, positions, row_pages)

    def store(self, index_key, metadata, angles):
        """Normalize, rotate, and store this forward's index keys."""
        store_glm_index_key(index_key, self.weights["indexer.k_norm.weight"], self.weights["indexer.k_norm.bias"],
                            angles, metadata.token_slots[:index_key.shape[0]], self.cache, _K_NORM_EPS)
