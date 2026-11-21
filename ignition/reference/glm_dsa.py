# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded reference for GLM MoE DSA attention as the prepared recipe decomposes it.

The canonical Transformers module expands every cached position into per-head
keys and values and masks the positions its indexer did not select. The
prepared recipe never expands the cache. It stores one latent record per
position, moves the key expansion into the query (absorption), attends the
selected records in latent space, and expands only the attended result into
per-head values:

    q_content[h] . (W_key[h] @ latent) == (W_key[h]^T @ q_content[h]) . latent
    sum_t p[h, t] * (W_value[h] @ latent[t]) == W_value[h] @ sum_t p[h, t] * latent[t]

Rotary pairs stay interleaved in queries and records. Transformers
de-interleaves them, which permutes both operands of each dot product alike.

Context parallelism gives rank `position % world` each position's records as
local record `position // world`. A decode row gathers every rank's query
heads, each rank attends the selected records it owns, and the partial results
combine by their log-sum-exp. `local_records`, `pack_candidates`,
`merged_records`, `global_records` and `mark_positions` state in torch what the
record selection kernels of `ignition.recipes.glm_dsa.dcp` compute from integer
positions and packed candidates.

The attention functions run in FP32 on any device and take plain tensors and a
canonical attention module's weights. Every function allocates freely and is
not serving code.
"""

import torch

from ignition.recipes.glm_dsa.config import LATENT_RANK, PAGE_RECORDS, ROTARY_DIM

# A candidate below every selected key: the place of an empty selection.
_MISSING = -(1 << 63)


def rotary_angles(positions, base, dim=ROTARY_DIM):
    """Cosine and sine of every position's `dim // 2` rotary angles."""
    frequencies = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim))
    angles = positions.to(torch.float32)[:, None] * frequencies
    return angles.cos(), angles.sin()


def rotate(values, cos, sin):
    """Rotate interleaved pairs `(a, b)` of the last axis to `(a cos - b sin, a sin + b cos)`.

    `values` holds one row per position, optionally with a head axis: [rows, dim] or [rows, heads, dim].
    """
    if values.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    a, b = values[..., 0::2], values[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


class AttentionProjection:
    """One layer's per-position values derived from the normalized layer input."""

    def __init__(self, module, *, heads=None):
        """`heads` selects a rank's `(first, last)` head partition of the canonical module. None keeps every head."""
        first, last = (0, module.num_heads) if heads is None else heads
        self.module, self.first, self.last = module, first, last
        self.content, self.value = module.qk_nope_head_dim, module.v_head_dim
        weight = module.kv_b_proj.weight.to(torch.float32).view(module.num_heads, self.content + self.value, LATENT_RANK)
        # Per head, the key expansion absorbs content queries and the value expansion follows attention.
        self.absorb, self.expand = weight[first:last, :self.content], weight[first:last, self.content:]

    def __call__(self, hidden, positions, base):
        """Return the layer input's query, cache record fields, and index operands, all FP32.

        `absorbed` is [rows, heads, 576]: each head's content query in latent
        space followed by its rotated rotary query. `latent` and `key_rotary`
        form a position's cache record. Index operands are None for a layer
        without an indexer.
        """
        m, x = self.module, hidden.to(torch.float32)
        rows = x.shape[0]
        cos, sin = rotary_angles(positions, base)
        query_low = torch.nn.functional.rms_norm(x @ m.q_a_proj.weight.to(torch.float32).T,
                                                 m.q_a_layernorm.weight.shape, m.q_a_layernorm.weight.to(torch.float32),
                                                 m.q_a_layernorm.variance_epsilon)
        width = self.content + ROTARY_DIM
        query = (query_low @ m.q_b_proj.weight.to(torch.float32).T).view(rows, m.num_heads, width)[:, self.first:self.last]
        absorbed = torch.einsum("rhc,hcl->rhl", query[..., :self.content], self.absorb)
        absorbed = torch.cat((absorbed, rotate(query[..., self.content:], cos, sin)), -1)
        compressed = x @ m.kv_a_proj_with_mqa.weight.to(torch.float32).T
        latent = torch.nn.functional.rms_norm(compressed[:, :LATENT_RANK], m.kv_a_layernorm.weight.shape,
                                              m.kv_a_layernorm.weight.to(torch.float32),
                                              m.kv_a_layernorm.variance_epsilon)
        key_rotary = rotate(compressed[:, LATENT_RANK:], cos, sin)
        index_query = index_key = index_weights = None
        if m.indexer is not None:
            i = m.indexer
            index_query = (query_low @ i.wq_b.weight.to(torch.float32).T).view(rows, i.n_heads, i.head_dim)
            index_query = torch.cat((rotate(index_query[..., :ROTARY_DIM], cos, sin), index_query[..., ROTARY_DIM:]), -1)
            key = torch.nn.functional.layer_norm(x @ i.wk.weight.to(torch.float32).T, (i.head_dim,),
                                                 i.k_norm.weight.to(torch.float32), i.k_norm.bias.to(torch.float32),
                                                 i.k_norm.eps)
            index_key = torch.cat((rotate(key[:, :ROTARY_DIM], cos, sin), key[:, ROTARY_DIM:]), -1)
            # The score scale and the head-count normalization fold into the head weights.
            index_weights = (x @ i.weights_proj.weight.to(torch.float32).T) * (i.softmax_scale * i.n_heads ** -0.5)
        return dict(query_low=query_low, absorbed=absorbed, latent=latent, key_rotary=key_rotary,
                    index_query=index_query, index_key=index_key, index_weights=index_weights)

    def finish(self, attended):
        """Expand attended latents [rows, heads, 512] into this partition's per-head values, then project.

        A head partition's result is a partial hidden vector, and the partitions' results sum.
        """
        m = self.module
        values = torch.einsum("rhl,hvl->rhv", attended.to(torch.float32), self.expand).flatten(1)
        columns = m.o_proj.weight.to(torch.float32)[:, self.first * self.value:self.last * self.value]
        return values @ columns.T


def select(index_query, index_weights, index_keys, positions, top_k):
    """Select each row's `top_k` highest-scoring positions at or before its own.

    `index_keys` holds one key per global position. Returns sorted positions [rows, top_k] padded with -1.
    """
    scores = torch.relu(torch.einsum("rhd,td->rht", index_query, index_keys))
    scores = torch.einsum("rh,rht->rt", index_weights, scores)
    visible = torch.arange(index_keys.shape[0], device=positions.device)[None, :] <= positions[:, None]
    scores = scores.masked_fill(~visible, -torch.inf)
    count = min(top_k, index_keys.shape[0])
    values, selected = scores.topk(count, dim=-1)
    selected = torch.where(torch.isfinite(values), selected, -1)
    selected = torch.where(selected >= 0, selected, index_keys.shape[0]).sort(-1).values
    selected = torch.where(selected < index_keys.shape[0], selected, -1)
    padding = selected.new_full((selected.shape[0], top_k - count), -1)
    return torch.cat((selected, padding), -1)


def attend(absorbed, records, selected, scale):
    """Attend each row's selected records. Returns latents [rows, heads, 512] and natural LSE [rows, heads].

    `records` are FP32 [records, 576]: latent then rotary key. `selected` holds record indices padded with -1.
    A row without records returns zeros and an LSE of -inf.
    """
    live = selected >= 0
    chosen = records[selected.clamp_min(0)]
    scores = torch.einsum("rhd,rkd->rhk", absorbed, chosen) * scale
    scores = scores.masked_fill(~live[:, None, :], -torch.inf)
    lse = torch.logsumexp(scores, -1)
    weights = torch.softmax(scores, -1).nan_to_num(0.0)
    return torch.einsum("rhk,rkl->rhl", weights, chosen[..., :LATENT_RANK]), lse


def local_selection(selected, rank, world):
    """A context rank's records among globally selected positions: `position // world` where it owns them."""
    owned = (selected >= 0) & (selected % world == rank)
    return torch.where(owned, selected // world, -1)


def local_records(selected, positions, tables, *, rank=0, world=1, first=0):
    """Reference of the `_local_records` kernel in `ignition.recipes.glm_dsa.dcp`.

    Returns rank `rank`'s physical records of `selected`, sorted global positions [rows, topk] padded with -1,
    compacted to the front of int32 [rows, topk], with int32 counts [rows]. `positions` holds each row's own
    position, [rows], and `tables` the page-table rows, [rows, width].
    """
    logical = selected.long()
    local = logical.clamp_min(0) // world
    column = (local // PAGE_RECORDS).clamp(max=tables.shape[-1] - 1)
    page = tables.gather(-1, column).long()
    live = ((logical >= first) & (logical <= positions[:, None]) & (page >= 0)
            & (logical % world == rank))
    physical = torch.where(live, page * PAGE_RECORDS + local % PAGE_RECORDS, -1)
    return _compact(physical, live)


def pack_candidates(selected, scores, *, rank=0, world=1):
    """Reference of the `_pack_candidates` kernel in `ignition.recipes.glm_dsa.dcp`.

    Returns rank `rank`'s local keys `selected`, [rows, topk] padded with -1, and their FP32 `scores`,
    [rows, topk], as int64 candidates [rows, topk] in descending order.
    """
    bits = scores.view(torch.int32)
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).long()
    position = selected.long() * world + rank
    keys = torch.where(selected >= 0, (ordered << 32) | (0x7FFFFFFF - position), _MISSING)
    return keys.sort(-1, descending=True).values


def merged_records(candidates, positions, tables, *, rank=0, first=0):
    """Reference of the `_merge_records` kernel in `ignition.recipes.glm_dsa.dcp`.

    Returns rank `rank`'s physical records of its candidates among the `topk` highest of every rank's
    `candidates`, [rows, world, topk], compacted to the front of int32 [rows, topk], with int32 counts [rows].
    `positions` holds each row's own position, [rows], and `tables` the page-table rows, [rows, width].
    """
    rows, world, topk = candidates.shape
    own = candidates[:, rank]
    # The lowest of the 2048 highest candidates. A row with fewer candidates keeps them all.
    bound = candidates.reshape(rows, -1).topk(topk, -1).values[:, -1:]
    logical = 0x7FFFFFFF - (own & 0xFFFFFFFF)
    local = logical // world
    column = (local // PAGE_RECORDS).clamp(max=tables.shape[-1] - 1)
    page = tables.gather(-1, column).long()
    live = ((own != _MISSING) & (own >= bound) & (logical >= first) & (logical <= positions[:, None])
            & (page >= 0))
    physical = torch.where(live, page * PAGE_RECORDS + local % PAGE_RECORDS, -1)
    return _compact(physical, live)


def global_records(selected, positions, *, first=0, marks=None):
    """Reference of the `_global_records` kernel in `ignition.recipes.glm_dsa.dcp`.

    Returns the global positions of `selected`, [rows, topk] padded with -1, from `first` through each row's
    own in `positions`, [rows], compacted to the front of int32 [rows, topk], with int32 counts [rows]. With
    `marks`, also writes the marks of `mark_positions`.
    """
    logical = selected.long()
    live = (logical >= first) & (logical <= positions[:, None])
    records, counts = _compact(torch.where(live, logical, -1), live)
    if marks is not None:
        mark_positions(records, marks)
    return records, counts


def mark_positions(records, marks):
    """Reference of the marks that the `_global_records` kernel in `ignition.recipes.glm_dsa.dcp` writes.

    `marks` is uint8 [positions + 1], and its last entry is a discard. Sets the mark of every position that
    `records` names, clears every other mark, and returns `marks`.
    """
    marks.zero_()
    flat = records.reshape(-1).long()
    marks.scatter_(0, torch.where(flat >= 0, flat, marks.shape[0] - 1), 1)
    return marks


def _compact(records, live):
    # A stable descending sort of liveness moves live records forward in their selected order.
    order = torch.sort(live.to(torch.int8), dim=-1, descending=True, stable=True).indices
    return records.gather(-1, order).to(torch.int32), live.sum(-1, dtype=torch.int32)


def combine(partials, lses):
    """Combine per-rank partial attention results by their natural log-sum-exp.

    `partials` is [ranks, rows, heads, 512] and `lses` is [ranks, rows, heads].
    """
    weights = torch.softmax(lses, 0).nan_to_num(0.0)
    return (partials * weights[..., None]).sum(0)
