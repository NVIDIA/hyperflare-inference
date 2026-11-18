# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Whole-head attention shards with FP8 histories, relative bias and causal convolutions.

Prefill stages the complete query chunk after its preceding sliding window.
Verification stages candidate window entries without changing committed rings
or convolution histories. Accepted counts commit those histories afterward.
Full-attention candidates write private request pages. Attention uses BF16
queries and relative-bias operands, FP8 K/V and FP32 row sinks. The gated
BF16 result projects to FP32 partial sums for cross-rank reduction before the
replicated BF16 output convolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.components.causal_convolution import PreparedCausalConvolution
from ignition.components.head_gate import PreparedHeadGate
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.state import gather_rows, scatter_rows
from ignition.preparation.persistent import persistent_empty

from .arithmetic import l2_clip, log_scaling
from .pool import KV_DTYPE, PAGE_TOKENS

RELATIVE_RANK = 16


@dataclass(frozen=True)
class Columns:
    """Column ranges of one rank's fused attention projections, by stream."""

    streams: tuple[tuple[tuple[str, int], ...], ...]

    @classmethod
    def of(cls, local, heads, kv_heads, dim, width):
        query = (("query", heads * dim), ("gate", heads * dim))
        relative = (("relative", heads * RELATIVE_RANK), ("sink", heads))
        if local:
            return cls(((*query, ("key", kv_heads * dim), ("value", kv_heads * dim), *relative),))
        return cls(((*query, *relative), (("key", kv_heads * dim), ("key_lags", kv_heads * width)),
                    (("value", kv_heads * dim), ("value_lags", kv_heads * width))))

    def widths(self):
        """Fused output widths, padded to 8 columns so that every BF16 row starts 16-byte aligned."""
        return tuple(-(-sum(width for _, width in stream) // 8) * 8 for stream in self.streams)

    def split(self, outputs):
        parts = {}
        for stream, output in zip(self.streams, outputs, strict=True):
            offset = 0
            for name, width in stream:
                parts[name] = output[:, offset:offset + width]
                offset += width
        return parts


@dataclass(frozen=True)
class AttentionWeights:
    """One rank's attention tensors of one layer in serving layouts.

    `projections` hold one fused BF16 weight per input stream with the rows of
    `Columns`. Local layers carry static `[channels, 1, W]` K/V convolution taps;
    global layers carry `[W, channels]` static lag taps. `relative_projection` is
    the checkpoint's `[16, extent]` relative-logit projection and
    `relative_rows` its contiguous `[extent, 16]` transpose.
    """

    projections: tuple[torch.Tensor, ...]
    query_clip_scale: torch.Tensor
    relative_clip_scale: torch.Tensor
    relative_projection: torch.Tensor
    relative_rows: torch.Tensor
    key_convolution: torch.Tensor
    value_convolution: torch.Tensor
    output: torch.Tensor


def fuse_projections(placed, columns: Columns, *, local):
    """Concatenate a rank's projection rows in `Columns` order; padding rows are zero."""
    from ignition.preparation.persistent import persistent_zeros

    names = {"query": "query", "gate": "output_gate", "key": "key", "value": "value", "relative": "relative_query",
             "sink": "sink", "key_lags": "key_convolution_dynamic", "value_lags": "value_convolution_dynamic"}
    fused = []
    for stream, width in zip(columns.streams, columns.widths(), strict=True):
        hidden = placed["query"].shape[1]
        weight = persistent_zeros((width, hidden), dtype=torch.bfloat16, device=placed["query"].device)
        offset = 0
        for name, rows in stream:
            source = placed[names[name]]
            if source.shape != (rows, hidden):
                raise ValueError(f"{names[name]} has {tuple(source.shape)} rows, the fused projection requires {rows}")
            weight[offset:offset + rows].copy_(source)
            offset += rows
        fused.append(weight)
    return tuple(fused)


def attention_weights(placed, *, local, heads, kv_heads, dim, width):
    """Serving attention tensors from one rank's placed checkpoint roles (`weights.checkpoint_layout`)."""
    from ignition.preparation.persistent import persistent_copy

    columns = Columns.of(local, heads, kv_heads, dim, width)
    projection = placed["relative_projection"]
    if local:
        key, value = placed["key_convolution"], placed["value_convolution"]
    else:
        key, value = placed["key_convolution_static"], placed["value_convolution_static"]
    return AttentionWeights(fuse_projections(placed, columns, local=local), placed["query_clip_scale"],
        placed["relative_clip_scale"], projection, persistent_copy(projection.t().contiguous()), key, value,
        placed["output"])


@dataclass(frozen=True)
class LayerCache:
    """A layer's rank-local storage: K/V (paged or ring pages) and K/V convolution histories.

    Local layers also expose each continuation row's K and V rings as bytes
    (`key_rings`, `value_rings`, `[rows, ring bytes]`), which prefill reads and
    rewrites one request row at a time.
    """

    key: torch.Tensor
    value: torch.Tensor
    key_history: torch.Tensor
    value_history: torch.Tensor
    ring_row_pages: int = 0
    key_rings: torch.Tensor | None = None
    value_rings: torch.Tensor | None = None

    @classmethod
    def of(cls, pool, layer):
        if pool.text.local(layer):
            return cls(pool.ring_pages(layer, "key"), pool.ring_pages(layer, "value"),
                       pool.history(layer, "key_convolution"), pool.history(layer, "value_convolution"),
                       pool.ring_row_pages, pool.ring_rows(layer, "key"), pool.ring_rows(layer, "value"))
        return cls(pool.cache(layer, "key"), pool.cache(layer, "value"), pool.history(layer, "key_convolution"),
                   pool.history(layer, "value_convolution"))


def ring_slots(positions, rows, ring, row_pages):
    """Flat cache slots of positions in their requests' rings."""
    slot = positions.clamp_min(0).remainder(ring)
    page = rows * row_pages + slot // PAGE_TOKENS
    return torch.where(positions >= 0, page * PAGE_TOKENS + slot % PAGE_TOKENS, -1)


def rotated_ring(metadata, ring, window, row_pages):
    """Page tables and lengths that list each decoding request's window in position order.

    Returns int32 `[requests, ring pages]` pages, starting with the page that holds
    position `max(p - window + 1, 0)` of the request's current position `p`, and
    each request's key count from that page's first position through `p`.
    """
    m = metadata
    count = ring // PAGE_TOKENS
    current = (m.starts + m.lengths - 1).clamp_min(0)
    first = (current - window + 1).clamp_min(0) // PAGE_TOKENS
    columns = torch.arange(count, device=current.device)
    pages = m.state_rows[:, None] * row_pages + (first[:, None] + columns).remainder(count)
    lengths = torch.where(m.lengths > 0, current + 1 - first * PAGE_TOKENS, 0)
    return pages.int(), lengths.int()


class PreparedAttention(torch.nn.Module):
    """A rank's attention for prefill, decode or speculative verification."""

    def __init__(self, layer, weights: AttentionWeights, text, cache: LayerCache, metadata, *, mode, tokens,
                 session, workspace, name, examples=None, tuning_page_count=None):
        super().__init__()
        if mode not in ("prefill", "decode", "verify"):
            raise ValueError(f"unsupported Inkling attention mode: {mode}")
        self.layer, self.mode, self.text, self.cache = layer, mode, text, cache
        self.local = text.local(layer)
        self.heads, self.kv_heads, self.dim = weights.query_clip_scale.shape[0], cache.key.shape[2], text.head_dim
        self.window, self.ring = text.window, text.window + PAGE_TOKENS
        self.columns = Columns.of(self.local, self.heads, self.kv_heads, self.dim, text.convolution_width)
        self.weights = weights
        self.key_convolution = PreparedCausalConvolution(weights.key_convolution, cache.key_history,
            dynamic_heads=0 if self.local else self.kv_heads)
        self.value_convolution = PreparedCausalConvolution(weights.value_convolution, cache.value_history,
            dynamic_heads=0 if self.local else self.kv_heads)
        device = weights.output.device
        examples = {} if examples is None else examples

        def example(role, shape, dtype=torch.bfloat16):
            from ignition.preparation.examples import declaration_tensor
            return declaration_tensor(examples, role, shape, dtype, device)

        source = example("inkling.attention.source", (tokens, text.hidden))
        self.projections = tuple(UnquantizedProjection(source, weight,
            example(f"inkling.attention.projection{index}", (tokens, width)), session=session,
            name=f"{name}.projection{index}").functional(session=session)
            for index, (weight, width) in enumerate(zip(weights.projections, self.columns.widths(), strict=True)))
        attended = example("inkling.attention.attended", (tokens, self.heads * self.dim))
        self.output_projection = UnquantizedProjection(attended, weights.output,
            example("inkling.attention.partial", (tokens, text.hidden), torch.float32),
            session=session, name=f"{name}.output").functional(session=session)
        query = example("inkling.attention.query", (tokens, self.heads, self.dim))
        output = example("inkling.attention.output", (tokens, self.heads, self.dim))
        sink = example("inkling.attention.sink", (tokens, self.heads), torch.float32)
        key = example("inkling.attention.key", (tokens, self.kv_heads, self.dim))
        value = example("inkling.attention.value", (tokens, self.kv_heads, self.dim))
        gain_key = ("inkling.attention.value_norm_gain", self.dim, device)
        gain = examples.get(gain_key)
        if gain is None:
            gain = examples[gain_key] = persistent_empty((self.dim,), torch.bfloat16, device).fill_(1)
        self.value_norm = PreparedRMSNorm(value.view(-1, self.dim), gain,
            example("inkling.attention.normalized_value", (tokens * self.kv_heads, self.dim)),
            eps=text.eps, session=session, name=f"{name}.value_norm").functional()
        projection = example("inkling.attention.projection0", (tokens, self.columns.widths()[0]))
        channels = self.heads * self.dim
        self.gate = PreparedHeadGate(attended.view(tokens, channels, 1), projection[:, channels:2 * channels],
            session=session, name=f"{name}.gate").functional(session=session)
        slots = example("inkling.attention.slots", (tokens,), torch.int64)
        requests = metadata.lengths.numel()
        self.writer = None
        if self.local and mode in ("prefill", "verify"):
            self.staged_pages = -(-(self.window - 1 + metadata.max_query_tokens) // PAGE_TOKENS)
            count = requests * self.staged_pages
            cache_key = example("inkling.attention.staged_key", (count, PAGE_TOKENS, self.kv_heads, self.dim), KV_DTYPE)
            cache_value = example("inkling.attention.staged_value", (count, PAGE_TOKENS, self.kv_heads, self.dim), KV_DTYPE)
            pages = torch.arange(count, dtype=torch.int32, device=device).view(requests, self.staged_pages)
            self.staged_table = pages
        else:
            cache_key, cache_value = cache.key, cache.value
            pages = (example("inkling.attention.ring_table", (requests, self.ring // PAGE_TOKENS), torch.int32)
                     if self.local else metadata.page_table)
            self.writer = PreparedPagedKVWriter(key, value, cache_key, cache_value, slots,
                session=session, name=f"{name}.cache_write").functional()
        lengths = example("inkling.attention.lengths", (requests,), torch.int32)
        if mode == "prefill":
            bias_arguments = dict(relative_bias_factors=example("inkling.attention.relative_factor",
                (tokens, self.heads, RELATIVE_RANK)), relative_bias_projection=weights.relative_rows)
        else:
            bias_arguments = dict(relative_bias=example("inkling.attention.relative_bias",
                (tokens, self.heads, weights.relative_projection.shape[1])))
        self.attention = PreparedPagedAttention(query, cache_key, cache_value, pages, lengths,
            metadata.query_start_loc, output, sink=sink, mode="extend" if mode == "prefill" else mode,
            width=metadata.max_query_tokens, window=self.window - 1 if self.local else -1,
            tuning_page_count=min(tuning_page_count or cache_key.shape[0], cache_key.shape[0]),
            session=session, workspace=workspace, name=f"{name}.attention", **bias_arguments)
        self.attention = self.attention.functional(session=session,
            k_cache=None if self.local and mode in ("prefill", "verify") else cache.key,
            v_cache=None if self.local and mode in ("prefill", "verify") else cache.value,
            cache_seqlens=None, page_table=None, cu_seqlens_q=None, sink=None,
            **({"relative_bias_factors": None, "relative_bias_projection": None} if mode == "prefill"
               else {"relative_bias": None}))

    def project(self, streams, metadata):
        """Fused projections, convolved and clipped K/V and queries, sinks and gates of `streams`."""
        t, m = self.text, metadata
        inputs = streams if len(streams) == len(self.projections) else streams * len(self.projections)
        parts = self.columns.split([projection(source) for projection, source in zip(self.projections, inputs,
                                                                                    strict=True)])
        tokens = m.positions.shape[0]
        heads, kv_heads, dim = self.heads, self.kv_heads, self.dim
        key_raw, value_raw = parts["key"], parts["value"]
        dynamic_shape = (tokens, kv_heads, t.convolution_width)
        key, key_previous = self.key_convolution(key_raw, m,
            dynamic=None if self.local else parts["key_lags"].reshape(dynamic_shape))
        value, value_previous = self.value_convolution(value_raw, m,
            dynamic=None if self.local else parts["value_lags"].reshape(dynamic_shape))
        tau = None if self.local else log_scaling(m.positions, t.log_scaling_floor, t.log_scaling_alpha)
        if self.mode != "verify":
            self.key_convolution.commit(key_previous, key_raw, m)
            self.value_convolution.commit(value_previous, value_raw, m)
        query = l2_clip(parts["query"].reshape(tokens, heads, dim), t.qk_clip, scale=self.weights.query_clip_scale,
                        tau=tau)
        key = l2_clip(key.reshape(tokens, kv_heads, dim), t.qk_clip)
        value = self.value_norm(value.view(-1, dim)).view(tokens, kv_heads, dim)
        relative = l2_clip(parts["relative"].reshape(tokens, heads, RELATIVE_RANK), t.relative_clip,
                           scale=self.weights.relative_clip_scale, tau=tau)
        return dict(query=query, key=key, value=value, relative=relative, sink=parts["sink"].to(dtype=torch.float32, memory_format=torch.contiguous_format),
                    gate=parts["gate"].reshape(tokens, heads, dim), key_raw=key_raw, value_raw=value_raw,
                    key_previous=key_previous, value_previous=value_previous)

    def attend(self, streams, metadata):
        """The gated attention `[tokens, heads * dim]` and its partial output projection."""
        m = metadata
        values = self.project(streams, m)
        if self.mode == "prefill":
            attended = self._prefill(values, m)
        elif self.mode == "verify":
            attended = self._verify(values, m)
        else:
            attended = self._decode(values, m)
        attended = self.gate(attended.view(attended.shape[0], -1, 1),
            values["gate"].reshape(attended.shape[0], -1)).flatten(1)
        return attended, self.output_projection(attended), values

    def forward(self, streams, metadata):
        return self.attend(streams, metadata)[1]

    def _prefill(self, v, m):
        key, value = v["key"], v["value"]
        if self.local:
            key_ring = _request_ring(self.cache.key_rings, key, m)
            value_ring = _request_ring(self.cache.value_rings, value, m)
            staged_key, staged = _stage(key_ring, key, m, window=self.window, pages=self.staged_pages)
            staged_value, _ = _stage(value_ring, value, m, window=self.window, pages=self.staged_pages)
            _retain(self.cache.key_rings, key_ring, key, m)
            _retain(self.cache.value_rings, value_ring, value, m)
            return self.attention(v["query"], sink=v["sink"], relative_bias_factors=v["relative"],
                                  relative_bias_projection=self.weights.relative_rows,
                                  k_cache=staged_key, v_cache=staged_value, cache_seqlens=staged.reshape(1),
                                  page_table=self.staged_table, cu_seqlens_q=m.query_start_loc)
        self.writer(key, value, _page_slots(m))
        lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int()
        return self.attention(v["query"], sink=v["sink"], relative_bias_factors=v["relative"],
                                  relative_bias_projection=self.weights.relative_rows,
                              cache_seqlens=lengths, page_table=m.page_table, cu_seqlens_q=m.query_start_loc)

    def _decode(self, v, m):
        if self.local:
            slots = ring_slots(m.positions, m.state_rows[m.requests], self.ring, self.cache.ring_row_pages)
            pages, lengths = rotated_ring(m, self.ring, self.window, self.cache.ring_row_pages)
        else:
            slots = _page_slots(m)
            pages = m.page_table
            lengths = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int()
        self.writer(v["key"], v["value"], slots)
        tokens = v["relative"].shape[0]
        bias = (v["relative"].reshape(tokens * self.heads, RELATIVE_RANK) @ self.weights.relative_projection)
        attended = self.attention(v["query"], sink=v["sink"], relative_bias=bias.view(tokens, self.heads, -1),
                                  cache_seqlens=lengths, page_table=pages, cu_seqlens_q=m.query_start_loc)
        return attended

    def _verify(self, v, m):
        if self.local:
            cache_key, lengths = stage_batch(self.cache.key_rings, v["key"], m,
                                              window=self.window, pages=self.staged_pages)
            cache_value, _ = stage_batch(self.cache.value_rings, v["value"], m,
                                         window=self.window, pages=self.staged_pages)
            pages = self.staged_table
            kwargs = dict(k_cache=cache_key, v_cache=cache_value)
        else:
            self.writer(v["key"], v["value"], _page_slots(m))
            lengths, pages = torch.where(m.lengths > 0, m.starts + m.lengths, 0).int(), m.page_table
            kwargs = {}
        bias = (v["relative"].flatten(0, 1) @ self.weights.relative_projection).view(
            v["relative"].shape[0], self.heads, -1)
        return self.attention(v["query"], sink=v["sink"], relative_bias=bias,
            cache_seqlens=lengths, page_table=pages, cu_seqlens_q=m.query_start_loc, **kwargs)

    def commit(self, values, counts, metadata):
        self.key_convolution.commit(values["key_previous"], values["key_raw"], metadata, counts=counts)
        self.value_convolution.commit(values["value_previous"], values["value_raw"], metadata, counts=counts)
        if self.local:
            retain_batch(self.cache.key_rings, values["key"], counts, metadata)
            retain_batch(self.cache.value_rings, values["value"], counts, metadata)


def stage_batch(rings, values, m, *, window, pages):
    rows, capacity = m.starts.numel(), pages * PAGE_TOKENS
    ring = torch.empty((rows, rings.shape[1]), dtype=torch.uint8, device=rings.device)
    gather_rows(rings, m.state_rows, ring)
    ring = ring.view(KV_DTYPE).view(rows, -1, *values.shape[1:])
    first = (m.starts - window + 1).clamp_min(0)
    positions = first[:, None] + torch.arange(capacity, device=values.device)
    previous = ring[torch.arange(rows, device=values.device)[:, None], positions.remainder(ring.shape[1])].to(values.dtype)
    index = (m.query_start_loc[:-1, None] + positions - m.starts[:, None]).clamp(0, values.shape[0] - 1)
    staged = torch.where((positions < m.starts[:, None])[:, :, None, None], previous, values[index])
    lengths = torch.where(m.lengths > 0, m.starts - first + m.lengths, 0).int()
    staged = torch.where((positions < (m.starts + m.lengths)[:, None])[:, :, None, None], staged, 0)
    return staged.clamp(-448, 448).to(KV_DTYPE).view(rows * pages, PAGE_TOKENS, *values.shape[1:]), lengths


def retain_batch(rings, values, counts, m):
    rows = m.starts.numel()
    previous = torch.empty((rows, rings.shape[1]), dtype=torch.uint8, device=rings.device)
    gather_rows(rings, m.state_rows, previous)
    ring = previous.view(KV_DTYPE).view(rows, -1, *values.shape[1:])
    capacity = ring.shape[1]
    first = m.starts + counts - capacity
    positions = first[:, None] + (torch.arange(capacity, device=values.device) - first[:, None]).remainder(capacity)
    indices = (m.query_start_loc[:-1, None] + positions - m.starts[:, None]).clamp(0, values.shape[0] - 1)
    current = values[indices].clamp(-448, 448)
    keep = ((positions < m.starts[:, None]) | (positions < 0))[:, :, None, None]
    retained = torch.where(keep, ring.to(values.dtype), current).to(KV_DTYPE)
    scatter_rows(retained.view(torch.uint8).view(rows, -1), m.state_rows, rings, counts > 0)


def _page_slots(m):
    page = m.token_page_table.gather(1, (m.positions.clamp(min=0) // PAGE_TOKENS)[:, None]).squeeze(1)
    return torch.where((m.positions >= 0) & (page >= 0), page.long() * PAGE_TOKENS + m.positions % PAGE_TOKENS, -1)


def _request_ring(rings, values, m):
    """The prefill request's ring `[ring, kv_heads, dim]` (FP8), read from its continuation row of `rings`."""
    ring = torch.empty((1, rings.shape[1]), dtype=torch.uint8, device=rings.device)
    gather_rows(rings, m.state_rows[:1], ring)
    return ring.view(KV_DTYPE).view(-1, *values.shape[1:])


def _stage(ring, values, m, *, window, pages):
    """One prefill request's preceding `window - 1` ring entries followed by its chunk, as FP8 pages.

    `ring` is `_request_ring` of the request: position `p` occupies slot `p % ring`.
    Staged index `i` is position `max(start - window + 1, 0) + i`; returns the staged
    pages and the staged key count.
    """
    start, length = m.starts[0], m.lengths[0].long()
    first = (start - window + 1).clamp_min(0)
    positions = first + torch.arange(pages * PAGE_TOKENS, device=values.device)
    previous = ring[positions.remainder(ring.shape[0])].to(values.dtype)
    current = values[(positions - start).clamp(0, values.shape[0] - 1)]
    staged = torch.where((positions < start)[:, None, None], previous, current)
    staged = torch.where((positions < start + length)[:, None, None], staged, 0)
    staged = staged.clamp(-448, 448).to(KV_DTYPE).view(pages, PAGE_TOKENS, *values.shape[1:])
    return staged, (start - first + length).int()


def _retain(rings, ring, values, m):
    """Store the request's ring with the chunk's positions among its last `ring` positions.

    Slot `s` holds the position `p` in `[end - ring, end)` with `p % ring == s`; slots
    of positions before the chunk or before the sequence keep their bytes, and a
    request without query rows stores nothing.
    """
    count = ring.shape[0]
    start, length = m.starts[0], m.lengths[0].long()
    first = start + length - count
    positions = first + (torch.arange(count, device=values.device) - first).remainder(count)
    current = values[(positions - start).clamp(0, values.shape[0] - 1)].clamp(-448, 448)
    keep = ((positions < start) | (positions < 0))[:, None, None]
    # E4M3 values round-trip exactly through BF16.
    retained = torch.where(keep, ring.to(values.dtype), current).to(KV_DTYPE)
    scatter_rows(retained.view(torch.uint8).view(1, -1), m.state_rows[:1], rings, m.lengths[:1] > 0)
