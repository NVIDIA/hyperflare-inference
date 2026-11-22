# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Head-partitioned latent attention with rotary positions over paged FP8 latent records.

Every rank holds the same latent records and partitions the 64 attention
heads. A record is 656 bytes, the layout b12x sparse MLA reads as `GLM_NSA`:
512 E4M3 latent values, four FP32 scales for their groups of 128, and the 64
rotated key values in BF16. A model page of 256 tokens is four 64-record
cache pages.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from b12x.attention import sparse_mla

from ignition.components.glm_mla_cache import store_glm_latent
from ignition.components.linear import BatchedUnquantizedProjection, UnquantizedProjection, prepare_projection
from ignition.components.rotary import PreparedRotaryAngles
from ignition.components.sparse_mla import PreparedSparseMLA
from ignition.preparation.examples import declaration_tensor

RECORD_BYTES = 656
PAGE_RECORDS = 64
PAGE_TOKENS = 256
TOPK = 2048
QUERY_RANK, LATENT, ROTARY, NOPE, VALUE = 2048, 512, 64, 192, 256
# BF16 layer-input projection: query rank, latent, rotary key, then the index key.
INPUT_WIDTH = QUERY_RANK + LATENT + ROTARY
INDEX_WIDTH = 128


def rotary_frequencies(theta, device):
    """One FP32 frequency per interleaved pair of the 64 rotary values."""
    from ignition.preparation.persistent import persistent_copy

    return persistent_copy(1 / theta ** (torch.arange(32, dtype=torch.float32, device=device) / 32))


@dataclass(frozen=True)
class AttentionMetadata:
    """Token placement shared by every layer of one forward.

    `positions` and `token_slots` hold -1 for rows without a token. A slot is
    the physical record index of the token's latent record. `block_table`
    lists each request's model pages; `requests` names each token's row in it.
    """

    positions: torch.Tensor
    token_slots: torch.Tensor
    requests: torch.Tensor
    block_table: torch.Tensor
    index_tables: tuple[torch.Tensor, torch.Tensor] | None = None

    @classmethod
    def allocate(cls, tokens, sequences, blocks, device):
        if min(tokens, sequences, blocks) <= 0:
            raise ValueError("GLM 5.3 attention metadata requires positive capacities")
        return cls(torch.full((tokens,), -1, dtype=torch.int64, device=device),
                   torch.full((tokens,), -1, dtype=torch.int64, device=device),
                   torch.zeros(tokens, dtype=torch.int32, device=device),
                   torch.full((sequences, blocks), -1, dtype=torch.int32, device=device))


class ExampleBuffers:
    """Zeroed tensors that operators receive as examples when they are declared and tuned.

    Tuning runs operators one at a time, so operators share one allocation per
    role and row width: every layer of a forward, and forwards of smaller
    capacity, which take a prefix of the largest one. Without sharing, a
    full-depth prefill forward would hold tens of gigabytes of examples until
    tuning completes. `clear` drops the tensors once tuning has run.
    """

    def __init__(self, device):
        self.device, self.tensors = device, {}

    def __call__(self, role, *shape, dtype=torch.bfloat16):
        return declaration_tensor(self.tensors, f"glm53_full.{role}{shape[1:]}", shape, dtype, self.device)

    def clear(self):
        self.tensors.clear()


class Selection(NamedTuple):
    """Physical latent record indices each query row attends to, and how many are valid."""

    records: torch.Tensor
    lengths: torch.Tensor


class Projected(NamedTuple):
    query_residual: torch.Tensor
    absorbed: torch.Tensor
    index_key: torch.Tensor | None
    index_weights: torch.Tensor | None


class PreparedAttention:
    """One rank's attention of one layer, before the output sum over ranks.

    `project` computes queries, writes this forward's latent records, and
    returns what selection needs; `attend` reads the selected records. A layer
    with an indexer also projects its index key and head weights.
    """

    def __init__(self, weights, source, metadata: AttentionMetadata, cache, angles: PreparedRotaryAngles, *,
                 heads, selects, eps, softmax_scale, mode, session, name, workspace, max_cache_pages=None,
                 buffers: ExampleBuffers | None = None, cache_only=False):
        rows = source.shape[0]
        width = INPUT_WIDTH + (INDEX_WIDTH if selects else 0)
        device = source.device
        if source.shape != (rows, 6144) or source.dtype != torch.bfloat16:
            raise ValueError("GLM 5.3 attention requires BF16 [rows, 6144] input")
        if mode not in ("decode", "prefill"):
            raise ValueError("GLM 5.3 attention prepares decode or prefill")
        if (cache.ndim != 3 or cache.shape[1:] != (PAGE_RECORDS, RECORD_BYTES) or cache.dtype != torch.uint8
                or weights["input_proj"].shape != (width, 6144)
                or weights["q_b_proj"].shape != (heads * (NOPE + ROTARY), QUERY_RANK)
                or weights["key"].shape != (heads, NOPE, LATENT)
                or weights["value"].shape != (heads, LATENT, VALUE)
                or weights["o_proj"].shape != (6144, heads * VALUE)):
            raise ValueError("GLM 5.3 attention cache or head partition geometry mismatch")
        if selects and weights["indexer.weights_proj"].shape != (32, 6144):
            raise ValueError("GLM 5.3 index head projection requires a [32, 6144] weight matrix")
        self.weights, self.cache, self.angles = weights, cache, angles
        self.heads, self.selects, self.eps, self.mode, self.rows = heads, selects, eps, mode, rows

        buffers = ExampleBuffers(device) if buffers is None else buffers
        roles = iter(range(64))

        def zeros(*shape, dtype=torch.bfloat16):
            # Each call below is a distinct role; operands of one operator never alias.
            return buffers(f"attention.{next(roles)}", *shape, dtype=dtype)

        projected, residual = zeros(rows, width), zeros(rows, QUERY_RANK)
        query, absorbed = zeros(rows, heads, NOPE + ROTARY), zeros(rows, heads, LATENT + ROTARY)
        attended, expanded = zeros(rows, heads, LATENT), zeros(rows, heads, VALUE)
        self.projections = {
            "input_proj": prepare_projection(source, weights["input_proj"], projected,
                session=session, name=f"{name}.input_proj", workspace=workspace)}
        self.absorb = self.expand = self.attention = None
        if not cache_only:
            if selects:
                self.projections["indexer.weights_proj"] = UnquantizedProjection(
                    source, weights["indexer.weights_proj"], zeros(rows, 32, dtype=torch.float32),
                    session=session, name=f"{name}.indexer.weights_proj")
            self.projections.update({
                "q_b_proj": prepare_projection(residual, weights["q_b_proj"], query.view(rows, -1),
                    session=session, name=f"{name}.q_b_proj", workspace=workspace),
                "o_proj": prepare_projection(expanded.view(rows, -1), weights["o_proj"], zeros(rows, 6144),
                    session=session, name=f"{name}.o_proj", workspace=workspace)})
            self.absorb = BatchedUnquantizedProjection(query[:, :, :NOPE].transpose(0, 1), weights["key"],
                absorbed[:, :, :LATENT].transpose(0, 1), major="n", session=session, name=f"{name}.absorb_query")
            self.expand = BatchedUnquantizedProjection(attended.transpose(0, 1), weights["value"],
                expanded.transpose(0, 1), major="n", session=session, name=f"{name}.expand_value")
        if cache_only:
            return
        selected = zeros(rows, TOPK, dtype=torch.int32)
        visible, lengths = zeros(rows, dtype=torch.int32), zeros(rows, dtype=torch.int32)
        caps = sparse_mla.Caps(device=device, num_q_heads=heads, max_q_rows=rows, max_width=TOPK,
            softmax_scale=softmax_scale, max_batch=rows,
            max_kv_rows=max(cache.shape[0], 4 * (max_cache_pages or 0)) * PAGE_RECORDS,
            kv_dtype=torch.uint8, head_dim=LATENT + ROTARY, v_head_dim=LATENT, page_size=PAGE_RECORDS,
            model_type=sparse_mla.ModelType.GLM_NSA, cache_record_bytes=RECORD_BYTES,
            mode="decode" if mode == "decode" else "extend")
        self.attention = PreparedSparseMLA(absorbed, cache, selected, visible, lengths, attended, caps=caps,
            session=session, name=f"{name}.attention", workspace=workspace)

    def rebind_pages(self, cache):
        """Write and attend through a replacement page pool."""
        self.cache = cache
        if self.attention is not None:
            self.attention.rebind_cache(cache)

    def functional(self, *, session):
        for projection in self.projections.values():
            projection.functional(session=session)
        for operator in (self.absorb, self.expand, self.attention):
            if operator is not None:
                operator.functional(session=session)
        return self

    def project(self, source, metadata, angles):
        """Project one layer input, store its latent records, and return the absorbed queries."""
        rows, heads, device = source.shape[0], self.heads, source.device
        projected = self.projections["input_proj"](source)
        residual = torch.empty((rows, QUERY_RANK), dtype=torch.bfloat16, device=device)
        self._store(projected, metadata, angles, query=residual)
        query = self.projections["q_b_proj"](residual).view(rows, heads, NOPE + ROTARY)
        self.angles.apply(query, angles)
        absorbed = torch.empty((rows, heads, LATENT + ROTARY), dtype=torch.bfloat16, device=device)
        self.absorb(query[:, :, :NOPE].transpose(0, 1), output=absorbed[:, :, :LATENT].transpose(0, 1))
        absorbed[:, :, LATENT:].copy_(query[:, :, NOPE:])
        if not self.selects:
            return Projected(residual, absorbed, None, None)
        return Projected(residual, absorbed, projected[:, INPUT_WIDTH:INPUT_WIDTH + 128],
                         self.projections["indexer.weights_proj"](source))

    def store(self, source, metadata, angles):
        """Project one layer input and store its latent records; return the index key input, or None."""
        projected = self.projections["input_proj"](source)
        self._store(projected, metadata, angles)
        return projected[:, INPUT_WIDTH:INPUT_WIDTH + 128] if self.selects else None

    def _store(self, projected, metadata, angles, *, query=None):
        store_glm_latent(projected, self.weights.get("q_a_layernorm.weight"), self.weights["kv_a_layernorm.weight"],
                         angles, metadata.token_slots[:projected.shape[0]], self.cache, self.eps, query=query)

    def attend(self, absorbed, selection: Selection, metadata):
        """Attend to the selected records and return this rank's partial hidden update."""
        rows = absorbed.shape[0]
        visible = (metadata.positions[:rows] + 1).to(torch.int32)
        attended = self.attention(absorbed, selection.records, visible, selection.lengths, cache=self.cache)
        expanded = self.expand(attended.transpose(0, 1)).transpose(0, 1)
        return self.projections["o_proj"](expanded.reshape(rows, -1))
