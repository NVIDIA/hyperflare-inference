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
from b12x.preparation import PreparedCall
from flashinfer.triton.norm import rms_norm

from ignition.components.linear import BatchedUnquantizedProjection, prepare_projection
from ignition.components.rotary import PreparedRotaryAngles
from ignition.components.sparse_mla import PreparedSparseMLA
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import declare

RECORD_BYTES = 656
# The leading bytes of a record: E4M3 latent values and their FP32 group scales.
LATENT_BYTES = 528
PAGE_RECORDS = 64
PAGE_TOKENS = 256
TOPK = 2048
QUERY_RANK, LATENT, ROTARY, NOPE, VALUE = 2048, 512, 64, 192, 256
# Fused layer-input projection: query rank, latent, rotary key, then the index key and head weights.
INPUT_WIDTH = QUERY_RANK + LATENT + ROTARY
INDEX_WIDTH = 128 + 32


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


def write_records(latent_records, keys, slots, cache):
    """Store each valid row's 528 latent bytes and BF16 rotated key as one 656-byte record.

    Rows whose slot is negative write nothing. Every such row repeats the first
    valid row's write, so duplicate indices carry identical bytes; with no valid
    row, record zero is rewritten with its own contents.
    """
    rows = slots.shape[0]
    records = cache.view(-1, RECORD_BYTES)
    valid = slots >= 0
    first = valid.to(torch.int32).argmax()
    order = torch.where(valid, torch.arange(rows, device=slots.device), first)
    target = slots[order].clamp(min=0)
    payload = torch.cat((latent_records, keys.contiguous().view(torch.uint8).view(rows, 2 * ROTARY)), dim=1)[order]
    payload = torch.where(valid.any(), payload, records[target])
    records.index_copy_(0, target, payload)


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
            self.projections.update({
                "q_b_proj": prepare_projection(residual, weights["q_b_proj"], query.view(rows, -1),
                    session=session, name=f"{name}.q_b_proj", workspace=workspace),
                "o_proj": prepare_projection(expanded.view(rows, -1), weights["o_proj"], zeros(rows, 6144),
                    session=session, name=f"{name}.o_proj", workspace=workspace)})
            # Batched products take head-major contiguous operands; the forward copies between that order
            # and the row-major order of the projections and the attention kernel.
            self.absorb = BatchedUnquantizedProjection(zeros(heads, rows, NOPE), weights["key"],
                zeros(heads, rows, LATENT), major="n", session=session, name=f"{name}.absorb_query")
            self.expand = BatchedUnquantizedProjection(zeros(heads, rows, LATENT), weights["value"],
                zeros(heads, rows, VALUE), major="n", session=session, name=f"{name}.expand_value")
        # The b12x writer produces the 528 latent bytes of each record in packed staging rows.
        latent = zeros(rows, LATENT)
        staging = zeros(-(-rows // PAGE_RECORDS), PAGE_RECORDS, LATENT_BYTES, dtype=torch.uint8)
        self.staging_pages = staging.shape[0]
        self.writer_plan = sparse_mla.plan_cache_writer(latent, staging, metadata.token_slots[:rows])
        slots = metadata.token_slots[:rows]
        declare(session, (self.writer_plan.request(name=f"{name}.cache_write",
            prepare_call=lambda state: PreparedCall(run=lambda: state.run(latent, staging, slots),
                                                    owners=(latent, staging, slots))),))
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
        rms_norm(projected[:, :QUERY_RANK], self.weights["q_a_layernorm.weight"], residual, self.eps,
                 round_before_weight=True)
        query = self.projections["q_b_proj"](residual).view(rows, heads, NOPE + ROTARY)
        self.angles.apply(query, angles)
        absorbed = torch.empty((rows, heads, LATENT + ROTARY), dtype=torch.bfloat16, device=device)
        absorbed[:, :, :LATENT].copy_(self.absorb(query[:, :, :NOPE].transpose(0, 1).contiguous()).transpose(0, 1))
        absorbed[:, :, LATENT:].copy_(query[:, :, NOPE:])
        self._store(projected, metadata, angles)
        if not self.selects:
            return Projected(residual, absorbed, None, None)
        return Projected(residual, absorbed, projected[:, INPUT_WIDTH:INPUT_WIDTH + 128],
                         projected[:, INPUT_WIDTH + 128:])

    def store(self, source, metadata, angles):
        """Project one layer input and store its latent records; return the index key input, or None."""
        projected = self.projections["input_proj"](source)
        self._store(projected, metadata, angles)
        return projected[:, INPUT_WIDTH:INPUT_WIDTH + 128] if self.selects else None

    def _store(self, projected, metadata, angles):
        rows, device = projected.shape[0], projected.device
        latent = torch.empty((rows, LATENT), dtype=torch.bfloat16, device=device)
        rms_norm(projected[:, QUERY_RANK:QUERY_RANK + LATENT], self.weights["kv_a_layernorm.weight"], latent,
                 self.eps, round_before_weight=True)
        key = projected[:, QUERY_RANK + LATENT:INPUT_WIDTH].contiguous()
        self.angles.apply(key, angles)
        slots = metadata.token_slots[:rows]
        staging = torch.zeros((self.staging_pages, PAGE_RECORDS, LATENT_BYTES), dtype=torch.uint8, device=device)
        packed = torch.where(slots >= 0, torch.arange(rows, device=device), -1)
        sparse_mla.concat_and_cache_glm_next_mla(latent, staging, packed, plan=self.writer_plan)
        write_records(staging.view(-1, LATENT_BYTES)[:rows], key, slots, self.cache)

    def attend(self, absorbed, selection: Selection, metadata):
        """Attend to the selected records and return this rank's partial hidden update."""
        rows = absorbed.shape[0]
        visible = (metadata.positions[:rows] + 1).to(torch.int32)
        attended = self.attention(absorbed, selection.records, visible, selection.lengths, cache=self.cache)
        expanded = self.expand(attended.transpose(0, 1).contiguous()).transpose(0, 1)
        return self.projections["o_proj"](expanded.reshape(rows, -1))
