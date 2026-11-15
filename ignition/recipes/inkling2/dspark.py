# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling2 noncausal DSpark blocks and accepted target-feature context storage."""

from dataclasses import dataclass

import torch

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.recurrent_proposal import PreparedRecurrentProposalHead
from ignition.components.rms_norm import PreparedRMSNorm
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.persistent import persistent_copy, persistent_zeros

from .dspark_state import (block_inputs, context_slots, mix_features, rotary_table, rotate, stage, tail_plan,
                           window_plan)


def inverse_frequencies(draft, device):
    return persistent_copy(1. / (draft.theta ** (torch.arange(0, draft.head_dim, 2,
        dtype=torch.float32, device=device) / draft.head_dim)))


def norm(source, weight, output, *, eps, session, name):
    return PreparedRMSNorm(source, weight, output, eps=eps, session=session, name=name).functional()


class PreparedContext:
    """Overwrite accepted proposal positions with context projected from target taps."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace=None, **kwargs):
        d, draft = weights.config.draft, weights.draft
        self.pool, self.alpha, self.window = pool, draft.alpha, d.window
        self.eps = d.text["rms_norm_eps"]
        self.recurrent_norm = draft.recurrent.context_norm
        self.inverse = inverse_frequencies(d, auxiliary.device)
        tokens, hidden = auxiliary.shape[0], weights.config.text["hidden_size"]
        kv_heads = d.kv_heads // weights.shard.world_size
        self.key_shape = (tokens, kv_heads, d.head_dim)
        source = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=auxiliary.device)
        kv = torch.empty((tokens, 2 * kv_heads * d.head_dim), dtype=torch.bfloat16, device=auxiliary.device)
        key = torch.empty(self.key_shape, dtype=torch.bfloat16, device=auxiliary.device)
        slots = torch.full((tokens,), -1, dtype=torch.int64, device=auxiliary.device)
        self.stages = []
        for index, layer in enumerate(draft.layers):
            projection = UnquantizedProjection(source, layer.context_kv, kv,
                session=session, name=f"{name}.{index}.kv").functional(session=session)
            key_norm = norm(key.flatten(0, 1), layer.context_key_norm, key.flatten(0, 1),
                eps=self.eps, session=session, name=f"{name}.{index}.key_norm")
            local = index in d.local_layers
            caches = ((pool.reserved_region(f"draft{index}.key"), pool.reserved_region(f"draft{index}.value"))
                      if local else (pool.key(index, draft=True), pool.value(index, draft=True)))
            writer = PreparedPagedKVWriter(key, key, *caches, slots,
                session=session, name=f"{name}.{index}.write").functional()
            self.stages.append((projection, key_norm, layer.context_norm, writer, local))

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        m = metadata
        full_slots = context_slots(m, counts)
        local_slots = context_slots(m, counts, window=self.window)
        table = rotary_table(m.positions, self.inverse)
        for index, (projection, key_norm, context_norm, writer, local) in enumerate(self.stages):
            context = mix_features(auxiliary, self.alpha[index], context_norm, self.eps)
            key, value = projection(context).chunk(2, -1)
            key = key_norm(key.contiguous().view(-1, self.key_shape[-1])).view(self.key_shape)
            writer(rotate(key, table), value.contiguous().view(self.key_shape),
                local_slots if local else full_slots)
        indices = (m.query_start_loc[:-1].long() + counts.long() - 1).clamp(0, auxiliary.shape[0] - 1)
        context = mix_features(auxiliary[indices], self.alpha[-1], self.recurrent_norm, self.eps)
        save_hidden(context, self.pool.reserved_region("draft.rnn_context"), m.state_rows, counts > 0)

    def scratch_tensors(self):
        return ()


@dataclass(frozen=True)
class DraftScratch:
    # One row per request: the block's first position is the only one that embeds a token.
    anchor: torch.Tensor
    hidden: torch.Tensor
    accumulator: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    attended: torch.Tensor
    partial: torch.Tensor
    mlp: MLPScratch
    local_logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, block, device):
        d, hidden = config.draft, config.text["hidden_size"]
        q, kv = d.heads // shard.world_size, d.kv_heads // shard.world_size
        start, end = shard.padded_partition(config.vocabulary, alignment=64)
        def empty(*shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=device)
        if tokens % block:
            raise ValueError("Inkling2 DSpark scratch holds whole blocks")
        return cls(empty(tokens // block, hidden), empty(tokens, hidden), empty(tokens, hidden, dtype=torch.float32),
            empty(tokens, (q + 2 * kv) * d.head_dim), empty(tokens, q, d.head_dim),
            empty(tokens, kv, d.head_dim), empty(tokens, q, d.head_dim), empty(tokens, hidden),
            MLPScratch.allocate(tokens, hidden, d.text["intermediate_size"] // shard.world_size, device),
            empty(tokens, end - start))

    def rows(self, tokens, samples):
        if samples != tokens:
            raise ValueError("Inkling2 DSpark projects every block position")
        block = self.hidden.shape[0] // self.anchor.shape[0]
        if tokens % block:
            raise ValueError("Inkling2 DSpark scratch holds whole blocks")
        counts = {"mlp": None, "anchor": tokens // block}
        return type(self)(*(self.mlp.rows(tokens) if field == "mlp" else getattr(self, field)[:counts.get(field, tokens)]
                           for field in self.__dataclass_fields__))


class DraftAttention:
    def __init__(self, index, weights, config, scratch, metadata, pool, *, shard, runtime, coordinator,
                 session, name, workspace):
        d = config.draft
        self.query_shape, self.key_shape = tuple(scratch.query.shape), tuple(scratch.key.shape)
        self.window, self.local = d.window, index in d.local_layers
        self.qkv = UnquantizedProjection(scratch.hidden, weights.qkv, scratch.qkv,
            session=session, name=name + ".qkv").functional(session=session)
        self.query_norm = norm(scratch.query.flatten(0, 1), weights.query_norm, scratch.query.flatten(0, 1),
            eps=d.text["rms_norm_eps"], session=session, name=name + ".q_norm")
        self.key_norm = norm(scratch.key.flatten(0, 1), weights.key_norm, scratch.key.flatten(0, 1),
            eps=d.text["rms_norm_eps"], session=session, name=name + ".k_norm")
        self.pages_per_row = (d.window + d.block_size + 127) // 128
        rows = metadata.lengths.numel()
        if self.local:
            self.key_cache, self.value_cache = (pool.reserved_region(f"draft{index}.{role}") for role in ("key", "value"))
            self.pages = persistent_copy(torch.arange(rows * self.pages_per_row, device=scratch.hidden.device,
                dtype=torch.int32).view(rows, self.pages_per_row))
            shape = (rows * self.pages_per_row, 128, *self.key_shape[1:])
            key, value = (torch.empty(shape, dtype=torch.bfloat16, device=scratch.hidden.device) for _ in range(2))
            tail = {}
            self.staging = (window_plan, d.window, self.pages_per_row)
        else:
            self.key_cache, self.value_cache = pool.key(index, draft=True), pool.value(index, draft=True)
            self.pages = None
            key, value = self.key_cache, self.value_cache
            self.tail_pages = (key.shape[1] - 1 + d.block_size + key.shape[1] - 1) // key.shape[1]
            shape = (rows * self.tail_pages, key.shape[1], *self.key_shape[1:])
            tail = {role: torch.empty(shape, dtype=key.dtype, device=key.device)
                    for role in ("tail_key", "tail_value")}
            self.staging = (tail_plan, key.shape[1], self.tail_pages)
        self.attention = PreparedPagedAttention(scratch.query, key, value,
            self.pages if self.local else metadata.page_table, metadata.lengths, metadata.query_start_loc,
            scratch.attended, mode="verify", width=d.block_size, window=-1, causal=False,
            session=session, workspace=workspace, name=name + ".attention", caller_owned_placeholders=True,
            **tail).functional(
                session=session, k_cache=None, v_cache=None, cache_seqlens=None, cu_seqlens_q=None,
                **({} if self.local else {"page_table": None, "tail_key": None, "tail_value": None}))
        heads = scratch.attended.flatten(1)
        self.output = UnquantizedProjection(heads, weights.output, scratch.partial,
            session=session, name=name + ".output").functional(session=session)
        self.reduce = PreparedAllReduce(scratch.partial, scratch.hidden, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".reduce").functional(session=session)

    def __call__(self, hidden, metadata, plan, table):
        """Attend with `plan`, this layer's staging plan, and `table`, the rows' rotary cosines and sines."""
        m = metadata
        q_width = self.query_shape[1] * self.query_shape[2]
        kv_width = self.key_shape[1] * self.key_shape[2]
        q, k, v = self.qkv(hidden).split((q_width, kv_width, kv_width), -1)
        q = self.query_norm(q.contiguous().view(-1, self.query_shape[-1])).view(self.query_shape)
        k = self.key_norm(k.contiguous().view(-1, self.key_shape[-1])).view(self.key_shape)
        q, k = rotate(q, table), rotate(k, table)
        v = v.contiguous().view(self.key_shape)
        if self.local:
            key, value = stage(self.key_cache, k, plan), stage(self.value_cache, v, plan)
            tail = {}
        else:
            key, value = self.key_cache, self.value_cache
            tail = dict(tail_key=stage(key, k, plan), tail_value=stage(value, v, plan))
        attended = self.attention(q, k_cache=key, v_cache=value, cache_seqlens=plan.lengths,
            cu_seqlens_q=m.query_start_loc, **tail, **({} if self.local else {"page_table": m.page_table}))
        return self.reduce(self.output(attended.flatten(1)))


class DraftLayer:
    def __init__(self, index, weights, owner, scratch, metadata, pool, *, session, name, workspace, runtime, coordinator):
        eps = owner.config.draft.text["rms_norm_eps"]
        source = scratch.hidden if index == 0 else scratch.accumulator
        self.input_norm = norm(source, weights.input_norm, scratch.hidden, eps=eps, session=session, name=name + ".input_norm")
        self.post_norm = norm(scratch.accumulator, weights.post_attention_norm, scratch.hidden,
            eps=eps, session=session, name=name + ".post_norm")
        self.attention = DraftAttention(index, weights, owner.config, scratch, metadata, pool, shard=owner.shard,
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".attention", workspace=workspace)
        s = scratch.mlp
        self.mlp = PreparedMLP(scratch.hidden, weights.mlp, s.gate_up, s.intermediate, s.output,
            session=session, name=name + ".mlp", workspace=workspace, activation_mode="quantized",
            round_silu=False).functional(session=session)
        self.reduce = PreparedAllReduce(s.output, scratch.hidden, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".mlp.reduce").functional(session=session)

    def __call__(self, hidden, residual, metadata, plan, table):
        if residual is None:
            residual = hidden
        else:
            hidden = hidden.float() + residual.float()
            residual = hidden.bfloat16()
        attended = self.attention(self.input_norm(hidden), metadata, plan, table)
        summed = attended.float() + residual.float()
        residual = summed.bfloat16()
        return self.reduce(self.mlp(self.post_norm(summed))), residual


class PreparedDraft:
    """Predict positions one through seven from a normalized anchor and learned masks."""

    def __init__(self, weights, scratch, metadata, pool, *, runtime, coordinator, session, name, workspace,
                 vocabulary_parallel=True, **kwargs):
        d, device = weights.config.draft, scratch.hidden.device
        self.pool, self.block = pool, d.block_size
        self.recurrent_context = pool.reserved_region("draft.rnn_context")
        self.mask = weights.draft.mask_embedding
        self.local_logit_shape = self.logit_shape = tuple(scratch.local_logits.shape)
        self.local_logit_dtype = self.logit_dtype = torch.bfloat16
        first, _ = weights.shard.padded_partition(weights.config.vocabulary, alignment=64)
        self.vocabulary_start = persistent_copy(torch.tensor([first], dtype=torch.int64, device=device))
        self.embedding_weight = weights.embedding
        self.inverse = inverse_frequencies(d, device)
        # Positions after the first take the learned mask, so only each request's first row is
        # embedded, reduced across the vocabulary shards and normalized.
        ids = metadata.token_ids[::self.block].clamp(0, weights.embedding.shape[0] - 1)
        self.embedding = PreparedEmbedding(weights.embedding, ids, scratch.anchor, session=session,
            name=name + ".embed").functional(session=session, dynamic_inputs=True)
        self.embed_reduce = PreparedAllReduce(scratch.anchor, scratch.anchor, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".embed.reduce").functional(session=session)
        self.embed_norm = norm(scratch.anchor, weights.embedding_norm, scratch.anchor,
            eps=d.text["rms_norm_eps"], session=session, name=name + ".embed.norm")
        self.layers = [DraftLayer(i, layer, weights, scratch, metadata, pool,
            runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.{i}", workspace=workspace)
            for i, layer in enumerate(weights.draft.layers)]
        # Layers with equal history geometry share one staging plan per forward.
        self.staging = tuple(dict.fromkeys(layer.attention.staging for layer in self.layers))
        self.layer_plans = tuple(self.staging.index(layer.attention.staging) for layer in self.layers)
        self.norm = norm(scratch.accumulator, weights.draft.norm, scratch.hidden,
            eps=d.text["rms_norm_eps"], session=session, name=name + ".norm")
        if weights.draft.draft_head is None:
            self.head = UnquantizedProjection(scratch.hidden, weights.head, scratch.local_logits,
                session=session, name=name + ".head").functional(session=session)
        else:
            # A quantized copy of the target's vocabulary shard serves proposals; verification keeps the target's.
            self.head = QuantizedProjection(scratch.hidden, weights.draft.draft_head, scratch.local_logits,
                session=session, name=name + ".head", activation_mode="a16",
                workspace=workspace).functional(session=session)
        self.recurrent = PreparedRecurrentProposalHead(weights.draft.recurrent,
            torch.zeros(metadata.lengths.numel(), dtype=torch.int64, device=device),
            positions=self.block - 1, session=session, name=name + ".recurrent")

    def release_outputs(self):
        return self

    def __call__(self, metadata):
        ids = metadata.token_ids[::self.block] - self.vocabulary_start
        owned = (ids >= 0) & (ids < self.embedding_weight.shape[0])
        local = self.embedding(token_ids=ids.clamp(0, self.embedding_weight.shape[0] - 1))
        anchor = self.embed_norm(self.embed_reduce(torch.where(owned[:, None], local, 0)))
        hidden = block_inputs(anchor, self.mask, self.block)
        table = rotary_table(metadata.positions, self.inverse)
        plans = [plan(metadata, *geometry) for plan, *geometry in self.staging]
        residual = None
        for layer, plan in zip(self.layers, self.layer_plans):
            hidden, residual = layer(hidden, residual, metadata, plans[plan], table)
        hidden = self.norm(hidden.float() + residual.float())[metadata.sample_indices]
        logits = self.head(hidden)
        context = torch.empty((metadata.state_rows.numel(), self.recurrent_context.shape[1]),
            dtype=self.recurrent_context.dtype, device=hidden.device)
        gather_rows(self.recurrent_context, metadata.state_rows, context)
        return logits, None, {"hidden": hidden, "recurrent_context": context}


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object
    markov: object = None


class SpeculativePreparation:
    def __init__(self, weights, pool, requests, inputs, scratch, *, decoder, completion, frame, sequence,
                 session, workspace, collectives, draft_tokens=7, page_tokens=128,
                 sampling_runtime=None, logprob_runtime=None):
        if not 0 <= draft_tokens <= 7:
            raise ValueError("Inkling2 DSpark permits zero through seven proposals per eight-position block")
        self.weights, self.pool, self.requests, self.inputs, self.scratch = weights, pool, requests, inputs, scratch
        self.tables, self.page_tokens = None, page_tokens
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace, self.collectives = session, workspace, collectives
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.forwards = DraftForwards(decoder, PreparedContext, PreparedDraft)
        self.vocabulary_parallel_sampling, self.markov_runtime = True, None
        self.sampling_runtime, self.logprob_runtime = sampling_runtime, logprob_runtime
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = (persistent_zeros(requests.generation.shape, requests.generation.dtype,
            requests.generation.device) for _ in range(2))
        self.draft_scratch = DraftScratch.allocate(weights.config, weights.shard,
            requests.capacity * self.block_size, self.block_size, pool.backing.device)

    def release_examples(self):
        self.scratch = self.draft_scratch = None
