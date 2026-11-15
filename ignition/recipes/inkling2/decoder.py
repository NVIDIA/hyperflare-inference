# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel target execution with complete accepted-state commits.

Ranks on one host reduce FP32 partial sums. Ranks on separate hosts reduce through RoCEnante,
which carries BF16: each rank rounds its partial sum to BF16 before the reduction.

Between reductions a layer's stages are row-local: the attention-residual mixtures with their norms,
the output convolutions, the expert router and the residual additions. Every rank computes them for
every row, except in prefill chunks of at least `SHARDED_PREFILL_ROWS` rows over RoCEnante, whose
rows are sharded across ranks (`ignition.components.row_shards`). There each reduction is a
reduce-scatter, a rank keeps its own rows of the residual state, and shard gathers supply every rank
what attention heads and experts, which are sharded by column, read for every row: the attention
input streams, the expert input with its routes, the drafter taps and the final normalized rows.
The first layer normalizes the replicated embedding rows, so its attention needs no gather.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from ignition.components.causal_convolution import PreparedCausalConvolution
from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllReduce, _roce
from ignition.components.linear import UnquantizedProjection
from ignition.components.rms_norm import PreparedTorchRMSNorm
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.row_shards import PreparedRouteGather, RowShardCollectives, RowShards, row_sharding_fits
from ignition.preparation.examples import declaration_tensor

from .attention import LayerCache, PreparedAttention
from .residual import PreparedResidualMix
from .vocabulary import PreparedLogits, embedding_rows


@dataclass(frozen=True)
class Scratch:
    logits: torch.Tensor
    local_logits: torch.Tensor
    tokens: int

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device):
        first, last = shard.padded_partition(config.text.vocabulary, alignment=64)
        return cls(torch.zeros((samples, config.text.valid_vocabulary), dtype=torch.float32, device=device),
                   torch.zeros((samples, last - first), dtype=torch.float32, device=device), tokens)

    def rows(self, tokens, samples):
        return type(self)(self.logits[:samples], self.local_logits[:samples], tokens)


# The smallest prefill capacity whose rows are sharded. A sharded layer adds three small collectives
# (two edge exchanges and the route gather) to the wire bytes of its two reductions, and a global
# layer two more row gathers; below this capacity the row-local stages they save are shorter than
# those collectives are estimated to take.
SHARDED_PREFILL_ROWS = 2048


def reduction_dtype(runtime):
    """Element type of cross-rank hidden reductions: BF16 over RoCEnante, FP32 on one host."""
    return torch.bfloat16 if runtime is not None and _roce(runtime) else torch.float32


@dataclass(frozen=True)
class ResidualState:
    blocks: tuple[torch.Tensor, ...]
    partial: torch.Tensor


class PreparedLayer(torch.nn.Module):
    """One decoder layer. With `rows`, its row-local stages run on this rank's row shard of the chunk."""

    def __init__(self, layer, weights, text, pool, metadata, reduction, reduction_dtype, *, mode, tokens,
                 session, workspace, name, examples=None, rows=None, route_gather=None):
        super().__init__()
        self.layer, self.weights, self.text, self.mode = layer, weights, text, mode
        self.reduction, self.reduction_dtype = reduction, reduction_dtype
        self.rows, self.route_gather = rows, route_gather
        # Rows of the row-local stages: the whole chunk, or this rank's shard of it.
        local = tokens if rows is None else rows.shards.rows
        self.attention = PreparedAttention(layer, weights.attention, text, LayerCache.of(pool, layer), metadata,
            mode=mode, tokens=tokens, session=session, workspace=workspace, name=name + ".attention", examples=examples,
            # Prefill writes the attention partial sum in the reduction's element type. At prefill row counts the
            # BF16 projection equals the FP32 projection rounded once; decode and verification keep FP32.
            # An MXFP8 output weight writes BF16 in every mode.
            partial_dtype=reduction_dtype if mode == "prefill" else torch.float32,
            tuning_page_count=((int(pool.memory_capacity[0].item()) - 1) * pool.ring_row_pages + pool.geometry.ring // 128
                               if text.local(layer) else int(pool.memory_capacity[1].item())))
        device = weights.router.device
        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "inkling.mlp." + role, shape, dtype, device)
        source = example("source", (tokens, text.hidden))
        local_source = example("source", (local, text.hidden))
        logits = example("router_logits", (local, text.experts), torch.float32)
        # BF16 operands with the sum carried in FP32: each product of two BF16 values is exact in FP32.
        self.router_projection = UnquantizedProjection(local_source, weights.router, logits, session=session,
            name=name + ".router", accumulation="fp32").functional(session=session)
        routes = RouterScratch.allocate(tokens, device, experts=text.experts, top_k=text.experts_per_token)
        self.router = PreparedRouter(logits, routes.rows(local), session=session, name=name + ".routing",
            score_func="sigmoid", renormalize=True, routed_scaling_factor=text.route_scale,
            correction_bias=weights.correction, num_valid_tokens=metadata.live_counts[:1]).functional(session=session)
        self.experts = PreparedRoutedExperts(source, weights.experts, routes.ids, routes.weights,
            example("partial", source.shape), session=session, name=name + ".experts", workspace=workspace,
            activation_mode="a4" if mode == "prefill" else "a16").functional(session=session)
        self.attention_convolution = PreparedCausalConvolution(weights.attention_convolution,
            pool.history(layer, "attention_convolution"))
        self.mlp_convolution = PreparedCausalConvolution(weights.mlp_convolution, pool.history(layer, "mlp_convolution"))
        blocks = (layer + text.residual_block - 1) // text.residual_block
        self.input_mix = (None if weights.attention_query is None else PreparedResidualMix(local_source,
            weights.attention_query, weights.input_norms, sources=blocks + 1, eps=text.eps,
            session=session, name=name + ".attention_input", examples=examples))
        self.input_norms = (() if self.input_mix is not None else tuple(PreparedTorchRMSNorm(source, norm,
            example(f"input_norm{index}", source.shape), eps=text.eps, session=session, name=f"{name}.input_norm{index}").functional(session=session)
            for index, norm in enumerate(weights.input_norms)))
        self.mlp_mix = PreparedResidualMix(local_source, weights.mlp_query, (weights.mlp_norm,),
            sources=blocks + (2 if layer % text.residual_block == 0 else 1), eps=text.eps,
            session=session, name=name + ".mlp_input", addend=layer % text.residual_block != 0, examples=examples)

    def forward(self, state, metadata, shard=None, replicated=None):
        if self.rows is not None:
            return self._forward_rows(state, metadata, shard, replicated)
        w, t, m = self.weights, self.text, metadata
        streams = (tuple(norm(state.partial) for norm in self.input_norms) if self.input_mix is None
                   else self.input_mix((*state.blocks, state.partial)))
        _, attention_partial, attention_values = self.attention.attend(streams, m)
        attention_raw = self.reduction(attention_partial.to(self.reduction_dtype)).to(torch.bfloat16)
        attention, attention_previous = self.attention_convolution(attention_raw, m)
        if self.layer % t.residual_block == 0:
            mlp_input = self.mlp_mix((*state.blocks, state.partial, attention))[0]
        else:
            mlp_input = self.mlp_mix((*state.blocks, state.partial), addend=attention)[0]
        routes = self.router(self.router_projection(mlp_input), num_valid_tokens=m.live_counts[:1])
        expert_partial = self.experts(mlp_input, routes.ids, routes.weights)
        mlp_raw = self.reduction(expert_partial.to(self.reduction_dtype)).to(torch.bfloat16)
        mlp, mlp_previous = self.mlp_convolution(mlp_raw, m)
        if self.mode != "verify":
            self.attention_convolution.commit(attention_previous, attention_raw, m)
            self.mlp_convolution.commit(mlp_previous, mlp_raw, m)
        records = (attention_values, attention_previous, attention_raw, mlp_previous, mlp_raw)
        if self.layer % t.residual_block == 0:
            state = ResidualState((*state.blocks, state.partial), (attention.float() + mlp.float()).to(torch.bfloat16))
        else:
            state = ResidualState(state.blocks, (state.partial.float() + attention.float() + mlp.float()).to(torch.bfloat16))
        return state, records

    def _forward_rows(self, state, m, shard, replicated):
        """The prefill layer over this rank's rows of the residual state.

        `shard` is this rank's view of the metadata `m`. `replicated` holds every row of the state's
        partial before any mixture exists, where the layer normalizes the rows themselves.
        """
        t, rows = self.text, self.rows
        halo = rows.shards.halo
        if self.input_mix is None:
            streams = tuple(norm(replicated) for norm in self.input_norms)
        else:
            streams = tuple(rows.gather(stream) for stream in self.input_mix((*state.blocks, state.partial)))
        _, attention_partial, _ = self.attention.attend(streams, m)
        attention_raw, attention_tail = rows.reduce(attention_partial.to(self.reduction_dtype), shard)
        attention, attention_previous = self.attention_convolution(attention_raw, shard.convolution)
        attention = attention[halo:]
        if self.layer % t.residual_block == 0:
            mlp_input = self.mlp_mix((*state.blocks, state.partial, attention))[0]
        else:
            mlp_input = self.mlp_mix((*state.blocks, state.partial), addend=attention)[0]
        routes = self.router(self.router_projection(mlp_input), num_valid_tokens=shard.live_count)
        ids, weights = self.route_gather(routes.ids, routes.weights)
        expert_partial = self.experts(rows.gather(mlp_input), ids, weights)
        mlp_raw, mlp_tail = rows.reduce(expert_partial.to(self.reduction_dtype), shard)
        mlp, mlp_previous = self.mlp_convolution(mlp_raw, shard.convolution)
        mlp = mlp[halo:]
        # Each request's last raw rows reach every rank through the edge exchange, so history stays replicated.
        self.attention_convolution.commit(attention_previous, attention_tail, shard.tail)
        self.mlp_convolution.commit(mlp_previous, mlp_tail, shard.tail)
        if self.layer % t.residual_block == 0:
            return ResidualState((*state.blocks, state.partial), (attention.float() + mlp.float()).to(torch.bfloat16)), ()
        return ResidualState(state.blocks, (state.partial.float() + attention.float() + mlp.float()).to(torch.bfloat16)), ()

    def commit(self, records, counts, metadata):
        values, attention_previous, attention_raw, mlp_previous, mlp_raw = records
        self.attention.commit(values, counts, metadata)
        self.attention_convolution.commit(attention_previous, attention_raw, metadata, counts=counts)
        self.mlp_convolution.commit(mlp_previous, mlp_raw, metadata, counts=counts)


class PreparedStack:
    """The decoder layers and the final mixture: normalized embedding rows to final normalized rows and drafter taps.

    With `rows`, the residual state holds this rank's row shard of the chunk; the taps and the final
    rows are gathered, so callers receive every row either way.
    """

    def __init__(self, weights, pool, metadata, reduction, reduction_dtype, *, mode, tokens, session, workspace,
                 name, examples=None, rows=None, route_gather=None):
        text = weights.config.text
        self.mode, self.rows = mode, rows
        self.taps = weights.config.draft.taps if weights.draft is not None else ()
        self.layers = [PreparedLayer(index, layer, text, pool, metadata, reduction, reduction_dtype, mode=mode,
            tokens=tokens, session=session, workspace=workspace, name=f"{name}.layer{index}", examples=examples,
            rows=rows, route_gather=route_gather)
            for index, layer in enumerate(weights.layers)]
        source = declaration_tensor(examples, "inkling.decoder.embedded",
            (tokens if rows is None else rows.shards.rows, text.hidden), torch.bfloat16, weights.final_norm.device)
        self.output_mix = PreparedResidualMix(source, weights.output_query, (weights.final_norm,),
            sources=(len(weights.layers) + text.residual_block - 1) // text.residual_block + 1,
            eps=text.eps, session=session, name=name + ".output_mix", examples=examples)

    def __call__(self, hidden, metadata):
        rows, shard = self.rows, None
        state, records, taps = ResidualState((), hidden), [], []
        if rows is not None:
            shard = rows.shards.metadata(metadata)
            state = ResidualState((), hidden.index_select(0, rows.shards.own_rows))
        for index, layer in enumerate(self.layers):
            if index in self.taps:
                taps.append(state.partial if rows is None else rows.gather(state.partial))
            if rows is None:
                state, record = layer(state, metadata)
            else:
                state, record = layer(state, metadata, shard, hidden if index == 0 else None)
            if self.mode == "verify":
                records.append(record)
            del record
        normalized = self.output_mix((*state.blocks, state.partial))[0]
        if rows is not None:
            normalized = rows.gather(normalized)
        if self.taps:
            taps.append(normalized)
        return normalized, taps, records


class PreparedDecoder:
    def __init__(self, weights, tables, scratch, metadata, pool, *, mode, session, name, workspace,
                 runtime, vocabulary_runtime, coordinator, examples=None, vision_state=None,
                 vocabulary_parallel=False, **kwargs):
        del tables, kwargs
        self.weights, self.pool, self.mode, self.scratch = weights, pool, mode, scratch
        self.vision_state, self.vocabulary_parallel = vision_state, vocabulary_parallel
        text, tokens, samples = weights.config.text, scratch.tokens, scratch.logits.shape[0]
        device = weights.embedding.device
        examples = {} if examples is None else examples
        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "inkling.decoder." + role, shape, dtype, device)
        # Traced forwards read this attribute, never the runtime, which compile workers see as a placeholder.
        self.reduction_dtype = reduction_dtype(runtime)
        partial = example("partial", (tokens, text.hidden), self.reduction_dtype)
        self.reduce = PreparedChunkedAllReduce(partial, example("reduced", partial.shape, self.reduction_dtype),
            runtime=runtime, coordinator=coordinator, session=session, name=name + ".reduction").functional(session=session)
        rows = route_gather = None
        halo = text.convolution_width - 1
        if mode == "prefill" and row_sharding_fits(runtime, partial, halo=halo, minimum_rows=SHARDED_PREFILL_ROWS):
            shards = RowShards.of(weights.rank, weights.world, tokens, device, halo=halo)
            collectives = dict(runtime=runtime, session=session, coordinator=coordinator, examples=examples)
            rows = RowShardCollectives(shards, text.hidden, requests=metadata.lengths.numel(),
                                       name=name + ".rows", **collectives)
            route_gather = PreparedRouteGather(shards, text.experts_per_token, name=name + ".route_gather", **collectives)
        self.stack = PreparedStack(weights, pool, metadata, self.reduce, self.reduction_dtype, mode=mode, tokens=tokens,
            session=session, workspace=workspace, name=name, examples=examples, rows=rows, route_gather=route_gather)
        self.layers = self.stack.layers
        self.logits = PreparedLogits(weights.head, text, weights.rank, tokens=samples,
                                     session=session, name=name + ".logits", examples=examples)
        local = example("local_logits", (samples, weights.head.shape[0]), torch.float32)
        gathered = example("gathered_logits", (samples, weights.head.shape[0] * weights.world), torch.float32)
        self.gather = PreparedAllGather(local.view(torch.bfloat16), gathered.view(torch.bfloat16),
            runtime=vocabulary_runtime, coordinator=coordinator, session=session,
            name=name + ".logit_gather").functional(session=session)
        self.head = SimpleNamespace(gather=SimpleNamespace(runtime=vocabulary_runtime))
        self.first_token = torch.full((1,), weights.rank * weights.embedding.shape[0], dtype=torch.int64, device=device)
        self.auxiliary = (example("auxiliary", (tokens, len(weights.config.draft.taps), text.hidden))
                          if weights.draft is not None else None)
        self.local_logits = scratch.local_logits
        embedded = example("embedded", (tokens, text.hidden))
        self.embedding_norm = PreparedTorchRMSNorm(embedded, weights.embedding_norm, example("normalized_embedding", embedded.shape),
            eps=text.eps, session=session, name=name + ".embedding_norm").functional(session=session)

    def release_outputs(self):
        self.scratch = self.local_logits = self.auxiliary = None

    def __call__(self, metadata):
        w, t, m = self.weights, self.weights.config.text, metadata
        embedded = embedding_rows(w.embedding, m.token_ids, self.first_token)
        hidden = self.reduce(embedded.to(self.reduction_dtype)).to(torch.bfloat16)
        hidden = self.embedding_norm(hidden)
        if self.vision_state is not None and self.mode == "prefill":
            self.vision_state.inject(m, hidden)
        hidden = torch.where((m.positions >= 0)[:, None], hidden, 0)
        normalized, taps, records = self.stack(hidden, m)
        sampled_hidden = normalized[m.sample_indices]
        local = self.logits(sampled_hidden)
        logits = local if self.vocabulary_parallel else self.gather(local.view(torch.bfloat16)).view(torch.float32)[:, :t.valid_vocabulary].contiguous()
        auxiliary = torch.stack(taps, dim=1) if taps else None
        return sampled_hidden, logits, auxiliary, tuple(records)

    def commit(self, counts, records, metadata):
        for layer, record in zip(self.layers, records, strict=True):
            layer.commit(record, counts, metadata)
