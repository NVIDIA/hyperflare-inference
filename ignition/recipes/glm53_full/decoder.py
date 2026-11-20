# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 decoder layers and the text stack for one tensor-parallel rank.

A layer adds an attention update and an MLP update to the residual stream.
Each rank computes the part of an update its heads or intermediate channels
produce; an all-reduce sums the parts. A layer without an indexer attends to
the records the preceding indexer layer selected in the same forward.
"""

from types import SimpleNamespace

import torch
from flashinfer.triton.norm import rms_norm

from ignition.components.bf16_experts import PreparedBF16Experts
from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.paged_batch import Metadata
from ignition.components.rotary import PreparedRotaryAngles
from ignition.components.state import gather_rows
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import PAGE_TOKENS, AttentionMetadata, ExampleBuffers, PreparedAttention, Selection, rotary_frequencies
from .indexer import PreparedIndexer
from .pool import ContinuationPool
from .weights import BF16Experts, LayerWeights, ModelWeights, MoEWeights

HIDDEN, VOCABULARY = 6144, 154880


def attention_metadata(metadata: Metadata) -> AttentionMetadata:
    """Token placement of one packed batch: each live token's latent record follows from its request's pages."""
    positions = metadata.positions
    live = positions >= 0
    position = positions.clamp(min=0)
    pages = metadata.page_table[metadata.requests, position // PAGE_TOKENS].long()
    slots = torch.where(live, pages * PAGE_TOKENS + position % PAGE_TOKENS, -1)
    return AttentionMetadata(positions, slots, metadata.requests, metadata.page_table)


class PreparedMoE:
    """Sigmoid top-8 routing over 256 experts plus the shared expert, before the sum over ranks."""

    def __init__(self, config, source, weights: MoEWeights, *, session, name, workspace, buffers: ExampleBuffers):
        raw = config.raw
        rows = source.shape[0]
        experts, top = raw["n_routed_experts"], raw["num_experts_per_tok"]
        unquantized = isinstance(weights.experts, BF16Experts)
        if unquantized:
            geometry = weights.experts.down.shape[0], weights.experts.down.shape[1]
            local = weights.experts.down.shape[2]
        else:
            geometry = weights.experts.num_experts, weights.experts.hidden_size
            local = weights.experts.intermediate_size
        if geometry != (experts, HIDDEN):
            raise ValueError("GLM 5.3 routed experts require every expert on each rank")
        logits = buffers("moe.logits", rows, experts, dtype=torch.float32)
        scratch = RouterScratch(buffers("moe.selected", rows, top, dtype=torch.float32),
                                buffers("moe.ids", rows, top, dtype=torch.int32),
                                buffers("moe.weights", rows, top, dtype=torch.float32))
        self.router_projection = UnquantizedProjection(source, weights.router, logits,
                                                       session=session, name=f"{name}.router")
        self.routing = PreparedRouter(logits, scratch, session=session, name=f"{name}.route_topk",
            score_func="sigmoid", renormalize=True, routed_scaling_factor=config.routed_scale,
            correction_bias=weights.correction)
        self.shared = PreparedMLP(source, weights.shared, buffers("mlp.gate_up", rows, 2 * local),
            buffers("mlp.intermediate", rows, local), buffers("mlp.output", rows, HIDDEN), session=session,
            name=f"{name}.shared", workspace=workspace)
        if unquantized:
            # The draft layer's experts are BF16 in the checkpoint; this operator returns one row per route.
            self.experts = PreparedBF16Experts(source, weights.experts.gate_up, weights.experts.down, scratch.ids,
                scratch.weights, buffers("moe.routes", rows, top, HIDDEN),
                buffers("moe.valid", 1, dtype=torch.int32))
        else:
            self.experts = PreparedRoutedExperts(source, weights.experts, scratch.ids, scratch.weights,
                buffers("moe.routed", rows, HIDDEN), session=session, name=f"{name}.experts", workspace=workspace,
                router_plan=self.routing.plan)
        self.unquantized = unquantized

    def functional(self, *, session):
        for operator in (self.router_projection, self.routing, self.shared, self.experts):
            operator.functional(session=session)
        return self

    def __call__(self, source):
        routes = self.routing(self.router_projection(source))
        if self.unquantized:
            valid = torch.full((1,), source.shape[0], dtype=torch.int32, device=source.device)
            rows = self.experts.run(source, routes.ids, routes.weights, valid)
            routed = (rows.float() * routes.weights[:, :, None]).sum(1).to(torch.bfloat16)
        else:
            routed = self.experts(source, routes.ids, routes.weights)
        return torch.add(routed, self.shared(source))


class PreparedLayer:
    """One decoder layer of one rank, with the two all-reduces that complete its updates."""

    def __init__(self, config, index, weights: LayerWeights, pool: ContinuationPool, metadata: AttentionMetadata,
                 angles: PreparedRotaryAngles, *, rows, heads, mode, runtime, coordinator, session, name, workspace,
                 buffers: ExampleBuffers, cache_only=False):
        self.index, self.selects, self.eps, self.norms = index, config.selects(index), config.norm_eps, weights.norms
        cache, index_cache = pool.latent(index), pool.index(index)
        if self.selects != (index_cache is not None):
            raise ValueError("GLM 5.3 index key pages exist exactly for layers that select")
        source = buffers("layer.source", rows, HIDDEN)
        self.attention = PreparedAttention(weights.attention, source, metadata, cache, angles, heads=heads,
            selects=self.selects, eps=1e-6, softmax_scale=config.raw["qk_head_dim"] ** -0.5, mode=mode,
            session=session, name=f"{name}.attention", workspace=workspace, buffers=buffers,
            cache_only=cache_only)
        self.indexer = self.attention_reduce = self.mlp = self.mlp_reduce = None
        if self.selects:
            self.indexer = PreparedIndexer(weights.attention, buffers("indexer.residual", rows, 2048),
                buffers("indexer.key", rows, 128), buffers("indexer.head_weights", rows, 32, dtype=torch.float32),
                metadata, index_cache, angles, mode=mode, session=session, name=f"{name}.indexer", workspace=workspace, buffers=buffers,
                cache_only=cache_only)
        if cache_only:
            return
        self.attention_reduce = PreparedAllReduce(buffers("layer.part", rows, HIDDEN), buffers("layer.sum", rows, HIDDEN),
            runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.attention_reduce")
        if isinstance(weights.mlp, MoEWeights):
            self.mlp = PreparedMoE(config, source, weights.mlp, session=session, name=f"{name}.moe",
                                   workspace=workspace, buffers=buffers)
        else:
            local = weights.mlp.down.shape[1]
            self.mlp = PreparedMLP(source, weights.mlp, buffers("mlp.gate_up", rows, 2 * local),
                buffers("mlp.intermediate", rows, local), buffers("mlp.output", rows, HIDDEN), session=session,
                name=f"{name}.mlp", workspace=workspace)
        self.mlp_reduce = PreparedAllReduce(buffers("layer.part", rows, HIDDEN), buffers("layer.sum", rows, HIDDEN),
            runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.mlp_reduce")

    def functional(self, *, session):
        for operator in (self.attention, self.indexer, self.attention_reduce, self.mlp, self.mlp_reduce):
            if operator is not None:
                operator.functional(session=session)
        return self

    def _normalized(self, hidden, name):
        normalized = torch.empty_like(hidden)
        rms_norm(hidden, self.norms[name], normalized, self.eps, round_before_weight=True)
        return normalized

    def store(self, hidden, metadata, angles):
        """Store this layer's latent records and index keys of one input, without attending."""
        key = self.attention.store(self._normalized(hidden, "input_layernorm"), metadata, angles)
        if self.indexer is not None:
            self.indexer.store(key, metadata, angles)

    def attention_part(self, hidden, metadata, angles, selection, *, reuse_selection=False):
        """This rank's part of the attention update, and the selection the layer attended to."""
        projected = self.attention.project(self._normalized(hidden, "input_layernorm"), metadata, angles)
        if self.indexer is not None and not reuse_selection:
            selection = self.indexer(projected.query_residual, projected.index_key, projected.index_weights,
                                     metadata, angles)
        elif selection is None:
            raise ValueError("a GLM 5.3 layer without an indexer requires the preceding selection")
        return self.attention.attend(projected.absorbed, selection, metadata), selection

    def mlp_part(self, hidden):
        """This rank's part of the MLP update."""
        return self.mlp(self._normalized(hidden, "post_attention_layernorm"))

    def __call__(self, hidden, metadata, angles, selection: Selection | None, *, reuse_selection=False):
        part, selection = self.attention_part(hidden, metadata, angles, selection, reuse_selection=reuse_selection)
        hidden = torch.add(hidden, self.attention_reduce(part))
        return torch.add(hidden, self.mlp_reduce(self.mlp_part(hidden))), selection


class PreparedDecoder:
    """Embedding, every decoder layer, final normalization, and the partitioned vocabulary projection.

    The forward takes one packed batch and returns the normalized hidden state
    and logits of its sampled rows. Its operators allocate their outputs inside
    the forward; `scratch.logits` is an example of the logits until
    `release_outputs`.
    """

    def __init__(self, weights: ModelWeights, metadata: Metadata, pool: ContinuationPool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, examples: ExampleBuffers | None = None):
        config, shard = weights.config, weights.shard
        tokens, samples = metadata.token_ids.shape[0], metadata.sample_indices.shape[0]
        device = weights.embedding.device
        self.weights, self.pool, self.eps, self.mode = weights, pool, config.norm_eps, mode
        self.auxiliary = None
        # With a draft layer, the forward also returns what the draft's accepted context reads.
        self.feeds_draft = weights.draft is not None
        buffers = ExampleBuffers(device) if examples is None else examples
        example = AttentionMetadata(metadata.positions, buffers("slots", tokens, dtype=torch.int64),
                                    metadata.requests, metadata.page_table)
        self.frequencies = rotary_frequencies(config.rope_theta, device)
        self.angles = PreparedRotaryAngles(metadata.positions, self.frequencies)
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, buffers("layer.sum", tokens, HIDDEN),
            num_rows=metadata.live_counts[:1], session=session, name=f"{name}.embedding")
        self.layers = tuple(PreparedLayer(config, index, layer, pool, example, self.angles, rows=tokens,
            heads=config.raw["num_attention_heads"] // shard.world_size, mode=mode, runtime=runtime,
            coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
            buffers=buffers)
            for index, layer in enumerate(weights.layers))
        self.scratch = SimpleNamespace(logits=buffers("head.logits", samples, VOCABULARY))
        self.head = PreparedVocabularyProjection(buffers("head.source", samples, HIDDEN), weights.head,
            buffers("head.local", samples, weights.head.shape[0]), self.scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=f"{name}.lm_head",
            workspace=workspace)
        self.embedding.functional(session=session, dynamic_inputs=True)
        self.angles.functional()
        for layer in self.layers:
            layer.functional(session=session)
        self.head.functional(session=session)

    def release_outputs(self):
        self.scratch = None

    def commit(self, counts, records, metadata):
        """Verification rows store their records in the requests' pages; rows past the accepted count stay
        beyond each request's consumed length, where no later row reads them before they are rewritten."""

    def __call__(self, metadata: Metadata, *, observe=None):
        """Return (normalized hidden, logits, draft inputs, None) for the sampled rows.

        The draft inputs are None without a draft layer. Otherwise they are
        every row's token embedding and normalized final hidden state, and the
        hidden state each request's continuation row held before this forward.
        `observe(index, hidden, selection)` sees each layer's output; tests use it.
        """
        placement = attention_metadata(metadata)
        saved = None
        if self.feeds_draft:
            saved = gather_rows(self.pool.last_hidden, metadata.state_rows,
                torch.empty((metadata.state_rows.shape[0], HIDDEN), dtype=torch.bfloat16,
                            device=metadata.state_rows.device))
        embedded = hidden = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        angles = self.angles(metadata.positions)
        selection = None
        for layer in self.layers:
            hidden, selection = layer(hidden, placement, angles, selection)
            if observe is not None:
                observe(layer.index, hidden, selection)
        if self.feeds_draft:
            every = torch.empty_like(hidden)
            rms_norm(hidden, self.weights.norm, every, self.eps, round_before_weight=True)
            normalized = every[metadata.sample_indices]
            return normalized, self.head(normalized), (embedded, every, saved), None
        sampled = hidden[metadata.sample_indices]
        normalized = torch.empty_like(sampled)
        rms_norm(sampled, self.weights.norm, normalized, self.eps, round_before_weight=True)
        return normalized, self.head(normalized), None, None
