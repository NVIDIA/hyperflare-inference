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
from ignition.components.collectives import PreparedAllReduce, PreparedAllReduceRMSNorm
from ignition.components.embedding import PreparedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.paged_batch import Metadata
from ignition.components.rotary import PreparedRotaryAngles
from ignition.components.state import gather_rows
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.weight_first_projection import PreparedWeightFirstProjection
from ignition.components.weight_prefetch import PreparedWeightPrefetch, join_weight_prefetch
from ignition.preparation.tuning import register_finalizer

from .attention import PAGE_TOKENS, AttentionMetadata, ExampleBuffers, PreparedAttention, Selection, rotary_frequencies
from .indexer import PreparedIndexer, index_tables
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
    visible = (positions + 1).clamp(min=0).to(torch.int32)
    return AttentionMetadata(positions, slots, metadata.requests, metadata.page_table,
                             index_tables(metadata.page_table, metadata.requests), visible)


class PreparedMoE:
    """Sigmoid top-8 routing over 256 experts plus the shared expert, before the sum over ranks."""

    def __init__(self, config, source, weights: MoEWeights, *, session, name, workspace, buffers: ExampleBuffers,
                 overlap_gate_up=False):
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
        self.concurrent_shared = None
        if overlap_gate_up and workspace.overlap and not unquantized:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared, projected_gate_up=True,
                parallel_plans=(self.routing.plan, self.experts.plan), session=session, workspace=workspace)
        self.unquantized = unquantized

    def functional(self, *, session):
        for operator in (self.router_projection, self.routing, self.shared, self.experts):
            operator.functional(session=session)
        if self.concurrent_shared is not None:
            self.concurrent_shared.functional(session=session)
        return self

    def contributions(self, source, projections=None):
        """Return rank-local routed and shared updates before their BF16 sum."""
        logits = self.router_projection(source) if projections is None else projections[0]
        pending = (self.concurrent_shared.start(projections[1])
                   if projections is not None and self.concurrent_shared is not None else None)
        routes = self.routing(logits)
        if self.unquantized:
            valid = torch.full((1,), source.shape[0], dtype=torch.int32, device=source.device)
            rows = self.experts.run(source, routes.ids, routes.weights, valid)
            routed = (rows.float() * routes.weights[:, :, None]).sum(1).to(torch.bfloat16)
        else:
            routed = self.experts(source, routes.ids, routes.weights)
        if pending is not None:
            shared = self.concurrent_shared.finish(pending, routed)
        else:
            shared = self.shared(source) if projections is None else self.shared.from_gate_up(projections[1])
        return routed, shared

    def __call__(self, source, projections=None):
        return torch.add(*self.contributions(source, projections))


class PreparedLayer:
    """One decoder layer of one rank, with the two all-reduces that complete its updates."""

    def __init__(self, config, index, weights: LayerWeights, pool: ContinuationPool, metadata: AttentionMetadata,
                 angles: PreparedRotaryAngles, *, rows, heads, mode, runtime, coordinator, session, name, workspace,
                 buffers: ExampleBuffers, cache_only=False, weight_first=False, fuse_reductions=False,
                 following_norm=None, trigger_following=None):
        self.attention_norm = self.moe_projection = None
        self.use_attention_norm = self.use_weight_first = self.use_feed_forward_norm = False
        self.feed_forward_norm = None
        self.index, self.selects, self.eps, self.norms = index, config.selects(index), config.norm_eps, weights.norms
        self.following_prefetch = self.output_prefetch = None
        cache, index_cache = pool.latent(index), pool.index(index)
        if self.selects != (index_cache is not None):
            raise ValueError("GLM 5.3 index key pages exist exactly for layers that select")
        source = buffers("layer.source", rows, HIDDEN)
        self.attention = PreparedAttention(weights.attention, source, metadata, cache, angles, heads=heads,
            selects=self.selects, eps=1e-6, softmax_scale=config.qk_head_dim ** -0.5, mode=mode,
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
                                   workspace=workspace, buffers=buffers, overlap_gate_up=weight_first and rows == 4)
        else:
            local = weights.mlp.down.shape[1]
            self.mlp = PreparedMLP(source, weights.mlp, buffers("mlp.gate_up", rows, 2 * local),
                buffers("mlp.intermediate", rows, local), buffers("mlp.output", rows, HIDDEN), session=session,
                name=f"{name}.mlp", workspace=workspace)
        weight_first_moe = (weight_first and isinstance(weights.mlp, MoEWeights)
                            and not isinstance(weights.mlp.experts, BF16Experts)
                            and PreparedWeightFirstProjection.fits(source, weights.mlp.router, weights.mlp.shared.gate_up)
                            and tuple(weights.mlp.router.shape) == (256, HIDDEN)
                            and tuple(weights.mlp.shared.gate_up.shape) == (512, HIDDEN))
        fused_fits = PreparedAllReduceRMSNorm.fits(
            runtime, source, round_before_weight=True, reduction_order="shard_owner")
        if (fuse_reductions or weight_first_moe) and fused_fits:
            self.attention_norm = PreparedAllReduceRMSNorm(buffers("layer.part", rows, HIDDEN), source,
                weights.norms["post_attention_layernorm"], buffers("layer.norm", rows, HIDDEN),
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.attention_reduce_norm",
                eps=self.eps, reduction_dtype=torch.bfloat16, round_before_weight=True, reduction_order="shard_owner",
                trigger_dependents=weight_first_moe)
            if weight_first_moe:
                self.moe_projection = PreparedWeightFirstProjection(source, weights.mlp.router, weights.mlp.shared.gate_up,
                                                                     session=session, name=f"{name}.moe.input_projections")
            register_finalizer(session, self._select_weight_first)
        self.mlp_reduce = PreparedAllReduce(buffers("layer.part", rows, HIDDEN), buffers("layer.sum", rows, HIDDEN),
            runtime=runtime, session=session, coordinator=coordinator, name=f"{name}.mlp_reduce")

        if following_norm is not None and fused_fits:
            trigger = (not config.selects(index + 1)) if trigger_following is None else bool(trigger_following)
            self.feed_forward_norm = PreparedAllReduceRMSNorm(
                buffers("layer.part", rows, HIDDEN), source, following_norm, buffers("layer.norm", rows, HIDDEN),
                input_addend=buffers("mlp.output", rows, HIDDEN) if isinstance(self.mlp, PreparedMoE) else None,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.mlp_reduce_norm",
                eps=self.eps, reduction_dtype=torch.bfloat16, round_before_weight=True, reduction_order="shard_owner",
                trigger_dependents=trigger)
            register_finalizer(session, self._select_feed_forward_norm)

    def _select_feed_forward_norm(self):
        from b12x.preparation.types import require_prepared

        query = require_prepared(self.mlp_reduce.plan, "comm.pcie").query
        self.use_feed_forward_norm = (query.surface == "PCIeTwoShotBF16.all_reduce"
                                      and query.call.get("payload_schedule") != "pairwise")

    def _select_weight_first(self):
        from b12x.preparation.types import require_prepared

        # Different transports can accumulate BF16 rank contributions in a different order.
        # The fused producer uses the shard owner's two-shot order.
        query = require_prepared(self.attention_reduce.plan, "comm.pcie").query
        enabled = (query.surface == "PCIeTwoShotBF16.all_reduce"
                   and query.call.get("payload_schedule") != "pairwise")
        self.use_attention_norm = enabled
        self.use_weight_first = enabled and self.moe_projection is not None

    def functional(self, *, session):
        for operator in (self.attention, self.indexer, self.attention_reduce, self.attention_norm, self.mlp, self.mlp_reduce, self.feed_forward_norm):
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

    def attention_part(self, hidden, metadata, angles, selection, *, reuse_selection=False, normalized=None):
        """This rank's part of the attention update, and the selection the layer attended to."""
        if normalized is None:
            normalized = self._normalized(hidden, "input_layernorm")
        projected = self.attention.project(normalized, metadata, angles)
        if self.indexer is not None and not reuse_selection:
            selection = self.indexer(projected.query_residual, projected.index_key, projected.index_weights,
                                     metadata, angles)
        elif selection is None:
            raise ValueError("a GLM 5.3 layer without an indexer requires the preceding selection")
        return self.attention.attend(projected.absorbed, selection, metadata), selection

    def mlp_part(self, hidden):
        """This rank's part of the MLP update."""
        return self.mlp(self._normalized(hidden, "post_attention_layernorm"))

    def forward_layer(self, hidden, metadata, angles, selection, *, normalized=None, reuse_selection=False,
                      prefetch_sums=None, prepare_next=False):
        if prefetch_sums is not None and self.output_prefetch is not None:
            self.output_prefetch(hidden, prefetch_sums)
        part, selection = self.attention_part(hidden, metadata, angles, selection, normalized=normalized,
                                              reuse_selection=reuse_selection)
        projections = None
        if self.use_attention_norm:
            hidden = hidden.clone()
            source = self.attention_norm(part, hidden)
            if self.use_weight_first:
                projections = self.moe_projection(source)
        else:
            hidden = torch.add(hidden, self.attention_reduce(part))
            source = self._normalized(hidden, "post_attention_layernorm")
        fuse = prepare_next and self.use_feed_forward_norm
        addend = None
        if fuse and isinstance(self.mlp, PreparedMoE):
            local, addend = self.mlp.contributions(source, projections)
        else:
            local = self.mlp(source) if projections is None else self.mlp(source, projections)
        if prefetch_sums is not None and self.following_prefetch is not None:
            self.following_prefetch(local, prefetch_sums)
        if fuse:
            next_input = self.feed_forward_norm(local, hidden, input_addend=addend)
            return hidden, selection, next_input
        return torch.add(hidden, self.mlp_reduce(local)), selection, None

    def __call__(self, hidden, metadata, angles, selection: Selection | None, *, reuse_selection=False, prefetch_sums=None):
        hidden, selection, _ = self.forward_layer(hidden, metadata, angles, selection, reuse_selection=reuse_selection,
                                                  prefetch_sums=prefetch_sums)
        return hidden, selection


class PreparedDecoder:
    """Embedding, every decoder layer, final normalization, and the partitioned vocabulary projection.

    The forward takes one packed batch and returns the normalized hidden state
    and logits of its sampled rows. Its operators allocate their outputs inside
    the forward; `scratch.logits` is an example of the logits until
    `release_outputs`.
    """

    def __init__(self, weights: ModelWeights, metadata: Metadata, pool: ContinuationPool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, examples: ExampleBuffers | None = None,
                 overlap=False):
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
        small_tp8_decode = (mode == "decode" and tokens <= 64 and shard.world_size == 8
                            and device.type == "cuda" and torch.cuda.get_device_capability(device) == (12, 0))
        weight_first = small_tp8_decode and tokens <= 16 and getattr(session, "collective_tuning", "static") == "static"
        layer_count = len(weights.layers)

        def layer_following_norm(index):
            if not weight_first:
                return None
            if index + 1 < layer_count:
                return weights.layers[index + 1].norms["input_layernorm"]
            if self.feeds_draft or samples == tokens:
                return weights.norm
            return None

        self.layers = tuple(PreparedLayer(config, index, layer, pool, example, self.angles, rows=tokens,
            heads=config.raw["num_attention_heads"] // shard.world_size, mode=mode, runtime=runtime,
            coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
            buffers=buffers, weight_first=weight_first, fuse_reductions=weight_first,
            following_norm=layer_following_norm(index),
            trigger_following=(not config.selects(index + 1)) if index + 1 < layer_count else False)
            for index, layer in enumerate(weights.layers))
        self.stream_handle = None
        if overlap and small_tp8_decode:
            stream = workspace.auxiliary_stream
            self.stream_handle = stream.cuda_stream
            # Narrower reads leave more memory bandwidth for attention and the feed-forward reduction at three and four rows.
            output_programs, following_programs = (12, 24) if tokens in (3, 4) else (32, 64)
            if tokens <= 8:
                for layer, source in zip(self.layers, weights.layers, strict=True):
                    layer.output_prefetch = PreparedWeightPrefetch([source.attention["o_proj"]], output_programs, stream)
            for layer, following in zip(self.layers[:-1], weights.layers[1:], strict=True):
                attention = following.attention
                names = ("input_proj", "q_b_proj", "indexer.wq_b")
                targets = [attention[key] for key in names if key in attention]
                layer.following_prefetch = PreparedWeightPrefetch(targets, following_programs, stream)
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
        sums = None if self.stream_handle is None else torch.empty(64, dtype=torch.int64, device=hidden.device)
        normalized_input = None
        for layer in self.layers:
            hidden, selection, normalized_input = layer.forward_layer(
                hidden, placement, angles, selection, normalized=normalized_input, prefetch_sums=sums, prepare_next=True)
            if observe is not None:
                observe(layer.index, hidden, selection)
        if sums is not None:
            join_weight_prefetch(hidden, sums, self.stream_handle)
        if self.feeds_draft:
            if normalized_input is not None:
                every = normalized_input
            else:
                every = torch.empty_like(hidden)
                rms_norm(hidden, self.weights.norm, every, self.eps, round_before_weight=True)
            normalized = every[metadata.sample_indices]
            return normalized, self.head(normalized), (embedded, every, saved), None
        if normalized_input is not None:
            normalized = normalized_input[metadata.sample_indices]
        else:
            sampled = hidden[metadata.sample_indices]
            normalized = torch.empty_like(sampled)
            rms_norm(sampled, self.weights.norm, normalized, self.eps, round_before_weight=True)
        return normalized, self.head(normalized), None, None
