# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mini layer visits and message-bank execution with independent cache histories."""

from types import SimpleNamespace

import torch

from ignition.components.causal_convolution import PreparedCausalConvolution
from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllReduce
from ignition.components.linear import UnquantizedProjection
from ignition.components.rms_norm import PreparedTorchRMSNorm
from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.preparation.examples import declaration_tensor

from .attention import LayerCache, PreparedAttention
from .residual import PreparedResidualMix
from .loop import PreparedController
from .decoder import PreparedDecoder, PreparedLayer, ResidualState, reduction_dtype
from .vocabulary import PreparedLogits, embedding_rows


class PreparedLoopLayer(PreparedLayer):
    def __init__(self, layer, weights, text, pool, metadata, reduction, reduction_dtype, *, mode, tokens,
                 session, workspace, name, examples=None, cache_layer=None, messages=False):
        torch.nn.Module.__init__(self)
        self.layer, self.weights, self.text, self.mode = layer, weights, text, mode
        self.reduction, self.reduction_dtype = reduction, reduction_dtype
        cache_layer = layer if cache_layer is None else cache_layer
        self.attention = PreparedAttention(layer, weights.attention, text, LayerCache.of(pool, cache_layer), metadata,
            mode=mode, tokens=tokens, session=session, workspace=workspace, name=name + ".attention", examples=examples,
            tuning_page_count=((int(pool.memory_capacity[0].item()) - 1) * pool.ring_row_pages + pool.geometry.ring // 128
                               if text.local(layer) else int(pool.memory_capacity[1].item())))
        device = weights.router.device
        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "inkling.mlp." + role, shape, dtype, device)
        source = example("source", (tokens, text.hidden))
        logits = example("router_logits", (tokens, text.experts), torch.float32)
        self.router_dtype = weights.router.dtype
        self.router_projection = UnquantizedProjection(example("router_source", source.shape, self.router_dtype),
            weights.router, logits, session=session, name=name + ".router").functional(session=session)
        routes = RouterScratch.allocate(tokens, device, experts=text.experts, top_k=text.experts_per_token)
        self.router = PreparedRouter(logits, routes, session=session, name=name + ".routing", score_func="sigmoid",
            renormalize=True, routed_scaling_factor=text.route_scale, correction_bias=weights.correction,
            num_valid_tokens=metadata.live_counts[:1]).functional(session=session)
        from ignition.components.situ_experts import PreparedSiTUExperts
        self.experts = PreparedSiTUExperts(weights.experts, tokens=tokens, top_k=text.experts_per_token)
        self.attention_convolution = PreparedCausalConvolution(weights.attention_convolution,
            pool.history(cache_layer, "attention_convolution"))
        self.mlp_convolution = PreparedCausalConvolution(weights.mlp_convolution, pool.history(cache_layer, "mlp_convolution"))
        blocks = (layer + text.residual_block - 1) // text.residual_block
        self.input_mix = (None if weights.attention_query is None else PreparedResidualMix(source,
            weights.attention_query, weights.input_norms, sources=blocks + 1, eps=text.eps,
            session=session, name=name + ".attention_input", examples=examples))
        self.input_norms = (() if self.input_mix is not None else tuple(PreparedTorchRMSNorm(source, norm,
            example(f"input_norm{index}", source.shape), eps=text.eps, session=session, name=f"{name}.input_norm{index}").functional(session=session)
            for index, norm in enumerate(weights.input_norms)))
        self.mlp_mix = PreparedResidualMix(source, weights.mlp_query, (weights.mlp_norm,),
            sources=blocks + (2 if layer % text.residual_block == 0 else 1), eps=text.eps,
            session=session, name=name + ".mlp_input", addend=layer % text.residual_block != 0, examples=examples)
        self.controllers = ()
        if messages:
            norms = weights.input_norms if len(weights.input_norms) == 3 else weights.input_norms * 3
            self.controllers = tuple(PreparedController(w, n, tokens=tokens, eps=text.eps,
                session=session, name=f"{name}.controller{i}", examples=examples)
                for i, (w, n) in enumerate(zip(weights.controllers, (*norms, weights.mlp_norm), strict=True)))
            bank_sources = (text.loop.core_end - text.loop.core_start) // text.residual_block
            self.attention_messages = PreparedResidualMix(source, weights.message_queries[:3], norms,
                sources=bank_sources, eps=text.eps, session=session, name=name + ".attention_messages", examples=examples)
            self.mlp_messages = PreparedResidualMix(source, weights.message_queries[3:], (weights.mlp_norm,),
                sources=bank_sources, eps=text.eps, session=session, name=name + ".mlp_messages", examples=examples)

    def forward(self, state, metadata, bank=None):
        w, t, m = self.weights, self.text, metadata
        if bank is None:
            streams = (tuple(norm(state.partial) for norm in self.input_norms) if self.input_mix is None
                       else self.input_mix((*state.blocks, state.partial)))
        else:
            dests = self.input_mix.mix((*state.blocks, state.partial))
            if len(dests) == 1:
                dests = dests * 3
            messages = self.attention_messages.mix(bank)
            streams = tuple(controller(dest, message) for controller, dest, message in
                            zip(self.controllers[:3], dests, messages, strict=True))
        _, attention_partial, attention_values = self.attention.attend(streams, m)
        attention_raw = self.reduction(attention_partial.to(self.reduction_dtype)).to(torch.bfloat16)
        attention, attention_previous = self.attention_convolution(attention_raw, m)
        sources = (*state.blocks, state.partial, attention) if self.layer % t.residual_block == 0 else (*state.blocks, state.partial)
        options = {} if self.layer % t.residual_block == 0 else dict(addend=attention)
        mlp_input = (self.mlp_mix(sources, **options)[0] if bank is None else
                     self.controllers[3](self.mlp_mix.mix(sources, **options)[0], self.mlp_messages.mix(bank)[0]))
        routes = self.router(self.router_projection(mlp_input.to(self.router_dtype)), num_valid_tokens=m.live_counts[:1])
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


class PreparedLoopDecoder(PreparedDecoder):
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
        self.visits = text.visits
        self.layers = [PreparedLoopLayer(index, weights.layers[index], text, pool, metadata, self.reduce,
            self.reduction_dtype, mode=mode, tokens=tokens, session=session, workspace=workspace,
            name=f"{name}.visit{visit}", examples=examples, cache_layer=visit, messages=pass_index > 0)
            for visit, (index, pass_index) in enumerate(self.visits)]
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
        self.output_mix = PreparedResidualMix(embedded, weights.output_query, (weights.final_norm,),
            sources=(len(weights.layers) + text.residual_block - 1) // text.residual_block + 1,
            eps=text.eps, session=session, name=name + ".output_mix", examples=examples)

    def __call__(self, metadata):
        w, t, m = self.weights, self.weights.config.text, metadata
        embedded = embedding_rows(w.embedding, m.token_ids, self.first_token)
        hidden = self.reduce(embedded.to(self.reduction_dtype)).to(torch.bfloat16)
        hidden = self.embedding_norm(hidden)
        if self.vision_state is not None and self.mode == "prefill":
            self.vision_state.inject(m, hidden)
        hidden = torch.where((m.positions >= 0)[:, None], hidden, 0)
        state, records, taps = ResidualState((), hidden), [], []
        prelude = bank = None
        for visit, (layer, (index, pass_index)) in enumerate(zip(self.layers, self.visits, strict=True)):
            if t.loop is not None and index == t.loop.core_start:
                if pass_index == 0:
                    prelude = state
                else:
                    bank = (*state.blocks[len(prelude.blocks) + 1:], state.partial)
                    state = prelude
            if w.draft is not None and visit in w.config.draft.taps:
                taps.append(state.partial)
            state, record = layer(state, m, bank if pass_index > 0 else None)
            if self.mode == "verify":
                records.append(record)
            del record
        normalized = self.output_mix((*state.blocks, state.partial))[0]
        if w.draft is not None:
            taps.append(normalized)
        sampled_hidden = normalized[m.sample_indices]
        local = self.logits(sampled_hidden)
        logits = local if self.vocabulary_parallel else self.gather(local.view(torch.bfloat16)).view(torch.float32)[:, :t.valid_vocabulary].contiguous()
        auxiliary = torch.stack(taps, dim=1) if taps else None
        return sampled_hidden, logits, auxiliary, tuple(records)
