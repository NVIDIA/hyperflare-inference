# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete GLM MoE DSA text forwards: pre-normalized attention and MLP residuals over every layer."""

from dataclasses import dataclass

import torch
from flashinfer.triton.norm import rms_norm

from ignition.components.collectives import (PreparedAllReduceAdd, PreparedAllReduceRMSNorm,
                                             PreparedChunkedAllReduce)
from ignition.components.embedding import PreparedEmbedding
from ignition.components.mlp import PreparedMLP
from ignition.components.state import gather_rows
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.examples import declaration_tensor

from .attention import PreparedAttention
from .dcp import PrefillSlabs
from .moe import PreparedMoE
from .records import rotary_frequencies
from .weights import MoEWeights


# b12x reduces two through eight rows across eight ranks with its owner-staged transport and longer
# inputs with a pull whose time grows with every row, so a longer decode forward reduces in slices.
REDUCE_ROWS = 8


class SlicedReduction:
    """One fused reduction applied to consecutive row slices of its operands.

    `prepare(length, suffix)` prepares the reduction of `length` rows, and equally long slices share
    it. With `normalizes`, the slices write their normalized rows into one output.
    """

    def __init__(self, rows, prepare, *, normalizes):
        count = -(-rows // REDUCE_ROWS)
        bounds, start = [], 0
        for index in range(count):
            end = start + rows // count + (index < rows % count)
            bounds.append((start, end))
            start = end
        self.bounds, self.normalizes, self.operations = tuple(bounds), normalizes, {}
        for start, end in self.bounds:
            length = end - start
            if length not in self.operations:
                self.operations[length] = prepare(length, "" if count == 1 else f".rows{length}")

    def __call__(self, source, residual):
        """Reduce `source` into `residual` in place and return the normalized rows, or the residual."""
        if len(self.bounds) == 1:
            return self.operations[source.shape[0]](source, residual)
        output = torch.empty_like(source) if self.normalizes else None
        for start, end in self.bounds:
            operation = self.operations[end - start]
            if self.normalizes:
                operation(source[start:end], residual[start:end], output=output[start:end])
            else:
                operation(source[start:end], residual[start:end])
        return output if self.normalizes else residual


@dataclass(frozen=True)
class Scratch:
    """Declaration examples for one graph variant's activations and sampled rows."""

    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, shard, device, *, examples=None):
        first, last = config.vocabulary_partition(shard)

        def tensor(role, shape):
            return declaration_tensor(examples, role, shape, torch.bfloat16, device)

        return cls(tensor("glm_dsa.hidden_input", (tokens, config.hidden)),
                   tensor("glm_dsa.hidden", (rows, config.hidden)),
                   tensor("glm_dsa.local_logits", (rows, last - first)),
                   tensor("glm_dsa.logits", (rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


def normalize(value, weight, eps):
    output = torch.empty_like(value)
    rms_norm(value, weight, output, eps, round_before_weight=True)
    return output


class PreparedDecoder:
    """Prepare every layer into a serial forward owned by one tensor-parallel rank.

    Each layer normalizes its input, attends, reduces the ranks' partial outputs,
    and adds the result to the residual, then does the same for its MLP. Decode
    fuses each reduction with the residual addition and the normalization that
    follows it, in slices of at most eight rows. A
    layer without an indexer attends the selection of the nearest preceding
    layer that owns one. Final normalization and the vocabulary projection run
    on sampled rows only. With `context`, context ranks share the latent records and
    index keys, and a prefill forward allocates the record slab, the position marks,
    and the key slab its layers reuse.

    A verification forward runs several consecutive positions of each request
    as decode rows: every row writes its cache records before any row attends,
    and a row attends positions through its own. Rows a round rejects leave
    records past the request's consumed positions, which nothing attends
    before a later forward rewrites them, so `commit` has nothing to publish.

    When the weights carry the draft layer, the forward also returns what
    draft feedback joins: the final normalized hidden state of every row, and
    each request's hidden state saved before the forward.
    This object executes in Python only during preparation and CUDA graph
    capture.
    """

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime, coordinator,
                 session, name, workspace, examples, context=None, a4_prefill=False, overlap=None):
        if mode not in ("prefill", "decode", "verify"):
            raise ValueError("GLM MoE DSA decoders prepare prefill, decode, and verification")
        if overlap is not None:
            raise ValueError("GLM MoE DSA decoders prepare no auxiliary-stream overlap")
        # Verification rows are decode rows: one position each, attending the cache through that position.
        mode = "decode" if mode == "verify" else mode
        config, language, source = weights.config, weights.tree.model, scratch.activations
        # A tree built for the draft layer carries its module after the decoder's.
        modules = language.layers[:config.layers]
        if len(weights.layers) != config.layers or len(modules) != config.layers:
            raise ValueError("GLM MoE DSA text execution requires every decoder layer")
        self.weights, self.scratch, self.pool = weights, scratch, pool
        self.slab_positions = context.slab_positions if context is not None and mode == "prefill" else 0
        # Draft feedback sizes itself from the metadata, so preparation passes it no example.
        self.auxiliary, self.feedback = None, weights.draft is not None
        device = source.device
        self.frequencies = rotary_frequencies(config.rotary_base, device)
        self.embedding = PreparedEmbedding(language.embed_tokens.weight, metadata.token_ids, source,
            num_rows=metadata.live_counts[:1], session=session,
            name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.fused = mode == "decode" and PreparedAllReduceRMSNorm.fits(runtime, source)

        def activation(role):
            return declaration_tensor(examples, "glm_dsa." + role, source.shape, source.dtype, device)

        self.layers = []
        for index, (module, layer) in enumerate(zip(modules, weights.layers, strict=True)):
            prefix = f"{name}.layer{index}"
            attention = PreparedAttention(module.self_attn, layer.attention, source, metadata, pool.attention(index),
                index_cache=pool.indexer_pages(index) if config.indexed(index) else None,
                frequencies=self.frequencies, mode=mode, session=session, name=prefix + ".attention",
                workspace=workspace, examples=examples, context=context, layer=index)
            if isinstance(layer.mlp, MoEWeights):
                mlp = PreparedMoE(module.mlp, source, layer.mlp, session=session, name=prefix + ".moe",
                                  workspace=workspace, examples=examples,
                                  prefill=mode == "prefill" and a4_prefill)
            else:
                local = layer.mlp.down.shape[1]

                def tensor(role, width):
                    return declaration_tensor(examples, "glm_dsa.mlp." + role, (source.shape[0], width),
                                              torch.bfloat16, device)

                mlp = PreparedMLP(source, layer.mlp, tensor("gate_up", 2 * local), tensor("intermediate", local),
                    tensor("output", config.hidden), session=session, name=prefix + ".mlp",
                    workspace=workspace).functional(session=session)
            norms = (layer.norms["input_layernorm.weight"], layer.norms["post_attention_layernorm.weight"])
            eps = (module.input_layernorm.variance_epsilon, module.post_attention_layernorm.variance_epsilon)
            if self.fused:
                # A prefill forward adds the ranks' sum after it rounds to BF16, so the fused sum rounds too.
                common = dict(runtime=runtime, coordinator=coordinator, session=session,
                              reduction_dtype=torch.bfloat16)
                partial, residual, normalized = activation("partial"), activation("residual"), activation("normalized")

                def normalizing(weight, epsilon, label, prefix=prefix):
                    return SlicedReduction(source.shape[0], lambda length, suffix: PreparedAllReduceRMSNorm(
                        partial[:length], residual[:length], weight, normalized[:length], eps=epsilon,
                        name=f"{prefix}.{label}{suffix}", **common).functional(session=session), normalizes=True)

                post = normalizing(norms[1], eps[1], "attention_reduce_norm")
                if index + 1 < config.layers:
                    # The MLP's reduction also normalizes the next layer's input.
                    following = language.layers[index + 1].input_layernorm
                    reduce = normalizing(weights.layers[index + 1].norms["input_layernorm.weight"],
                                         following.variance_epsilon, "mlp_reduce_norm")
                else:
                    reduce = SlicedReduction(source.shape[0], lambda length, suffix, prefix=prefix: PreparedAllReduceAdd(
                        partial[:length], residual[:length], name=f"{prefix}.mlp_reduce{suffix}",
                        **common).functional(session=session), normalizes=False)
                self.layers.append((attention, mlp, (post, reduce), norms, eps))
                continue
            reduce = PreparedChunkedAllReduce(source, activation("hidden_output"), runtime=runtime,
                coordinator=coordinator, session=session, name=prefix + ".reduce").functional(session=session)
            self.layers.append((attention, mlp, reduce, norms, eps))
        self.norm, self.norm_eps = language.norm.weight, language.norm.variance_epsilon
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session,
            name=name + ".head", workspace=workspace).functional(session=session)

    def release_outputs(self):
        self.scratch = None

    def commit(self, counts, records, metadata):
        """Accepted rows already hold their cache records."""

    def __call__(self, metadata):
        m = metadata
        saved = None
        if self.feedback:
            # Generation overwrites the saved rows before feedback reads them. The gather names the
            # tensor generation writes, so a compiled forward keeps it ahead of that write.
            states = self.pool.last_hidden
            saved = torch.empty((m.state_rows.shape[0], states.shape[1]), dtype=states.dtype, device=states.device)
            gather_rows(states, m.state_rows, saved)
        residual = self.embedding(token_ids=m.token_ids, num_rows=m.live_counts[:1])
        selection = None
        slabs = PrefillSlabs.allocate(self.slab_positions, residual.device) if self.slab_positions else None
        if self.fused:
            norms, eps = self.layers[0][3:]
            hidden = normalize(residual, norms[0], eps[0])
            for attention, mlp, (post, reduce), _, _ in self.layers:
                attended, selection = attention(hidden, m, None if attention.selects else selection)
                # Each fused reduction adds the ranks' sum to the residual in place.
                hidden = reduce(mlp(post(attended, residual)), residual)
        else:
            for attention, mlp, reduce, norms, eps in self.layers:
                attended, selection = attention(normalize(residual, norms[0], eps[0]), m,
                                                None if attention.selects else selection, slabs=slabs)
                residual = residual + reduce(attended)
                residual = residual + reduce(mlp(normalize(residual, norms[1], eps[1])))
        if not self.feedback:
            hidden = normalize(residual[m.sample_indices], self.norm, self.norm_eps)
            return hidden, self.head(hidden), None, None
        normalized = normalize(residual, self.norm, self.norm_eps)
        hidden = normalized[m.sample_indices]
        return hidden, self.head(hidden), (normalized, saved), None
