# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The GLM MoE DSA draft layer: learned proposals and absorption of target feedback into its cache.

The checkpoint's multi-token prediction layer is one decoder layer after the
stack with its own latent and index-key cache. At sequence position q it joins
the embedding of token q + 1 with the final hidden state of position q, each
through its own normalization, projects the pair to the hidden width, runs the
layer, and normalizes the result. That result predicts token q + 2 through the
vocabulary head, and is the hidden state the next proposal joins.

A draft record of position q depends on token q + 1, so the recipe stores it
in cache slot q + 1: the slot of the token that completes it. The draft cache
then advances with the target's, and a page never changes after its last slot
is consumed. Slot zero holds no sequence position and is never attended.

`PreparedFeedback` writes the records the target determines. For consecutive
consumed positions it joins each row's token with the preceding row's target
hidden state, or with the request's saved hidden state for the first row.
`PreparedProposals` drafts: the first step joins the pending token with the
saved hidden state, writes that slot, and selects the positions it attends.
Later steps join the previous proposal with the draft layer's own output and
attend the first step's selection, as the checkpoint's
`index_share_for_mtp_iteration` declares. Feedback from the verified round
then overwrites the slots the proposals wrote.
"""

from typing import NamedTuple

import torch
from flashinfer.triton.norm import rms_norm

from ignition.components.collectives import PreparedChunkedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import prepare_projection
from ignition.components.paged_batch import Metadata
from ignition.components.state import gather_rows
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.examples import declaration_tensor

from .attention import PreparedAttention, Selection
from .decoder import normalize
from .moe import PreparedMoE
from .records import rotary_frequencies


class PreparedDraftLayer:
    """The draft layer on rows that each hold one cache slot.

    Metadata positions name cache slots. `cache_only` prepares the join and the
    cache writes without attention, the MLP, or the head. Without `selects`
    the layer attends a selection passed to it.
    """

    def __init__(self, weights, rows, metadata, pool, *, mode, runtime, vocabulary_runtime, coordinator, session,
                 name, workspace, examples, context=None, cache_only=False, selects=True):
        config, language, draft = weights.config, weights.tree.model, weights.mtp
        if draft is None:
            raise ValueError("draft preparation requires the checkpoint's multi-token prediction layer")
        module, device, hidden = language.layers[config.layers], pool.backing.device, config.hidden
        self.hidden, self.eps = hidden, config.norm_eps

        def tensor(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "glm_dsa.draft." + role, shape, dtype, device)

        source = tensor("source", (rows, hidden))
        self.embedding = PreparedEmbedding(language.embed_tokens.weight, metadata.token_ids,
            tensor("embedded", (rows, hidden)), num_rows=metadata.live_counts[:1], session=session,
            name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.mixers = draft.tensors["enorm"], draft.tensors["hnorm"]
        self.project = prepare_projection(tensor("joined", (rows, 2 * hidden)), draft.tensors["eh_proj"], source,
                                          session=session, name=name + ".join", workspace=workspace)
        self.project.functional(session=session)
        layer = draft.layer
        self.attention = PreparedAttention(module.self_attn, layer.attention, source, metadata, pool.draft_attention(),
            index_cache=pool.draft_indexer_pages(), frequencies=rotary_frequencies(config.rotary_base, device),
            mode=mode, session=session, name=name + ".attention", workspace=workspace, examples=examples,
            context=context, layer="draft", selects=selects, cache_only=cache_only, rotary_offset=-1,
            first_record=1)
        self.norms = (layer.norms["input_layernorm.weight"], layer.norms["post_attention_layernorm.weight"],
                      draft.tensors["shared_head.norm"])
        self.norm_eps = (module.input_layernorm.variance_epsilon, module.post_attention_layernorm.variance_epsilon)
        self.mlp = self.reduce = self.head = None
        if cache_only:
            return
        self.mlp = PreparedMoE(module.mlp, source, layer.mlp, session=session, name=name + ".moe",
                               workspace=workspace, examples=examples)
        self.reduce = PreparedChunkedAllReduce(source, tensor("reduced", (rows, hidden)), runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce").functional(session=session)
        first, last = config.vocabulary_partition(weights.shard)
        self.head = PreparedVocabularyProjection(tensor("hidden", (rows, hidden)),
            weights.head if draft.draft_head is None else draft.draft_head,
            tensor("local_logits", (rows, last - first)), tensor("logits", (rows, config.vocabulary)),
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            workspace=workspace).functional(session=session)

    def _join(self, previous, metadata):
        """The layer input: the projected pair of each row's token embedding and previous hidden state."""
        embedded = self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
        joined = torch.empty((embedded.shape[0], 2 * self.hidden), dtype=embedded.dtype, device=embedded.device)
        rms_norm(embedded, self.mixers[0], joined[:, :self.hidden], self.eps, round_before_weight=True)
        rms_norm(previous, self.mixers[1], joined[:, self.hidden:], self.eps, round_before_weight=True)
        return self.project(joined)

    def cache(self, previous, metadata):
        """Write each row's draft record and index key without attending."""
        residual = self._join(previous, metadata)
        self.attention.write(normalize(residual, self.norms[0], self.norm_eps[0]), metadata)

    def __call__(self, previous, metadata, selection=None):
        """Return the normalized layer output, its logits, and the selection."""
        residual = self._join(previous, metadata)
        attended, selection = self.attention(normalize(residual, self.norms[0], self.norm_eps[0]), metadata,
                                             selection)
        residual = residual + self.reduce(attended)
        residual = residual + self.reduce(self.mlp(normalize(residual, self.norms[1], self.norm_eps[1])))
        hidden = normalize(residual, self.norms[2], self.eps)
        return hidden, self.head(hidden), selection


class PreparedFeedback:
    """Write the draft records that a block of consumed positions determines.

    `auxiliary` is the target decoder's pair: the final hidden state of every
    row, and each request's hidden state saved before the block. Rows keep the
    block's metadata: row i's record joins its token with the hidden state of
    the position before it, in the slot of its own position. A sequence's first
    token has no preceding state. Its slot is written from a zero state and
    never attended.
    """

    def __init__(self, weights, metadata, pool, *, mode, session, name, workspace, **collectives):
        self.layer = PreparedDraftLayer(weights, metadata.token_ids.shape[0], metadata, pool, mode=mode,
            session=session, name=name, workspace=workspace, cache_only=True, **collectives)

    def functional(self, *, session=None):
        return self

    def __call__(self, auxiliary, counts, metadata):
        hidden, saved = auxiliary
        m = metadata
        offsets = torch.arange(hidden.shape[0], device=hidden.device)
        first = offsets == m.query_start_loc[m.requests]
        previous = torch.where(first[:, None], saved[m.requests], hidden[(offsets - 1).clamp_min(0)])
        previous = torch.where((m.positions > 0)[:, None], previous, torch.zeros_like(previous))
        self.layer.cache(previous, m)


class DraftState(NamedTuple):
    """One round's proposal state: each request's previous hidden state and the first step's selection."""

    previous: torch.Tensor
    selection: Selection | None


class PreparedProposals:
    """Draft `depth` tokens for each of `batch` requests, one draft-layer pass per token.

    Row r belongs to request row r in every step, and a request that proposes
    nothing in a step keeps an inactive row. The first step selects the
    positions every later step attends.
    """

    def __init__(self, weights, pool, scratch, batch, depth, *, inputs, requests, session, name, workspace,
                 **collectives):
        self.batch, self.pool, self.inputs, self.capacity = batch, pool, inputs, requests.capacity
        config, device = weights.config, pool.backing.device
        self.logit_shape, self.logit_dtype = (batch, config.vocabulary), torch.bfloat16
        metadata = Metadata.allocate(batch, batch, inputs.page_tables.shape[1], device)
        common = dict(mode="decode", session=session, workspace=workspace, **collectives)
        self.first = self.rest = None
        if depth > 0:
            self.first = PreparedDraftLayer(weights, batch, metadata, pool, name=name + ".first", **common)
        if depth > 1:
            self.rest = PreparedDraftLayer(weights, batch, metadata, pool, name=name + ".rest", selects=False,
                                           **common)

    def release_outputs(self):
        pass

    def begin(self, states):
        saved = self.pool.last_hidden
        previous = torch.empty((self.batch, saved.shape[1]), dtype=saved.dtype, device=states.device)
        gather_rows(saved, states, previous)
        return DraftState(previous, None)

    def _metadata(self, index, tokens, before, states, proposing):
        """One slot per proposing request: the pending token's position for the first step, then one further."""
        device = tokens.device
        slots = self.inputs.row_slot_ids[:self.batch].long().clamp(0, self.capacity - 1)
        pages = torch.where(proposing[:, None], self.inputs.page_tables[slots], -1)
        rows = torch.arange(self.batch, device=device)
        lengths = proposing.to(torch.int32)
        return Metadata(
            torch.where(proposing, tokens, 0), rows, torch.arange(self.batch + 1, dtype=torch.int32, device=device),
            torch.where(proposing, states, 0), before + index, lengths, rows,
            torch.where(proposing, before + index, -1), pages, pages,
            torch.cat((torch.full((1,), self.batch, dtype=torch.int32, device=device),
                       self.inputs.live_rows.clamp(0, self.batch))), 1)

    def step(self, state, index, tokens, before, states, active, proposing):
        metadata = self._metadata(index, tokens, before, states, proposing)
        layer = self.first if index == 0 else self.rest
        hidden, logits, selection = layer(state.previous, metadata, None if index == 0 else state.selection)
        return logits, DraftState(hidden, selection)

    def finish(self, state, before, states, active):
        pass
