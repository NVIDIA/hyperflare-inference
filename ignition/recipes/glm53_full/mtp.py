# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The GLM 5.3 draft layer: proposals for speculative rounds and its cache of accepted context.

The draft layer is decoder layer 78 behind a projection that joins two
normalized inputs: the embedding of a token and the final normalized hidden
state of the position before it. A draft row for the token at position `p`
occupies position `p - 1` of the draft layer's latent and index pages, and its
output predicts the token at `p + 1`. The token embedding of the row at draft
position zero is zero.

Draft records are written from two sources:

- Accepted context (`PreparedContext`): every token the target consumes, joined
  with the target's hidden state of the preceding position. After the target
  has consumed `n` tokens of a request, draft positions below `n - 1` hold
  such records.
- Proposals (`PreparedProposals`): the first step of a round joins the pending
  token with the target's last hidden state and writes draft position `n - 1`,
  which accepted context later rewrites with the same inputs. A later step
  joins a proposed token with the preceding step's draft output; accepted
  context of the next forward overwrites its record before any row reads it as
  context of an accepted position.

When the checkpoint requests index sharing, the first proposal selects the
records for each request. Recursive proposals read those records while writing
their own latent records. Accepted context rebuilds the committed index keys.
"""

import torch
from flashinfer.triton.norm import rms_norm

from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.paged_batch import Metadata
from ignition.components.rotary import PreparedRotaryAngles
from ignition.components.state import gather_rows
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import PAGE_TOKENS, AttentionMetadata, ExampleBuffers, Selection, rotary_frequencies
from .decoder import HIDDEN, VOCABULARY, PreparedLayer


def draft_placement(positions, requests, page_table):
    """Records of draft rows: each live row's slot follows from its request's pages and its draft position."""
    live = positions >= 0
    position = positions.clamp(min=0)
    pages = page_table[requests, position // PAGE_TOKENS].long()
    slots = torch.where(live, pages * PAGE_TOKENS + position % PAGE_TOKENS, -1)
    return AttentionMetadata(positions, slots, requests, page_table)


class PreparedDraftLayer:
    """Join token embeddings with preceding hidden states and run the draft layer over packed rows.

    `cache_only` prepares the operators that store latent records and index
    keys, without attention, the MLP, or the all-reduces.
    """

    def __init__(self, weights, pool, positions, requests, page_table, *, cache_only, runtime, coordinator, session,
                 name, workspace, buffers: ExampleBuffers):
        config, draft = weights.config, weights.draft
        rows, device = positions.shape[0], positions.device
        self.draft, self.eps, self.cache_only = draft, config.norm_eps, cache_only
        example = AttentionMetadata(positions, buffers("slots", rows, dtype=torch.int64), requests, page_table)
        self.angles = PreparedRotaryAngles(positions, rotary_frequencies(config.rope_theta, device))
        self.projection = UnquantizedProjection(buffers("draft.joined", rows, 2 * HIDDEN), draft.projection,
            buffers("layer.source", rows, HIDDEN), session=session, name=f"{name}.projection")
        self.layer = PreparedLayer(config, config.draft_layer, draft.layer, pool, example, self.angles, rows=rows,
            heads=config.raw["num_attention_heads"] // weights.shard.world_size, mode="decode", runtime=runtime,
            coordinator=coordinator, session=session, name=f"{name}.layer", workspace=workspace, buffers=buffers,
            cache_only=cache_only)
        self.projection.functional(session=session)
        self.angles.functional()
        self.layer.functional(session=session)

    def __call__(self, embedded, previous, positions, requests, page_table, selection=None):
        """Store draft records and, for proposal rows, return hidden states and the attended records."""
        placement = draft_placement(positions, requests, page_table)
        embedded = embedded.masked_fill((positions == 0)[:, None], 0)
        joined = torch.empty((embedded.shape[0], 2 * HIDDEN), dtype=torch.bfloat16, device=embedded.device)
        rms_norm(embedded, self.draft.embedding_norm, joined[:, :HIDDEN], self.eps, round_before_weight=True)
        rms_norm(previous, self.draft.hidden_norm, joined[:, HIDDEN:], self.eps, round_before_weight=True)
        residual = self.projection(joined)
        angles = self.angles(positions)
        if self.cache_only:
            self.layer.store(residual, placement, angles)
            return None
        return self.layer(residual, placement, angles, selection, reuse_selection=selection is not None)


class PreparedContext:
    """Store draft records for the tokens a target forward consumed.

    The target decoder supplies, for every packed row, the token embedding and
    the normalized final hidden state, and for every request the hidden state
    its continuation row held before the forward. The row of the token at
    position `p` joins that token with the hidden state of `p - 1` and stores
    draft position `p - 1`; the token at position zero has no record.
    """

    def __init__(self, weights, auxiliary, metadata: Metadata, pool, *, runtime, coordinator, session, name,
                 workspace, buffers: ExampleBuffers | None = None, **_):
        device = metadata.token_ids.device
        buffers = ExampleBuffers(device) if buffers is None else buffers
        self.layer = PreparedDraftLayer(weights, pool, metadata.positions, metadata.requests, metadata.page_table,
            cache_only=True, runtime=runtime, coordinator=coordinator, session=session, name=name,
            workspace=workspace, buffers=buffers)

    def functional(self):
        return self

    def __call__(self, auxiliary, counts, metadata):
        embedded, normalized, saved = auxiliary
        rows = embedded.shape[0]
        index = torch.arange(rows, device=embedded.device)
        requests = metadata.requests.long()
        offset = index - metadata.query_start_loc[requests]
        previous = torch.where((offset == 0)[:, None], saved[requests], normalized[(index - 1).clamp(min=0)])
        accepted = (metadata.positions >= 1) & (offset < counts[requests])
        positions = torch.where(accepted, metadata.positions - 1, -1)
        self.layer(embedded, previous, positions, metadata.requests, metadata.page_table)


class PreparedProposals:
    """One draft step for every request of a speculative round.

    Step `i` runs the draft row at position `before - 1 + i`, where `before`
    is the request's consumed token count. Each step returns vocabulary logits
    and the normalized output that the following step joins with its token.
    A shared index selection lives only within this round.
    """

    def __init__(self, weights, pool, inputs, batch, depth, *, runtime, vocabulary_runtime, coordinator, session,
                 name, workspace, buffers: ExampleBuffers | None = None, **_):
        device = pool.backing.device
        buffers = ExampleBuffers(device) if buffers is None else buffers
        self.pool, self.inputs, self.batch, self.depth = pool, inputs, batch, depth
        self.weights, self.eps = weights, weights.config.norm_eps
        self.share_indices = weights.config.raw.get("index_share_for_mtp_iteration", False)
        self.logit_shape, self.logit_dtype = (batch, VOCABULARY), torch.bfloat16
        self.layer = self.embedding = self.head = None
        if depth == 0:
            return
        blocks = inputs.page_tables.shape[1]
        metadata = Metadata.allocate(batch, batch, blocks, device)
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, buffers("layer.sum", batch, HIDDEN),
            num_rows=metadata.live_counts[:1], session=session, name=f"{name}.embedding")
        self.layer = PreparedDraftLayer(weights, pool, metadata.positions, metadata.requests, metadata.page_table,
            cache_only=False, runtime=runtime, coordinator=coordinator, session=session, name=name,
            workspace=workspace, buffers=buffers)
        self.logits = buffers("head.logits", batch, VOCABULARY)
        self.head = PreparedVocabularyProjection(buffers("head.source", batch, HIDDEN), weights.head,
            buffers("head.local", batch, weights.head.shape[0]), self.logits, runtime=vocabulary_runtime,
            coordinator=coordinator, session=session, name=f"{name}.lm_head", workspace=workspace)
        self.embedding.functional(session=session, dynamic_inputs=True)
        self.head.functional(session=session)

    def release_outputs(self):
        self.logits = None

    def begin(self, states):
        """Start each round with the saved target hidden state and no preceding draft selection."""
        previous = torch.empty((self.batch, HIDDEN), dtype=torch.bfloat16, device=states.device)
        return gather_rows(self.pool.last_hidden, states, previous), None

    def step(self, state, index, tokens, before, states, active, proposing):
        previous, selection = state
        slots = self.inputs.row_slot_ids[:self.batch].long().clamp(0, self.inputs.page_tables.shape[0] - 1)
        requests = torch.arange(self.batch, device=tokens.device)
        positions = torch.where(proposing, before - 1 + index, -1)
        count = torch.full((1,), self.batch, dtype=torch.int32, device=tokens.device)
        embedded = self.embedding(token_ids=torch.where(proposing, tokens, 0), num_rows=count)
        if selection is not None:
            selection = Selection(torch.where(proposing[:, None], selection.records, -1),
                                  torch.where(proposing, selection.lengths, 0))
        hidden, selection = self.layer(embedded, previous, positions, requests, self.inputs.page_tables[slots], selection)
        normalized = torch.empty_like(hidden)
        rms_norm(hidden, self.weights.draft.head_norm, normalized, self.eps, round_before_weight=True)
        return self.head(normalized), (normalized, selection if self.share_indices else None)

    def finish(self, previous, before, states, active):
        """Proposals keep no state between rounds: accepted context rewrites the records they stored."""
