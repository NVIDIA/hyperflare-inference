# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM MoE DSA speculative rounds: draft-layer proposals that the target decoder verifies."""

from functools import partial

from ignition.components.causal_draft import PreparedCausalDraftRound

from .decoder import PreparedDecoder
from .mtp import PreparedFeedback, PreparedProposals


class PreparedRound(PreparedCausalDraftRound):
    """One round for `batch` requests that each verify up to `width` consecutive positions.

    The draft layer proposes up to `width - 1` tokens after each request's
    pending token. The target decoder verifies the pending token and the
    proposals as decode rows, and draft feedback rewrites the draft records of
    every verified position from the target's hidden states.
    """

    def __init__(self, owner, batch, width, name):
        def feedback(weights, metadata, pool, **common):
            return PreparedFeedback(weights, metadata, pool, mode="decode", name=name + ".feedback", **common)

        super().__init__(owner, batch, width, name, decoder_factory=PreparedDecoder, context_factory=feedback,
            proposal_factory=partial(PreparedProposals, inputs=owner.inputs, requests=owner.requests))
