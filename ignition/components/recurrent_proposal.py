# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared recurrent corrections for sequential draws from parallel draft logits."""

import torch

from .embedding import PreparedEmbedding
from .linear import UnquantizedProjection


def recurrent_gate(raw, state):
    """Round each gate, product, and state update to checkpoint BF16 precision."""
    gate, candidate, output = raw.bfloat16().chunk(3, -1)
    gate = gate.float().sigmoid().bfloat16()
    candidate = candidate.float().tanh().bfloat16()
    keep = (gate.float() * state.float()).bfloat16()
    update = ((1. - gate.float()).bfloat16().float() * candidate.float()).bfloat16()
    state = (keep.float() + update.float()).bfloat16()
    return state, output.float().tanh().bfloat16()


def corrected_logits(base, correction, softcap):
    """Add BF16 logits before applying the FP32 softcap and BF16 output rounding."""
    summed = (base.float() + correction.float()).bfloat16().float()
    return (softcap * (summed / softcap).tanh()).bfloat16()


class PreparedRecurrentProposalHead:
    """Keep recurrence state inside one block while retaining its actual sampled q."""

    def __init__(self, weights, tokens, *, session, name):
        rows, device = tokens.numel(), tokens.device
        rank = weights.embedding.shape[1]
        hidden = weights.context_weight.shape[1] // 2
        self.rank, self.softcap = rank, weights.softcap
        self.context_bias = weights.bias
        embedding = torch.empty((rows, rank), dtype=torch.bfloat16, device=device)
        dynamic = torch.empty((rows, 2 * rank), dtype=torch.bfloat16, device=device)
        context = torch.empty((rows, 2 * hidden), dtype=torch.bfloat16, device=device)
        raw = torch.empty((rows, 3 * rank), dtype=torch.float32, device=device)
        correction = torch.empty((rows, weights.head.shape[0]), dtype=torch.bfloat16, device=device)
        self.embed = PreparedEmbedding(weights.embedding, tokens, embedding,
            session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.context = UnquantizedProjection(context, weights.context_weight, raw,
            session=session, name=name + ".context").functional(session=session)
        self.dynamic = UnquantizedProjection(dynamic, weights.state_weight, raw,
            session=session, name=name + ".dynamic").functional(session=session)
        self.output = UnquantizedProjection(embedding, weights.head, correction,
            bias=weights.head_bias, session=session, name=name + ".output").functional(session=session)

    def begin(self, hidden):
        return torch.zeros((hidden.shape[0], self.rank), dtype=torch.bfloat16, device=hidden.device)

    def step(self, state, previous, hidden, context, base):
        fixed = self.context(torch.cat((hidden, context), -1)) + self.context_bias
        dynamic = self.dynamic(torch.cat((state, self.embed(token_ids=previous)), -1))
        state, latent = recurrent_gate(dynamic + fixed, state)
        return state, corrected_logits(base, self.output(latent), self.softcap)
