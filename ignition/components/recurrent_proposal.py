# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared recurrent corrections for sequential draws from parallel draft logits."""

import torch

from .embedding import PreparedEmbedding
from .linear import UnquantizedProjection
from .markov_sampling import PreparedRowScaledFP8MarkovProjection, RowScaledFP8MarkovWeight


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
    """Keep recurrence state inside one block while retaining its actual sampled q.

    The recurrence input has a part that depends on the previous draw (state and predecessor
    embedding) and a part that does not (draft hidden state and target context). `project_context`
    computes the second part for every position of the block in one projection; `step` adds the
    first. `weights.head` maps the recurrent output to vocabulary corrections as BF16 rows, or as
    row-scaled E4M3 rows whose FP32 product takes the BF16 bias before rounding to BF16.
    """

    def __init__(self, weights, tokens, *, positions, session, name):
        rows, device = tokens.numel(), tokens.device
        rank = weights.embedding.shape[1]
        hidden = weights.context_weight.shape[1] // 2
        if type(positions) is not int or positions < 1:
            raise ValueError("a recurrent proposal head corrects at least one position per row")
        self.rank, self.softcap, self.positions = rank, weights.softcap, positions
        self.context_bias = weights.bias
        embedding = torch.empty((rows, rank), dtype=torch.bfloat16, device=device)
        dynamic = torch.empty((rows, 2 * rank), dtype=torch.bfloat16, device=device)
        context = torch.empty((rows * positions, 2 * hidden), dtype=torch.bfloat16, device=device)
        fixed = torch.empty((rows * positions, 3 * rank), dtype=torch.float32, device=device)
        raw = torch.empty((rows, 3 * rank), dtype=torch.float32, device=device)
        self.embed = PreparedEmbedding(weights.embedding, tokens, embedding,
            session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.context = UnquantizedProjection(context, weights.context_weight, fixed,
            session=session, name=name + ".context").functional(session=session)
        self.dynamic = UnquantizedProjection(dynamic, weights.state_weight, raw,
            session=session, name=name + ".dynamic").functional(session=session)
        self.output_bias = None
        if isinstance(weights.head, RowScaledFP8MarkovWeight):
            correction = torch.empty((rows, weights.head.out_features), dtype=torch.float32, device=device)
            self.output = PreparedRowScaledFP8MarkovProjection(embedding, weights.head, correction,
                session=session, name=name + ".output").functional(session=session)
            self.output_bias = weights.head_bias
        else:
            correction = torch.empty((rows, weights.head.shape[0]), dtype=torch.bfloat16, device=device)
            self.output = UnquantizedProjection(embedding, weights.head, correction,
                bias=weights.head_bias, session=session, name=name + ".output").functional(session=session)

    def begin(self, hidden):
        return torch.zeros((hidden.shape[0], self.rank), dtype=torch.bfloat16, device=hidden.device)

    def project_context(self, hidden, context):
        """Return FP32 [rows, positions, 3 * rank] recurrence inputs from [rows, positions, hidden] draft states."""
        rows, positions, width = hidden.shape
        if positions != self.positions:
            raise ValueError("recurrent context projection covers every prepared block position")
        source = torch.cat((hidden, context[:, None].expand(rows, positions, width)), -1)
        fixed = self.context(source.view(rows * positions, 2 * width)) + self.context_bias
        return fixed.view(rows, positions, 3 * self.rank)

    def step(self, state, previous, fixed, base):
        dynamic = self.dynamic(torch.cat((state, self.embed(token_ids=previous)), -1))
        state, latent = recurrent_gate(dynamic + fixed, state)
        correction = self.output(latent)
        if self.output_bias is not None:
            correction = (correction + self.output_bias.float()).bfloat16()
        return state, corrected_logits(base, correction, self.softcap)
