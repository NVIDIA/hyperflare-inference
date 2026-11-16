# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary-parallel Inkling 2 embedding rows and softcapped logits of one rank.

Each rank holds `width` consecutive rows of the padded 201024-row embedding and
unembedding tables, starting at `rank * width`; rows past the table are zero.
The embedding contributes a token's row on the rank that holds it and zero
elsewhere, so the cross-rank sum is the row exactly. Logits project the final
normalized hidden state in FP32, apply `softcap * tanh(x / softcap)` and are
`-inf` at IDs from the valid vocabulary size on. An unembedding shard stored as
MXFP8 projects BF16 activations through b12x's packed-weight kernels, which round
each FP32 sum to BF16; the softcap and every later consumer read that value as FP32.
"""

from __future__ import annotations

import torch

from ignition.components.linear import UnquantizedProjection
from ignition.components.vocabulary import WidenedQuantizedProjection


def embedding_rows(table, ids, first):
    """This rank's embedding rows of `ids`, zero for IDs another rank holds and for padded rows (`ids < 0`).

    `first` is the token ID of the rank's first table row, `rank * width`; compiled
    forwards pass it as a device tensor so that every rank binds the same code.
    """
    width = table.shape[0]
    local = ids - first
    inside = (local >= 0) & (local < width)
    return torch.where(inside[:, None], table[local.clamp(0, width - 1)], 0)


class PreparedLogits:
    """FP32 logits of this rank's vocabulary columns with the softcap and the valid-vocabulary mask."""

    def __init__(self, weight, text, rank, *, tokens, session, name, examples=None, workspace=None):
        """`weight` is the rank's BF16 `[columns, hidden]` unembedding shard or its MXFP8 form."""
        from ignition.preparation.examples import declaration_tensor

        examples = {} if examples is None else examples
        packed = not isinstance(weight, torch.Tensor)
        device = weight.weight.values.device if packed else weight.device
        columns = weight.out_features if packed else weight.shape[0]
        source = declaration_tensor(examples, "inkling.logits.source", (tokens, text.hidden), torch.bfloat16, device)
        output = declaration_tensor(examples, "inkling.logits.local", (tokens, columns), torch.float32, device)
        if packed:
            rounded = declaration_tensor(examples, "inkling.logits.rounded", output.shape, torch.bfloat16, device)
            projection = WidenedQuantizedProjection(source, weight, output, session=session,
                name=f"{name}.projection", workspace=workspace, rounded=rounded)
        else:
            projection = UnquantizedProjection(source, weight, output, session=session, name=f"{name}.projection")
        self.projection = projection.functional(session=session)
        self.softcap = text.softcap
        # Rank-dependent values are device data, so every rank binds the same compiled code.
        self.valid_columns = rank * columns + torch.arange(columns, device=device) < text.valid_vocabulary

    def __call__(self, hidden):
        logits = self.projection(hidden)
        capped = self.softcap * torch.tanh(logits / self.softcap)
        return torch.where(self.valid_columns, capped, float("-inf"))
