# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Both GLM draft steps obey independently bounded stored-operand arithmetic.

The first step joins a fixed target hidden state and token embedding. The
recursive step uses the draft's normalized output and retains its selection.
The reference checks checkpoint projections, FP8 cache writes and attention,
BF16 expert stages, collectives, normalization, every vocabulary logit, and
the actual bounded proposal law. Wrong state, position, and selection are
rejected separately. These checks do not establish target-distribution
equivalence between ordinary decoding and speculative verification.
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from . import glm53_full_support as support
from .glm53_full_draft_capture import capture_rank
from .glm53_full_draft_reference import verify_arithmetic
from .glm53_full_draft_storage import verify_storage_and_probabilities


@support.requires_reference
@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
    reason="requires FLARE_RUN_TP_CUDA_TESTS=1 and eight assigned SM120 GPUs",
)
def test_draft_arithmetic_and_actual_proposal_law(tmp_path):
    if torch.cuda.device_count() != 8:
        pytest.skip("requires eight visible GPUs")
    if any(torch.cuda.get_device_capability(rank) != (12, 0) for rank in range(8)):
        pytest.skip("the stored-operand arithmetic contract is qualified on SM120")
    capture = tmp_path / "draft_operands"
    capture.mkdir()
    mp.spawn(
        capture_rank,
        args=(
            (tmp_path / "rendezvous").as_uri(),
            support.SNAPSHOT,
            support.REFERENCE,
            str(capture),
        ),
        nprocs=8,
    )
    for consumed in (2305, 2048, 2049, 2561):
        verify_arithmetic(capture, support.SNAPSHOT, support.REFERENCE, consumed)
        verify_storage_and_probabilities(capture, support.SNAPSHOT, consumed)
