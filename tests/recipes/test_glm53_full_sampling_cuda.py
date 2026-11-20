# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check the proposal and verification operators used by GLM 5.3 on known token distributions."""

import math
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


def test_one_draft_acceptance_and_corrected_token_frequencies():
    from ignition.components.draft_sampling import PreparedDraftDistribution
    from ignition.components.sampling import DeviceRequestState, RUNNING
    from ignition.components.speculation import (
        DRAFT_RNG_DOMAIN, PreparedSpeculativeDistributions, PreparedStochasticVerification, SpeculativeRngState,
    )
    from ignition.preparation.scratch import bind_scratch

    batch, vocab, replays = 8, 8, 8192
    device = torch.device("cuda")
    # Each accepted token has mass min(p, q). Rejection assigns the remaining target mass to corrected tokens.
    proposal_logits = torch.tensor([2, 1, 0, -1, -0.25, 0.75, -0.5, -1.5], dtype=torch.bfloat16, device=device)
    target_logits = proposal_logits.flip((0,))
    q, p = (logits.cpu().double().softmax(0) for logits in (proposal_logits, target_logits))
    draft_logits = proposal_logits.expand(batch, 1, vocab).contiguous()
    verify_logits = target_logits.expand(batch, 2, vocab).contiguous()
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch)
    state.status.fill_(RUNNING)
    state.generation.fill_(7)
    state.seed.copy_(torch.arange(batch, device=device) + 93851)
    state.parameters.copy_(torch.tensor([1., 1., 0., 1., 0., 0.], device=device))
    slots = torch.arange(batch, dtype=torch.int32, device=device)
    generations = torch.full((batch,), 7, dtype=torch.int64, device=device)
    live = torch.tensor([batch], dtype=torch.int32, device=device)
    rows = (state, slots, generations, live)
    ids = torch.zeros((batch, 1), dtype=torch.int64, device=device)
    seed = state.seed.view(torch.int64) ^ DRAFT_RNG_DOMAIN
    offset = torch.zeros_like(seed)
    sampled = torch.zeros_like(seed)
    rng = SpeculativeRngState.allocate(device, capacity=batch)
    rng.generation.fill_(7)
    draft = PreparedDraftDistribution(draft_logits, *rows, ids, seed=seed, offset=offset, sampled=sampled)
    target = PreparedSpeculativeDistributions(verify_logits, *rows, ids)
    verify = PreparedStochasticVerification(draft.probabilities, ids, target.probabilities, *rows, rng=rng,
                                            target_valid=target.valid, draft_valid=draft.valid)
    for op in (draft, target, verify):
        op.prepare()
    q_work, p_work, v_work = (bind_scratch(op.layout, device=device) for op in (draft, target, verify))
    for op in (draft, target, verify):
        op.functional()
    counts = torch.zeros((4, vocab), dtype=torch.int64, device=device)
    ones = torch.ones_like(sampled)

    def draw():
        offset.add_(4)
        rng.target_offset.add_(verify.draw_reservation)
        draft.run(q_work, draft_logits, candidates=ids, seed=seed, offset=offset, sampled=sampled)
        ids[:, 0].copy_(sampled)
        target.run(p_work, verify_logits, candidates=ids)
        out = verify.run(v_work, q_work["probabilities"], p_work["probabilities"], p_work["valid"],
                         drafts=ids, draft_valid=q_work["valid"], disjoint=True)
        accepted = out.accepted_drafts.long()
        first = out.token_ids[:, 0]
        counts[0].scatter_add_(0, sampled, ones)
        counts[1].scatter_add_(0, first, accepted)
        counts[2].scatter_add_(0, first, 1 - accepted)
        # A verifier which emits the original proposal after rejection assigns this mass to corrected tokens.
        counts[3].scatter_add_(0, sampled, 1 - accepted)

    draw()
    torch.testing.assert_close(q_work["probabilities"][0, 0].cpu().double(), q, rtol=3e-6, atol=1e-8)
    torch.testing.assert_close(p_work["probabilities"][0, 0].cpu().double(), p, rtol=3e-6, atol=1e-8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        draw()
    counts.zero_()
    for _ in range(replays):
        graph.replay()
    n = batch * replays
    proposed, accepted, corrected, wrong = counts.cpu().double() / n
    # Simultaneous Hoeffding bound for the 24 proposal, acceptance, and correction frequencies.
    bound = math.sqrt(math.log(2 * 3 * vocab / 1e-7) / (2 * n))
    expected_corrections = (p - q).clamp(min=0)
    assert float((proposed - q).abs().max()) < bound
    assert float((accepted - torch.minimum(p, q)).abs().max()) < bound
    assert float((corrected - expected_corrections).abs().max()) < bound
    assert float((wrong - expected_corrections).abs().max()) > bound
    assert proposed.sum() == 1 and accepted.sum() + corrected.sum() == 1
