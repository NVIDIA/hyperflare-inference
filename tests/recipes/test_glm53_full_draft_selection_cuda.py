# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recursive GLM 5.3 draft steps reuse the first step's physical records within one request."""

from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import glm53_full_support as support


@torch.no_grad()
def _rank(rank, rendezvous, snapshot, fixtures):
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import CollectivePreparation, create_all_gather_runtime, create_all_reduce_runtime
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53_full.attention import ExampleBuffers
    from ignition.recipes.glm53_full.mtp import PreparedDraftLayer, PreparedProposals
    from ignition.recipes.glm53_full.pool import ContinuationPool
    from ignition.recipes.glm53_full.weights import ModelWeights

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    timeout = timedelta(minutes=20)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=8, timeout=timeout)
    try:
        group = dist.group.WORLD
        coordinator = CollectivePreparation(group, timeout=timeout)
        weights = ModelWeights.load(inspect_indexed_checkpoint(Path(snapshot)), device,
                                    shard=TensorParallelShard(rank, 8), layers=1, draft=True)
        consumed, chunk, pages = 2305, 128, 10
        fixture = torch.load(Path(fixtures) / "long.pt", mmap=True)
        tokens = fixture["token_ids"].to(device)
        teacher = fixture["hidden"][:consumed].to(device=device, dtype=torch.bfloat16)
        table = torch.arange(2 * pages, dtype=torch.int32, device=device).reshape(2, pages)
        lanes = torch.arange(2, device=device)
        inputs = SimpleNamespace(page_tables=table, row_slot_ids=lanes)
        pool = ContinuationPool.for_model(weights.config, 2, 2 * pages, device, draft=True)
        arena = ScratchArena(device, overlap=False)
        reduction = create_all_reduce_runtime(group, device, max_tokens=chunk, hidden_size=weights.config.hidden)
        gather = create_all_gather_runtime(group, device, max_rows=2, columns=weights.head.shape[0] * 8)
        with RecordedPreparationSession(device=device, checkpoint="glm53-full-test", workload_schema="glm53-full-draft-selection",
                                        numerical_policy="default", coordinator=coordinator) as session:
            session.register_scratch(arena)
            common = dict(runtime=reduction, coordinator=coordinator, session=session, workspace=arena)
            offsets = torch.arange(chunk, device=device)
            requests = torch.zeros(chunk, device=device, dtype=torch.int32)
            context = PreparedDraftLayer(weights, pool, offsets, requests, table[:1], cache_only=True,
                                         buffers=ExampleBuffers(device), name="context", **common)
            proposal = PreparedProposals(weights, pool, inputs, 2, 2, vocabulary_runtime=gather, name="proposal", **common)
            session.tune()
            arena.release_preparation()
            pool.clear_for_preparation()
            for start in range(0, consumed - 1, chunk):
                context(weights.embedding[tokens[start + 1:start + chunk + 1]], teacher[start:start + chunk],
                        offsets + start, requests, table[:1])
            banks = [pool.pages[name] for name in ("draft.latent", "draft.indexer")]
            for bank in banks:
                bank[pages:].copy_(bank[:pages])
            pool.last_hidden.copy_(teacher[-1].expand_as(pool.last_hidden))
            before = torch.full((2,), consumed, device=device)
            both = torch.ones(2, dtype=torch.bool, device=device)
            one = torch.tensor([True, False], device=device)
            attention = proposal.layer.layer.attention
            attend = attention.attend
            observed = []

            def record(absorbed, selected, metadata):
                observed.append((selected.records.clone(), selected.lengths.clone()))
                return attend(absorbed, selected, metadata)

            with patch.object(attention, "attend", record):
                logits, state = proposal.step(proposal.begin(lanes), 0, tokens[consumed].expand(2), before, lanes, both, both)
                next_tokens = logits.argmax(1)
                saved = [bank.clone() for bank in banks]
                proposal.step(state, 1, next_tokens, before, lanes, both, one)
                assert all(torch.equal(bank[pages:], saved_bank[pages:]) for bank, saved_bank in zip(banks, saved))
                for bank, saved_bank in zip(banks, saved):
                    bank.copy_(saved_bank)
                # Dropping the previous selection forces a fresh ranking at the recursive token.
                proposal.step((state[0], None), 1, next_tokens, before, lanes, both, one)
            (first, lengths), (shared, shared_lengths), (fresh, _) = observed
            assert lengths.tolist() == [2048, 2048]
            assert torch.equal(first[0], shared[0])
            assert shared_lengths.tolist() == [2048, 0] and (shared[1] == -1).all()
            assert not torch.equal(first[0].sort().values, fresh[0].sort().values)
            session.freeze()
    finally:
        dist.destroy_process_group()


@support.requires_reference
@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1", reason="requires eight assigned GPUs")
def test_recursive_draft_selection_and_inactive_lane(tmp_path):
    if torch.cuda.device_count() != 8:
        pytest.skip("requires eight visible devices")
    mp.spawn(_rank, args=((tmp_path / "rendezvous").as_uri(), support.SNAPSHOT, support.REFERENCE), nprocs=8)
