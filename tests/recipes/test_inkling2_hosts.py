# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 TP4 host placement and channels preserve the local preparation contract."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ignition import planning
from ignition.recipes.inkling2 import bootstrap, decoder, recipe

HOSTS = ("hertz", "maxwell", "ampere", "faraday")
HIDDEN, LOCAL_VOCABULARY, DRAFT_BLOCK = 4608, 201024 // 4, 8
GEOMETRY = dict(hidden=HIDDEN, local_vocabulary=LOCAL_VOCABULARY, draft_block=DRAFT_BLOCK)


def test_four_hosts_may_each_use_their_own_gpu_zero():
    config = bootstrap.PreparationConfig(devices=(0,) * 4, hosts=HOSTS)
    assert config.devices == (0,) * 4
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 4)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 4, hosts=("hertz",) * 4)
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0,) * 4, hosts=HOSTS[:3])
    with pytest.raises(ValueError, match="distinct GPUs"):
        bootstrap.PreparationConfig(devices=(0, 1), hosts=HOSTS[:2])


def test_ring_and_chain_need_hosts_in_rank_order():
    with pytest.raises(ValueError, match="--roce-ring"):
        bootstrap.PreparationConfig(roce_ring=True)
    with pytest.raises(ValueError, match="exclude each other"):
        bootstrap.PreparationConfig(devices=(0,) * 4, hosts=HOSTS, roce_ring=True, roce_chain=True)
    assert bootstrap.PreparationConfig(devices=(0,) * 4, hosts=HOSTS, roce_ring=True).roce_ring
    assert bootstrap.PreparationConfig(devices=(0,) * 4, hosts=HOSTS, roce_chain=True).roce_chain


def _gb10(count):
    return tuple(planning.Gpu(i, "GB10", f"GPU-{i}", 121 << 30, 115 << 30, (12, 1)) for i in range(count))


@pytest.mark.parametrize("concurrency, chunk, capacity", [(8, 4096, 64)])
def test_four_gb10_hosts_fit_the_declared_configuration(concurrency, chunk, capacity):
    speculation = replace(recipe.RECIPE.speculation, token_capacity=capacity)
    plan = planning.plan(recipe.RECIPE, recipe.RECIPE.checkpoints[0], _gb10(4), selected=(0, 1, 2, 3),
                         concurrency=concurrency, reserve_bytes=4 << 30, speculation=speculation,
                         prefill_chunk=chunk)
    assert plan.fits and plan.world_size == 4
    assert (plan.concurrency, plan.prefill_chunk, plan.speculative_token_capacity) == (concurrency, chunk, capacity)
    assert plan.guaranteed_tokens >= planning.MINIMUM_REQUEST_TOKENS
    # Request rows, their prefix-cache snapshots and the chunk's scratch leave most of the supply to KV pages.
    assert plan.page_bytes >= 16 << 30


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("ring", [False, True])
def test_recipe_dispatches_the_complete_plan_to_the_rank_group(monkeypatch, remote, ring):
    from ignition.preparation import rank_group

    ring = ring and remote
    plan = planning.fit(recipe.RECIPE, recipe.RECIPE.checkpoints[0], _gb10(4), concurrency=8,
                        reserve_bytes=4 << 30, speculation=recipe.RECIPE.speculation, roce_ring=ring)
    ranks = SimpleNamespace(hosts=HOSTS) if remote else None
    calls, result, storage = [], object(), object()
    monkeypatch.setattr(rank_group, "prepare_group", lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    assert recipe.prepare("checkpoint", plan, compile_forwards=True, kernel_cache=None, progress=None,
                          core_storage=storage, ranks=ranks) is result
    (snapshot, config, forward), kwargs = calls[0]
    assert snapshot == "checkpoint" and forward is bootstrap.prepare_rank
    assert config.devices == ((0,) * 4 if remote else (0, 1, 2, 3))
    assert config.hosts == (HOSTS if remote else None)
    assert config.roce_ring is ring and not config.roce_chain
    assert config.max_active == 8 and config.drafting and config.vision and config.draft_tokens == 7
    assert kwargs["ranks"] is ranks and kwargs["checkpoint"] is plan.checkpoint
    assert kwargs["core_storage"] is storage


@pytest.mark.parametrize("drafting", [False, True])
@pytest.mark.parametrize("chain", [False, True])
def test_remote_channels_share_one_roce_runtime_without_pcie(monkeypatch, drafting, chain):
    from ignition.components.collectives import packed_broadcast_layout
    from ignition.components.vocabulary_logprobs import _SUMMARY
    from ignition.components.vocabulary_sampling import SUMMARY_WORDS

    config = bootstrap.PreparationConfig(devices=(0,) * 4, hosts=HOSTS, max_active=8, drafting=drafting,
                                         roce_chain=chain)
    calls, runtime, storage = [], object(), object()
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
                        lambda group, device, **kwargs: calls.append(kwargs) or runtime)
    for name in ("create_all_reduce_runtime", "create_all_gather_runtime"):
        monkeypatch.setattr(bootstrap, name, lambda *args, **kwargs: pytest.fail("host ranks cannot use CUDA IPC"))
    channels = bootstrap._collective_runtimes(config, None, None,
        coordinator=SimpleNamespace(spans_hosts=True), core_storage=storage, **GEOMETRY)
    assert channels == (runtime, runtime, runtime if drafting else None, runtime if drafting else None)
    (call,) = calls
    assert call["core_storage"] is storage and call["hidden_size"] == HIDDEN
    assert call["ring"] is False and call["ring_open"] is chain
    # One reduction covers a whole prefill chunk; decode, verification and draft rows reduce one-shot.
    assert call["pipe_tokens"] == max(config.prefill_chunk_size, config.decode_tokens) == 4096
    assert call["max_tokens"] >= config.decode_tokens
    assert call["max_tokens"] * HIDDEN * 2 % 16 == 0 and call["pipe_tokens"] * HIDDEN * 2 % 16 == 0
    rows = max(config.decode_samples, 8)
    # FP32 vocabulary partitions of every sampled row travel as BF16 column pairs.
    assert call["gather_bytes"] == rows * LOCAL_VOCABULARY * 4 and call["gather_bytes"] % 16 == 0
    sampled = [((rows,), torch.int64), ((rows,), torch.bool), ((rows,), torch.float32), ((rows,), torch.float32)]
    assert call["gather_bytes"] >= (packed_broadcast_layout(sampled)[1] + 15) // 16 * 16
    if drafting:
        assert call["max_tokens"] >= 8 * DRAFT_BLOCK
        assert (config.decode_tokens, config.decode_samples) == (64, 64)
        # Sampling statistics and log-probability summaries are FP32 rows of 32 and 256 words per rank.
        assert call["gather_bytes"] >= rows * max(SUMMARY_WORDS, _SUMMARY) * 4
    else:
        assert call["max_tokens"] == config.decode_tokens == 8


@pytest.mark.parametrize("drafting", [False, True])
def test_local_channels_are_unchanged(monkeypatch, drafting):
    config = bootstrap.PreparationConfig(max_active=8, drafting=drafting, collective_tuning="runtime")
    gathers, reductions = [], []
    monkeypatch.setattr(bootstrap, "create_all_gather_runtime",
                        lambda group, device, **kwargs: gathers.append(kwargs) or object())
    monkeypatch.setattr(bootstrap, "create_all_reduce_runtime",
                        lambda group, device, **kwargs: reductions.append(kwargs) or object())
    monkeypatch.setattr(bootstrap, "create_roce_runtime",
                        lambda *args, **kwargs: pytest.fail("local ranks share CUDA IPC"))
    ipc = object()
    channels = bootstrap._collective_runtimes(config, None, None,
        coordinator=SimpleNamespace(spans_hosts=False), core_storage=SimpleNamespace(peer_storage=ipc), **GEOMETRY)
    present = [channel for channel in channels if channel is not None]
    assert len(present) == (4 if drafting else 2) and len({id(channel) for channel in present}) == len(present)
    assert reductions == [dict(max_tokens=1024, hidden_size=HIDDEN, element_bytes=4, ipc=ipc,
                               collective_tuning="runtime")]
    samples = config.decode_samples
    assert gathers[0] == dict(max_rows=max(samples, 8), columns=2 * 201024, ipc=ipc)
    assert gathers[1:] == ([dict(max_rows=samples, columns=64 * 4, ipc=ipc),
                            dict(max_rows=samples, columns=512 * 4, ipc=ipc)] if drafting else [])


def test_hidden_reductions_round_to_bf16_only_over_roce(monkeypatch):
    local, remote = object(), object()
    monkeypatch.setattr(decoder, "_roce", lambda runtime: runtime is remote)
    assert decoder.reduction_dtype(None) == torch.float32
    assert decoder.reduction_dtype(local) == torch.float32
    assert decoder.reduction_dtype(remote) == torch.bfloat16
