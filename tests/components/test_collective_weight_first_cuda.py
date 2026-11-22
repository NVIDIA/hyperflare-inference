# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""An early-launch normalization must retain two-shot shard order and GLM's BF16 rounding."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _rank(rank, world, rendezvous):
    from b12x.comm.pcie import _twoshot_preparation as twoshot
    from b12x.preparation import CollectiveRequirement
    from b12x.preparation.types import require_prepared
    from ignition.components.collectives import PreparedAllReduceRMSNorm
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tuning import RecordedPreparationSession, declare
    from ignition.reference.glm_moe_dsa_bounds import rms_norm_interval

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=8))
    try:
        with CoreStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=4096,
                                                hidden_size=6144, ipc=storage.peer_storage)
            graphs = []
            try:
                with RecordedPreparationSession(device=device, checkpoint="shard-ordered-norm",
                        workload_schema="tp8-bf16-residual", numerical_policy="round-before-scale",
                        coordinator=coordinator, compile_workers=8, autotune_effort="off") as session:
                    weight = torch.linspace(.37, 1.31, 6144, dtype=torch.bfloat16, device=device)
                    calls = []
                    for rows in (1, 2, 3, 4, 6, 8, 13, 16):
                        src = torch.zeros((rows, 6144), device=device, dtype=torch.bfloat16)
                        residual, normalized, summed = (torch.zeros_like(src) for _ in range(3))
                        addend = torch.zeros_like(src) if rows == 4 else None
                        transport = runtime._bf16_twoshot
                        query = twoshot.query_from_runtime(transport, surface="PCIeTwoShotBF16.all_reduce",
                                                          call={"inp": src, "out": summed})
                        plain = twoshot.plan(query, runtime=transport)
                        name = f"two-shot.{rows}"
                        declare(session, (plain.request(name=name, collective=CollectiveRequirement(
                            key=name, ranks=tuple(range(world))), prepare_call=lambda state, src=src, summed=summed:
                            twoshot.prepared_call(state, payload=src, out=summed)),), coordinator=coordinator)
                        fused = PreparedAllReduceRMSNorm(src, residual, weight, normalized, eps=1e-5,
                            runtime=runtime, session=session, coordinator=coordinator, name=f"fused.{rows}",
                            reduction_dtype=torch.bfloat16, round_before_weight=True,
                            reduction_order="shard_owner", trigger_dependents=True, input_addend=addend)
                        calls.append((src, summed, residual, normalized, plain, fused, addend))
                    session.tune(show_progress=False)
                    if rank == 0:
                        print("Prepared the two-shot control and fused reductions", flush=True)
                    stream = torch.cuda.Stream(device=device)
                    for src, summed, residual, normalized, plain, fused, addend in calls:
                        state = require_prepared(fused.plan, "comm.pcie", device)
                        assert state.query.call["trigger_dependents"] and state.query.call["mode"] == "stage_pull"
                        transport.all_reduce(src if addend is None else torch.add(src, addend), summed, plan=plain)
                        fused(src, residual)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        stream.wait_stream(torch.cuda.current_stream())
                        with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                            transport.all_reduce(src if addend is None else torch.add(src, addend), summed, plan=plain)
                            fused(src, residual)
                        graphs.append(graph)
                        if rank == 0:
                            print(f"Captured {src.shape[0]} rows", flush=True)
                    session.freeze()
                    failures = []
                    for iteration in range(3):
                        for graph, (src, summed, residual, normalized, plain, _, addend) in zip(graphs, calls, strict=True):
                            generator = torch.Generator().manual_seed(1740 + src.shape[0] + 3 * iteration)
                            inputs = torch.randn((world, *src.shape), generator=generator).bfloat16()
                            pattern = (2.**24, 1., -(2.**24), 1., 1., 1., 1., 1.)
                            for peer in range(world):
                                inputs[peer, :, ::4] = pattern[(peer + iteration) % world]
                            initial = torch.randn(src.shape, generator=generator).bfloat16()
                            src.copy_(inputs[rank])
                            if addend is not None:
                                shares = torch.randn(inputs.shape, generator=generator).bfloat16()
                                shares[:, :, ::4] = 0
                                addend.copy_(shares[rank])
                                inputs = inputs + shares
                            residual.copy_(initial)
                            normalized.fill_(float("nan"))
                            expected = torch.empty_like(initial).flatten()
                            shard_size = ((src.numel() // 8 + world - 1) // world) * 8
                            for owner in range(world):
                                start, stop = owner * shard_size, min((owner + 1) * shard_size, src.numel())
                                value = torch.zeros(stop - start, dtype=torch.float32)
                                for step in range(world):
                                    value += inputs[(owner + step) % world].flatten()[start:stop].float()
                                expected[start:stop] = value
                            expected = expected.view_as(initial)
                            graph.replay()
                            torch.cuda.synchronize()
                            try:
                                torch.testing.assert_close(summed.cpu(), expected, rtol=0, atol=0)
                                if addend is not None:
                                    local_order = torch.zeros(src.shape)
                                    duplicated = torch.empty_like(expected).flatten()
                                    for peer in range(world):
                                        local_order += inputs[(rank + peer) % world].float()
                                    for owner in range(world):
                                        begin, end = owner * shard_size, min((owner+1)*shard_size, src.numel())
                                        value = torch.zeros(end-begin)
                                        for step in range(world):
                                            peer = (owner+step) % world
                                            value += (inputs[peer] + shares[peer]).flatten()[begin:end].float()
                                        duplicated[begin:end] = value
                                    assert not torch.equal(summed.cpu(), local_order.bfloat16())
                                    assert not torch.equal(summed.cpu(), duplicated.view_as(expected))
                                expected_residual = initial + expected
                                torch.testing.assert_close(residual.cpu(), expected_residual, rtol=0, atol=0)
                                lower, upper = rms_norm_interval(expected_residual, weight.cpu(), 1e-5)
                                actual = normalized.cpu().double()
                                assert bool(((actual >= lower) & (actual <= upper)).all())
                            except AssertionError as exc:
                                failures.append(f"{src.shape[0]} rows, replay {iteration}: {exc}")
                    assert not failures, "\n".join(failures)
            finally:
                for graph in graphs:
                    graph.reset()
                runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1" or torch.cuda.device_count() < 8,
                    reason="requires eight assigned PCIe GPUs")
def test_two_shot_shard_order_and_round_before_scale(tmp_path):
    mp.spawn(_rank, args=(8, (tmp_path / "collective").as_uri()), nprocs=8, join=True)
