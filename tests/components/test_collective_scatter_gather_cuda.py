# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused scatter-gather all-reduce RMSNorm preserves shard-owner BF16 rounding."""

from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_TP_CUDA_TESTS") != "1",
                               reason="requires explicitly assigned GPUs")


def _rank(rank, world, rendezvous):
    from dataclasses import replace
    from types import MappingProxyType

    from b12x._lib.compile_plan import load_programs
    from b12x.comm.pcie._oneshot_preparation import (
        _OneshotExecutionState, compile_oneshot_surface,
    )
    from b12x.comm.pcie._tuning import TUNING
    from b12x.preparation import FrozenMapping
    from b12x.preparation.types import require_prepared
    from ignition.components.collectives import PreparedAllReduceRMSNorm
    from ignition.preparation.collectives import CollectivePreparation, create_all_reduce_runtime
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=5))
    graphs, controls = [], []
    try:
        with CoreStorage(rank=rank, world=world, device=rank) as storage:
            coordinator = CollectivePreparation(dist.group.WORLD)
            runtime = create_all_reduce_runtime(dist.group.WORLD, device, max_tokens=64,
                hidden_size=6144, element_bytes=4, ipc=storage.peer_storage)
            try:
                with RecordedPreparationSession(device=device, checkpoint="scatter-gather-shard-owner",
                        workload_schema="shard-owner-cyclic-reduction", numerical_policy="source-dtype",
                        coordinator=coordinator, autotune_effort="off") as session:
                    calls = []
                    for rows in (1, 4):
                        for reduction_dtype in (torch.bfloat16, None):
                            source = torch.zeros((rows, 6144), device=device, dtype=torch.bfloat16)
                            addend = torch.zeros_like(source) if rows == 4 else None
                            residual = torch.zeros_like(source)
                            reference_residual = torch.zeros_like(source)
                            output, reference_output = torch.empty_like(source), torch.empty_like(source)
                            gain = torch.linspace(-.5, .5, 6144, device=device, dtype=torch.bfloat16)
                            operation = PreparedAllReduceRMSNorm(source, residual, gain, output,
                                eps=1e-6, runtime=runtime, session=session, coordinator=coordinator,
                                name=f"sg:{rows}:{reduction_dtype}", zero_centered=False,
                                input_addend=addend, round_before_weight=True,
                                reduction_dtype=reduction_dtype, reduction_order="shard_owner")
                            calls.append((rows, reduction_dtype, source, addend, residual,
                                          reference_residual, output, reference_output, gain, operation))
                    session.tune(show_progress=False)
                    stream = torch.cuda.Stream(device=device)
                    for rows, reduction_dtype, source, addend, residual, reference_residual, output, reference_output, gain, operation in calls:
                        state = require_prepared(operation.plan, "comm.pcie", device)
                        sg_mode = "stage_scatter_gather_packed" if reduction_dtype is not None else "stage_scatter_gather"
                        sg_call = dict(state.query.call, mode=sg_mode, threads=768, reg_packs=1,
                                       single_cta=True, register_normalize=True, ctas_per_row=1, blocks=rows)
                        pull_call = dict(state.query.call, mode="stage_pull", threads=256, reg_packs=3,
                                         single_cta=True, register_normalize=True, ctas_per_row=1, blocks=rows)
                        sg_query = replace(state.query, call=FrozenMapping(sg_call))
                        pull_query = replace(state.query, call=FrozenMapping(pull_call))
                        sg_state = _OneshotExecutionState(
                            sg_query, state.runtime,
                            MappingProxyType(load_programs(compile_oneshot_surface(TUNING.encode_query(sg_query), rank))))
                        pull_state = _OneshotExecutionState(
                            pull_query, state.runtime,
                            MappingProxyType(load_programs(compile_oneshot_surface(TUNING.encode_query(pull_query), rank))))
                        sg_state.bind_input(source)
                        pull_state.bind_input(source)
                        controls.extend((sg_state, pull_state))
                        sg_state.run_fused(source, residual, gain, output, residual, 1e-6, addend)
                        pull_state.run_fused(source, reference_residual, gain, reference_output, reference_residual, 1e-6, addend)
                        torch.cuda.synchronize(device)
                        for runner, res_buf, out_buf in (
                            (sg_state, residual, output),
                            (pull_state, reference_residual, reference_output),
                        ):
                            graph = torch.cuda.CUDAGraph()
                            stream.wait_stream(torch.cuda.current_stream(device))
                            with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
                                runner.run_fused(source, res_buf, gain, out_buf, res_buf, 1e-6, addend)
                            graphs.append(graph)
                    session.freeze()
                    for turn in range(3):
                        for index, (_, _, source, addend, residual, reference_residual, output, reference_output, _, _) in enumerate(calls):
                            source.normal_(std=.125)
                            amplitude = 2.**24
                            pattern = (amplitude, 2.**-12, -amplitude, 2.**-12)
                            source[:, ::2] = pattern[(rank + turn) % len(pattern)]
                            if addend is not None:
                                addend.zero_()
                                addend[:, 1::2].fill_(rank / 256)
                            residual.fill_((rank + 1) / 8192)
                            reference_residual.copy_(residual)
                            output.fill_(float("nan"))
                            reference_output.fill_(float("nan"))
                            for graph in graphs[index * 2:index * 2 + 2]:
                                torch.cuda.synchronize(device)
                                dist.barrier()
                                graph.replay()
                                torch.cuda.synchronize(device)
                            torch.testing.assert_close(residual, reference_residual, atol=0, rtol=0)
                            torch.testing.assert_close(output, reference_output, atol=0, rtol=0)
            finally:
                for graph in graphs:
                    graph.reset()
                runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [4, 8])
def test_scatter_gather_preserves_shard_owner_bits(tmp_path, world):
    import inspect
    from b12x.comm.pcie._oneshot_cute import _FusedOneshotLaunch

    if "reg_packs" not in inspect.signature(_FusedOneshotLaunch.__init__).parameters:
        pytest.skip("requires b12x stage_scatter_gather oneshot RMSNorm")
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned GPUs")
    mp.spawn(_rank, args=(world, (tmp_path / f"sg-{world}").as_uri()), nprocs=world, join=True)
