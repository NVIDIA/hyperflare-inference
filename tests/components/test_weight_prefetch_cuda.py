# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay compiled side-stream weight reads across allocation reuse and partial blocks."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="CUDA opt-in")


def test_compiled_prefetch_reads_current_weights_and_joins_before_scratch_reuse():
    from ignition import runtime_patches
    from ignition.components.weight_prefetch import PreparedWeightPrefetch, join_weight_prefetch
    from ignition.preparation.scratch import ScratchArena

    runtime_patches.apply()
    device = torch.device("cuda", 0)
    programs = 4
    words = torch.arange(113_499, dtype=torch.int64, device=device)
    stream = ScratchArena(device).auxiliary_stream
    reader = PreparedWeightPrefetch([words.view(torch.bfloat16)], programs, stream)
    handle = stream.cuda_stream

    def forward(operand):
        sums = torch.empty(programs, dtype=torch.int64, device=operand.device)
        reader(operand, sums)
        main = operand * 3
        join_weight_prefetch(main, sums, handle)
        return main, sums.clone()

    compiled = torch.compile(forward, fullgraph=True, options={"triton.cudagraphs": False})
    operand = torch.tensor([3], device=device)
    compiled(operand)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        main, sums = compiled(operand)
    for offset in (1, 23, 51):
        words.add_(offset)
        operand.add_(offset)
        graph.replay()
        assert main.item() == 3 * operand.item()
        assert sums.sum().item() == words.sum().item()


@pytest.mark.parametrize("enabled", [False, True])
def test_decode_graph_keeps_prefetch_after_its_sums_are_discarded(enabled):
    from cuda.bindings import driver
    from ignition import runtime_patches
    from ignition.components.weight_prefetch import PreparedWeightPrefetch, join_weight_prefetch
    from ignition.preparation.scratch import ScratchArena

    runtime_patches.apply()
    device = torch.device("cuda", 0)
    stream = ScratchArena(device).auxiliary_stream
    weight = torch.ones((257, 256), dtype=torch.bfloat16, device=device)
    reader = PreparedWeightPrefetch([weight], 4, stream)
    handle = stream.cuda_stream

    def forward(operand):
        if enabled:
            sums = torch.empty(4, dtype=torch.int64, device=operand.device)
            reader(operand, sums)
        output = operand + 1
        if enabled:
            join_weight_prefetch(output, sums, handle)
        return output

    compiled = torch.compile(forward, fullgraph=True, options={"triton.cudagraphs": False})
    operand = torch.tensor([2.0], device=device)
    compiled(operand)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        result = compiled(operand)
    raw = driver.CUgraph(graph.raw_cuda_graph())
    status, _, count = driver.cuGraphGetNodes(raw, 0)
    assert status == driver.CUresult.CUDA_SUCCESS
    status, nodes, _ = driver.cuGraphGetNodes(raw, count)
    assert status == driver.CUresult.CUDA_SUCCESS
    names = []
    for node in nodes:
        status, kind = driver.cuGraphNodeGetType(node)
        assert status == driver.CUresult.CUDA_SUCCESS
        if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            status, params = driver.cuGraphKernelNodeGetParams(node)
            assert status == driver.CUresult.CUDA_SUCCESS
            status, name = driver.cuFuncGetName(params.func)
            assert status == driver.CUresult.CUDA_SUCCESS
            names.append(name.decode())
    assert sum("_prefetch_weight_l2" in name for name in names) == int(enabled), names
    operand.fill_(5)
    graph.replay()
    assert result.item() == 6.0
