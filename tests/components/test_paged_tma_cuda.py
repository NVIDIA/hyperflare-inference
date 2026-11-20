# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM120 paged FP8 TMA prefill passes the gate and range arguments the kernel declares."""

import os

import pytest
import torch

from ignition.components.paged_attention import PreparedPagedAttention
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")


def test_paged_tma_prefill_forwards_gate_and_kv_tile_ranges():
    device = torch.device("cuda", 0)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        pytest.skip("paged TMA prefill requires SM120 or SM121")
    rows, heads, kv_heads, dim, page, pages = 256, 4, 1, 256, 128, 2
    generator = torch.Generator(device=device).manual_seed(1147)
    draw = lambda shape: torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    query = draw((rows, heads, dim)).to(torch.float8_e4m3fn)
    key = draw((pages, page, kv_heads, dim)).to(torch.float8_e4m3fn)
    value = draw((pages, page, kv_heads, dim)).to(torch.float8_e4m3fn)
    gate = draw((rows, heads, dim)) * 2
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    lengths = torch.tensor([rows], device=device, dtype=torch.int32)
    cu = torch.tensor([0, rows], device=device, dtype=torch.int32)
    outputs, operations = {}, {}
    with RecordedPreparationSession(device=device, checkpoint="paged-tma-arguments",
            workload_schema="paged-tma-arguments", numerical_policy="default") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        for gated in (False, True):
            outputs[gated] = torch.empty((rows, heads, dim), device=device, dtype=torch.bfloat16)
            operations[gated] = PreparedPagedAttention(query, key, value, table, lengths, cu,
                outputs[gated], mode="prefill", output_gate=gate if gated else None,
                session=session, workspace=arena, name=f"paged-tma-{gated}")
            assert operations[gated].tma is not None, "geometry must select the paged TMA path"
        session.tune()
        graphs = {}
        for gated, operation in operations.items():
            compiled = compile_forward(operation)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs[gated] = graph
        session.freeze()
        for gated, graph in graphs.items():
            outputs[gated].fill_(float("nan"))
            graph.replay()
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()

    flat_key = key[:, :, 0].float().reshape(pages * page, dim)
    flat_value = value[:, :, 0].float().reshape(pages * page, dim)
    flat_query = query.float()
    for gated, output in outputs.items():
        scale = dim ** .5
        for row in (0, 1, 127, 128, 255):
            scores = flat_query[row] @ flat_key[:row + 1].T / scale
            expected = scores.softmax(-1) @ flat_value[:row + 1]
            if gated:
                expected = expected * gate[row].float().sigmoid()
            torch.testing.assert_close(output[row].float(), expected, atol=.02, rtol=.05)
