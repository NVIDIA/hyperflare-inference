# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors.
# SPDX-License-Identifier: Apache-2.0

"""Paged relative bias and row sinks remain explicit operands through replay."""

import os

import pytest
import torch

from ignition.components.paged_attention import PreparedPagedAttention
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned SM120 GPU")


def _check(output, q, key, value, pages, lengths, cu, bias, sink, window):
    starts = cu.tolist()
    for request, length in enumerate(lengths.tolist()):
        begin, end = starts[request:request + 2]
        if begin == end:
            continue
        k = key[pages[request].long()].reshape(-1, 2, 128)[:length].double().repeat_interleave(6, 1) * .25
        v = value[pages[request].long()].reshape(-1, 2, 128)[:length].double().repeat_interleave(6, 1) * 3
        distance = (length - end + begin + torch.arange(end - begin, device=q.device)[:, None]
                    - torch.arange(length, device=q.device)[None])
        logits = torch.einsum("thd,khd->thk", q[begin:end].double(), k) / 128 ** .5
        in_extent = (distance >= 0) & (distance < 1024)
        indices = distance.clamp(0, 1023)[:, None].expand(-1, 12, -1)
        logits += bias[begin:end].double().gather(2, indices) * in_extent[:, None]
        visible = distance >= 0
        if window >= 0:
            visible &= distance <= window
        logits.masked_fill_(~visible[:, None], -float("inf"))
        row_sink = sink[begin:end].double()
        positive = torch.isposinf(row_sink)
        logits = torch.cat((logits, row_sink.masked_fill(positive, 0)[:, :, None]), -1)
        probability = logits.softmax(-1)[..., :-1].masked_fill(positive[:, :, None], 0)
        expected = torch.einsum("thk,khd->thd", probability, v)
        mass = torch.einsum("thk,khd->thd", probability, v.abs())
        bound = 2 ** -8 * (mass + expected.abs()) + 2 ** -11 * mass + 2 ** -126 * length * v.abs().max()
        excess = (output[begin:end].double() - expected).abs() - bound
        assert excess.max().item() <= 0, excess.max().item()
        assert torch.isfinite(output[begin:end]).all()


@pytest.mark.parametrize("mode,width", [("extend", 35), ("verify", 8), ("decode", 1)])
@pytest.mark.parametrize("window", [-1, 1023])
@pytest.mark.parametrize("functional", [False, True])
def test_relative_bias_and_sink_replay(mode, width, window, functional):
    device = torch.device("cuda", 0)
    rows = 3 * width
    q = torch.randn(rows, 12, 128, dtype=torch.bfloat16, device=device) * .3
    key = (torch.randn(33, 128, 2, 128, device=device) * .5).to(torch.float8_e4m3fn)
    value = (torch.randn(33, 128, 2, 128, device=device) * .5).to(torch.float8_e4m3fn)
    pages = torch.randperm(33, dtype=torch.int32, device=device).view(3, 11)
    lengths = torch.tensor([1408, 1024, 1025], dtype=torch.int32, device=device)
    cu = torch.tensor([0, width, width * 2, width * 3], dtype=torch.int32, device=device)
    factors = torch.randn(rows, 12, 16, dtype=torch.bfloat16, device=device) * .5
    projection = torch.randn(1024, 16, dtype=torch.bfloat16, device=device) * .5
    bias = torch.einsum("thr,dr->thd", factors.float(), projection.float()).bfloat16()
    sink = torch.randn(rows, 12, device=device) * 2
    sink[::3, ::5] = float("inf")
    sink[1::3, 1::5] = -float("inf")
    scale_k, scale_v = torch.tensor([.25], device=device), torch.tensor([3.], device=device)
    bias_arguments = (dict(relative_bias_factors=factors, relative_bias_projection=projection)
                      if mode == "extend" else dict(relative_bias=bias))
    with RecordedPreparationSession(device=device, checkpoint="paged-relative-bias",
            workload_schema="paged-relative-bias", numerical_policy="default", autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        attention = PreparedPagedAttention(q, key, value, pages, lengths, cu, torch.empty_like(q),
            mode=mode, width=width, window=window, sink=sink, k_descale=scale_k, v_descale=scale_v,
            session=session, workspace=arena, name="attention", **bias_arguments)
        if functional:
            operation = attention.functional(session=session, sink=None,
                **{name: None for name in bias_arguments})

            def run():
                return operation(q, sink=sink, **bias_arguments)
        else:
            run = attention
        session.tune()
        output = run()
        torch.cuda.synchronize()
        _check(output, q, key, value, pages, lengths, cu, bias, sink, window)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output = run()
        session.freeze()
        for counts in ([width, 0, 1], [1, width, max(0, width - 1)]):
            cu.copy_(torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], device=device))
            lengths.copy_(torch.tensor([1023, 1024, 1025], device=device))
            q.normal_(std=.3)
            factors.normal_(std=.5)
            projection.normal_(std=.5)
            bias.copy_(torch.einsum("thr,dr->thd", factors.float(), projection.float()).bfloat16())
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            _check(output, q, key, value, pages, lengths, cu, bias, sink, window)
        graph.reset()
