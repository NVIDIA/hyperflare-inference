# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Packed convolution arithmetic, history snapshots, and accepted-prefix replay."""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def _metadata(tokens, device):
    lengths = torch.tensor([tokens // 2, 0, tokens - tokens // 2 - 3, 1], dtype=torch.int32, device=device)
    begins = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), lengths.cumsum(0).int()))
    starts = torch.tensor([0, 7, 11, 19], dtype=torch.int32, device=device)
    rows = torch.arange(tokens, device=device)
    requests = (rows[:, None] >= begins[1:]).sum(-1).clamp_max(3)
    live = rows < begins[-1]
    positions = torch.where(live, starts[requests] + rows - begins[requests], -1)
    requests = torch.where(live, requests, -1)
    return SimpleNamespace(lengths=lengths, starts=starts, query_start_loc=begins,
        state_rows=torch.tensor([3, 5, 1, 6], dtype=torch.int64, device=device),
        requests=requests, positions=positions)


def _reference(source, weight, history, metadata, dynamic, counts):
    x, w, h = source.cpu(), weight.cpu().float(), history.cpu()
    d = None if dynamic is None else dynamic.cpu().float()
    output, updated = torch.zeros_like(x), h.clone()
    previous = h[metadata.state_rows.cpu()].clone()
    begins, starts, lengths = (getattr(metadata, name).cpu().tolist()
                               for name in ("query_start_loc", "starts", "lengths"))
    width, channels = h.shape[1] + 1, h.shape[2]
    for request, (start, length, accepted, slot) in enumerate(zip(
            starts, lengths, counts.cpu().tolist(), metadata.state_rows.cpu().tolist(), strict=True)):
        first = begins[request]
        prior = previous[request].float().clone()
        prior[torch.arange(width - 1) + start - width + 1 < 0] = 0
        context = torch.cat((prior, x[first:first + length].float()))
        total = None
        for tap in range(width):
            if d is None:
                term = context[tap:tap + length] * w.reshape(channels, width)[:, tap]
            else:
                delta = d[first:first + length, :, tap].repeat_interleave(channels // d.shape[1], dim=1)
                term = context[width - 1 - tap:width - 1 - tap + length] * (w[tap] + delta)
            total = term if total is None else total + term
        scale = (width * (1 if d is None else 2) + 1) ** -.5
        output[first:first + length] = ((total + x[first:first + length].float()) * scale).bfloat16()
        if accepted:
            updated[slot] = context[accepted:accepted + width - 1].bfloat16()
    return output, previous, updated


def _nodes(graph):
    from cuda.bindings import driver

    status, _, count = driver.cuGraphGetNodes(driver.CUgraph(graph.raw_cuda_graph()))
    assert status == driver.CUresult.CUDA_SUCCESS
    return count


@pytest.mark.parametrize("channels,heads,tokens,weight_dtype", [
    (4608, 0, 8, torch.float32), (4608, 0, 64, torch.float32), (4608, 0, 4096, torch.float32),
    (256, 2, 8, torch.bfloat16), (256, 2, 64, torch.bfloat16), (256, 2, 4096, torch.bfloat16),
    (257, 0, 11, torch.bfloat16), (258, 2, 11, torch.float32),
])
@torch.no_grad()
def test_packed_causal_convolution_replay_preserves_raw_history(channels, heads, tokens, weight_dtype):
    from ignition import runtime_patches
    from ignition.components.causal_convolution import PreparedCausalConvolution

    runtime_patches.apply()
    device = torch.device("cuda:0")
    torch.manual_seed(812)
    metadata = _metadata(tokens, device)
    source = torch.randn(tokens, channels + 17, dtype=torch.bfloat16, device=device)[:, :channels]
    backing = torch.randn(8, 3 * channels + 32, dtype=torch.bfloat16, device=device)
    history = backing[:, :3 * channels].view(8, 3, channels)
    original = history.clone()
    padding = backing[:, 3 * channels:].clone()
    weight = torch.randn((4, channels) if heads else (channels, 1, 4), dtype=weight_dtype, device=device)
    dynamic = (torch.randn(tokens, heads * 4 + 5, dtype=torch.bfloat16, device=device)[:, :heads * 4]
               .view(tokens, heads, 4)) if heads else None
    operation = PreparedCausalConvolution(weight, history, dynamic_heads=heads)
    counts = metadata.lengths.clone()
    output, previous = operation(source, metadata, dynamic=dynamic)
    assert torch.equal(history, original)
    operation.commit(previous, source, metadata, counts=counts)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        output, previous = operation(source, metadata, dynamic=dynamic)
        operation.commit(previous, source, metadata, counts=counts)
    graph.instantiate()
    assert _nodes(graph) == 2
    for accepted in (metadata.lengths, metadata.lengths // 2, torch.zeros_like(counts)):
        history.copy_(original)
        counts.copy_(accepted)
        output.fill_(float("nan"))
        previous.fill_(float("nan"))
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        expected, snapshot, updated = _reference(source, weight, original, metadata, dynamic, counts)
        assert torch.equal(output.cpu(), expected)
        assert torch.equal(previous.cpu(), snapshot)
        assert torch.equal(history.cpu(), updated)
        assert torch.equal(backing[:, 3 * channels:], padding)


@pytest.mark.parametrize("heads", [0, 2])
@torch.no_grad()
def test_verification_history_snapshot_survives_until_separate_commit(heads):
    from ignition import runtime_patches
    from ignition.components.causal_convolution import PreparedCausalConvolution

    runtime_patches.apply()
    device = torch.device("cuda:0")
    metadata = _metadata(64, device)
    source = torch.randn(64, 273, dtype=torch.bfloat16, device=device)[:, :256]
    history = torch.randn(8, 3, 256, dtype=torch.bfloat16, device=device)
    original = history.clone()
    weight = torch.randn((4, 256) if heads else (256, 1, 4), dtype=torch.float32, device=device)
    dynamic = torch.randn(64, heads, 4, dtype=torch.bfloat16, device=device) if heads else None
    operation = PreparedCausalConvolution(weight, history, dynamic_heads=heads)
    output, previous = operation(source, metadata, dynamic=dynamic)
    counts = torch.tensor([1, 0, 2, 0], dtype=torch.int32, device=device)
    operation.commit(previous, source, metadata, counts=counts)
    verify = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(verify):
        output, previous = operation(source, metadata, dynamic=dynamic)
    verify.instantiate()
    commit = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(commit):
        operation.commit(previous, source, metadata, counts=counts)
    commit.instantiate()
    assert _nodes(verify) == _nodes(commit) == 1
    history.copy_(original)
    verify.replay()
    expected, snapshot, updated = _reference(source, weight, original, metadata, dynamic, counts)
    assert torch.equal(history, original)
    history[metadata.state_rows[2]].fill_(float("nan"))
    commit.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.cpu(), expected)
    assert torch.equal(previous.cpu(), snapshot)
    assert torch.equal(history.cpu(), updated)
