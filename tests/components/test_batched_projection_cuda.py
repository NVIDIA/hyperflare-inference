# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Batched BF16 projections keep multi-row strided inputs within mapped device memory."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA device")


def _tail_slice(device: torch.device, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Place a BF16 slice at the end of a segment below freed virtual address space."""
    candidates = [torch.empty(64 * 1024 * 1024, dtype=torch.bfloat16, device=device) for _ in range(4)]
    candidates.sort(key=lambda tensor: tensor.data_ptr())
    backing = candidates[-2]
    del candidates[-1]
    candidates = candidates[:-1]
    del candidates
    torch.cuda.empty_cache()
    view = backing[-values.numel():]
    view.copy_(values)
    return backing, view


@pytest.mark.parametrize(
    ("rows", "heads", "full_in", "in_features", "out_features", "full_out"),
    [
        (2, 8, 512, 512, 256, 256),
        (4, 8, 256, 192, 512, 576),
        (5, 8, 192, 128, 512, 576),
    ],
)
@torch.no_grad()
def test_batched_bf16_projection_handles_segment_tail_interleaved_inputs(
    rows: int, heads: int, full_in: int, in_features: int, out_features: int, full_out: int
):
    from ignition.components.linear import BatchedUnquantizedProjection
    from ignition.preparation.compilation import compile_forward

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(5100 + rows * 31 + in_features)
    weight = (torch.randn((heads, in_features, out_features), generator=generator) * 0.03).to(
        dtype=torch.bfloat16, device=device)
    expected_source = torch.empty((rows, heads, full_in), dtype=torch.bfloat16, device=device)
    output_storage = torch.full((rows, heads, full_out), -17, dtype=torch.bfloat16, device=device)
    output_view = output_storage[:, :, :out_features].transpose(0, 1)

    projection = BatchedUnquantizedProjection(
        expected_source[:, :, :in_features].transpose(0, 1),
        weight,
        output_view,
        major="n",
        session=None,
        name="test.batched_tail",
    )
    functional = BatchedUnquantizedProjection(
        expected_source[:, :, :in_features].transpose(0, 1),
        weight,
        torch.empty_strided(output_view.shape, output_view.stride(), dtype=torch.bfloat16, device=device),
        major="n",
        session=None,
        name="test.batched_tail_functional",
    ).functional(session=None)

    # Prime cuBLAS handles and compute the reference before placing the operand adjacent to unmapped memory.
    host_values = torch.randn(rows * heads * full_in, generator=generator, dtype=torch.float32).to(torch.bfloat16)
    reference_input = host_values.to(device).view(rows, heads, full_in)[:, :, :in_features].transpose(0, 1).contiguous()
    reference = torch.bmm(reference_input, weight).transpose(0, 1)
    del reference_input
    projection(torch.zeros((heads, rows, in_features), dtype=torch.bfloat16, device=device))
    torch.cuda.synchronize(device)

    backing, tail = _tail_slice(device, host_values)
    interleaved = tail.view(rows, heads, full_in)[:, :, :in_features].transpose(0, 1)

    local = functional(interleaved)
    torch.cuda.synchronize(device)
    assert local.stride() == output_view.stride()
    torch.testing.assert_close(local.transpose(0, 1), reference, rtol=0, atol=0)

    def forward():
        projection(interleaved)

    compiled = compile_forward(forward)
    output_storage.fill_(-17)
    compiled()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(output_storage[:, :, :out_features], reference, rtol=0, atol=0)
    if full_out > out_features:
        assert output_storage[:, :, out_features:].eq(-17).all()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    output_storage.fill_(-17)
    before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
    torch.testing.assert_close(output_storage[:, :, :out_features], reference, rtol=0, atol=0)
    graph.reset()
    del backing


def test_single_row_batched_bf16_projection_preserves_strided_input_view(monkeypatch):
    from ignition.components import linear

    seen_ptrs: list[tuple[int, tuple[int, ...]]] = []

    def record_bmm(source, weight, out):
        seen_ptrs.append((source.data_ptr(), source.stride()))
        out.copy_(torch.bmm(source.contiguous(), weight))

    monkeypatch.setattr(linear, "_bmm_out", record_bmm)
    raw = torch.arange(8 * 256, dtype=torch.bfloat16).view(1, 8, 256)
    source = raw[:, :, :192].transpose(0, 1)
    weight = torch.ones((8, 192, 64), dtype=torch.bfloat16)
    output = torch.zeros((1, 8, 96), dtype=torch.bfloat16)[:, :, :64].transpose(0, 1)
    projection = linear.BatchedUnquantizedProjection(
        source, weight, output, major="n", session=None, name="test.single_row")
    projection(source)
    assert seen_ptrs == [(source.data_ptr(), source.stride())]
