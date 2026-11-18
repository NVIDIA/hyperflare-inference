# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared GroupNorm preserves native outputs and read-only input storage."""

import os

import pytest
import torch

from ignition import runtime_patches
from ignition.preparation.aot import aot_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


class GroupNormForward:
    def __init__(self, dtype, affine, strided):
        self.base = torch.empty((1, 128, 8, 16 if strided else 8), dtype=dtype, device="cuda")
        self.source = self.base[..., ::2] if strided else self.base
        self.weight = torch.empty(128, dtype=dtype, device="cuda") if affine else None
        self.bias = torch.empty(128, dtype=dtype, device="cuda") if affine else None
        self.output = torch.empty((1, 128, 8, 8), dtype=dtype, device="cuda")
        self.mean = torch.empty((1, 32), dtype=dtype, device="cuda")
        self.rstd = torch.empty_like(self.mean)

    def value(self):
        return torch.ops.aten.native_group_norm.default(
            self.source, self.weight, self.bias, 1, 128, 64, 32, 1e-6)

    def __call__(self):
        output, mean, rstd = self.value()
        self.output.copy_(output)
        self.mean.copy_(mean)
        self.rstd.copy_(rstd)

    def outputs(self):
        return self.output, self.mean, self.rstd

    def inputs(self):
        return (self.base,) if self.weight is None else (self.base, self.weight, self.bias)


def _fill_inputs(forward, seed):
    generator = torch.Generator().manual_seed(seed)
    forward.base.copy_(torch.randn(forward.base.shape, generator=generator) * 3)
    if forward.weight is not None:
        forward.weight.copy_(torch.randn(128, generator=generator) * .1 + 1)
        forward.bias.copy_(torch.randn(128, generator=generator) * .1)


def _assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _assert_outputs(forward, expected):
    for actual, reference in zip(forward.outputs(), expected, strict=True):
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
        _assert_bytes(actual, reference)


def _assert_inputs(forward, snapshots):
    # The base includes gaps outside the strided source view.
    for actual, reference in zip(forward.inputs(), snapshots, strict=True):
        _assert_bytes(actual, reference)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_native_group_norm_outputs_survive_serialization_and_graph_replay(tmp_path, dtype, affine, strided):
    runtime_patches.apply()
    forward = GroupNormForward(dtype, affine, strided)
    assert forward.source.is_contiguous() == (not strided)
    _fill_inputs(forward, 42)
    snapshots = tuple(value.clone() for value in forward.inputs())
    expected = forward.value()
    compiled = aot_forward(forward)
    compiled(forward)
    _assert_outputs(forward, expected)
    _assert_inputs(forward, snapshots)
    artifact = tmp_path / "group_norm.bin"
    compiled.save_compiled_function(str(artifact))

    fresh = GroupNormForward(dtype, affine, strided)
    _fill_inputs(fresh, 43)
    snapshots = tuple(value.clone() for value in fresh.inputs())
    expected = fresh.value()
    with artifact.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    restored(fresh)
    _assert_outputs(fresh, expected)
    _assert_inputs(fresh, snapshots)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            restored(fresh)
        _assert_outputs(fresh, expected)
        _assert_inputs(fresh, snapshots)
        previous = expected[0]
        for seed in (44, 45):
            _fill_inputs(fresh, seed)
            snapshots = tuple(value.clone() for value in fresh.inputs())
            expected = fresh.value()
            assert not torch.equal(expected[0], previous)
            previous = expected[0]
            fresh()
            _assert_outputs(fresh, expected)
            for _ in range(2):
                for output in fresh.outputs():
                    output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert after[key] == before[key], key
                _assert_outputs(fresh, expected)
                _assert_inputs(fresh, snapshots)
    finally:
        graph.reset()
