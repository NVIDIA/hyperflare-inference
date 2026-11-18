# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Scalar cubes retain eager low-precision rounding through CUDA graph replay."""

import os

import pytest
import torch

from ignition import runtime_patches
from ignition.preparation.aot import aot_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


class ScalarCubeForward:
    def __init__(self, dtype, exponent):
        self.exponent = exponent
        self.base = torch.empty((256, 512), dtype=dtype, device="cuda")
        self.source = self.base[:, ::2]
        self.output = torch.empty((256, 256), dtype=dtype, device="cuda")

    def value(self):
        return torch.pow(self.source, self.exponent)

    def __call__(self):
        self.output.copy_(self.value())


class NewGELUForward:
    def __init__(self, dtype):
        from transformers.activations import NewGELUActivation

        self.activation = NewGELUActivation().eval()
        self.base = torch.empty((4, 8192), dtype=dtype, device="cuda")
        self.source = self.base[:, ::2]
        self.output = torch.empty((4, 4096), dtype=dtype, device="cuda")

    def value(self):
        return self.activation(self.source)

    def __call__(self):
        self.output.copy_(self.value())


def _assert_bytes_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _finite_bit_patterns(dtype):
    bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16)
    values = bits.view(dtype)
    finite = torch.isfinite(values)
    assert int(finite.sum()) == {torch.float16: 63488, torch.bfloat16: 65280}[dtype]
    # Replace nonfinite encodings while retaining every finite encoding and both signed zeros.
    return torch.where(finite, values, torch.zeros_like(values)).reshape(256, 256)


def _check_serialized_replays(tmp_path, forward, inputs, *, expect_overflow):
    assert not forward.source.is_contiguous()
    forward.base.fill_(-0.375)
    forward.source.copy_(inputs[0])
    original = forward.base.clone()
    expected = forward.value()
    compiled = aot_forward(forward)
    compiled(forward)
    _assert_bytes_equal(forward.output, expected)
    _assert_bytes_equal(forward.base, original)
    artifact = tmp_path / "forward.bin"
    compiled.save_compiled_function(str(artifact))
    with artifact.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    forward.output.fill_(float("nan"))
    restored(forward)
    _assert_bytes_equal(forward.output, expected)
    _assert_bytes_equal(forward.base, original)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            restored(forward)
        for source in inputs:
            forward.base.fill_(-0.375)
            forward.source.copy_(source)
            original = forward.base.clone()
            assert torch.isfinite(forward.source).all()
            expected = forward.value()
            if expect_overflow:
                assert torch.isinf(expected).any()
                assert not torch.isnan(expected).any()
            else:
                assert torch.isfinite(expected).all()
            forward()
            _assert_bytes_equal(forward.output, expected)
            for _ in range(2):
                forward.output.fill_(float("nan"))
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert after[key] == before[key]
                _assert_bytes_equal(forward.output, expected)
                _assert_bytes_equal(forward.base, original)
    finally:
        graph.reset()


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("exponent", [pytest.param(3, id="integer"), pytest.param(3.0, id="float")])
def test_scalar_cube_all_finite_encodings_survive_graph_replay(tmp_path, dtype, exponent):
    runtime_patches.apply()
    source = _finite_bit_patterns(dtype)
    permuted = source.flatten().flip(0).reshape_as(source)
    assert not torch.equal(source.view(torch.uint8), permuted.view(torch.uint8))
    _check_serialized_replays(tmp_path, ScalarCubeForward(dtype, exponent),
                             (source, permuted), expect_overflow=True)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_new_gelu_scalar_cube_rounding_survives_graph_replay(tmp_path, dtype):
    runtime_patches.apply()
    sources = tuple(torch.randn((4, 4096), generator=torch.Generator().manual_seed(seed)) * 3
                    for seed in (42, 43))
    _check_serialized_replays(tmp_path, NewGELUForward(dtype), sources, expect_overflow=False)
