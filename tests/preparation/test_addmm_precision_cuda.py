# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared addmm preserves ATen arithmetic and read-only input storage."""

import os

import pytest
import torch

from ignition import runtime_patches
from ignition.preparation.aot import aot_forward
from ignition.preparation.compilation import _OPTIONS


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


class AddmmForward:
    def __init__(self, bias, left, right, output, scaled):
        self.bias, self.left, self.right = bias, left, right
        self.output, self.scaled = output, scaled

    def __call__(self):
        if self.scaled:
            value = torch.addmm(input=self.bias, mat1=self.left, mat2=self.right,
                                alpha=.75, beta=-1.25)
        else:
            value = torch.addmm(self.bias, self.left, self.right)
        self.output.copy_(value)


def _make_forward(dtype, profile):
    if profile == "projection-transposed":
        rows, inner, columns = 77, 96, 160
        bias = torch.empty(columns, dtype=dtype, device="cuda")
        left = torch.empty((rows, inner), dtype=dtype, device="cuda")
        weight = torch.empty((columns, inner), dtype=dtype, device="cuda")
        inputs = (bias, left, weight)
        right = weight.t()
        scaled = False
    else:
        rows, inner, columns = 9, 31, 17
        bias_storage = torch.empty((rows, columns * 2), dtype=dtype, device="cuda")
        left_storage = torch.empty((rows, inner * 2), dtype=dtype, device="cuda")
        right_storage = torch.empty((inner * 2, columns * 2), dtype=dtype, device="cuda")
        inputs = (bias_storage, left_storage, right_storage)
        bias = bias_storage[:, ::2]
        left = left_storage[:, ::2]
        right = right_storage[::2, ::2]
        scaled = True
    output = torch.empty((rows, columns), dtype=dtype, device="cuda")
    return AddmmForward(bias, left, right, output, scaled), inputs


def _fill_inputs(inputs, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for value in inputs:
        value.normal_(mean=0, std=.25, generator=generator)


def _assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def _assert_inputs(inputs, snapshots):
    # Check complete backing storage, including gaps outside strided operands.
    for value, snapshot in zip(inputs, snapshots, strict=True):
        _assert_bytes(value, snapshot)


def _reference(forward):
    if forward.scaled:
        return torch.addmm(forward.bias, forward.left, forward.right,
                           alpha=.75, beta=-1.25)
    return torch.addmm(forward.bias, forward.left, forward.right)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("profile", ["projection-transposed", "matrix-strided-scaled"])
@torch.no_grad()
def test_addmm_exact_after_restore_and_allocation_free_replay(tmp_path, dtype, profile):
    runtime_patches.apply()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("assigned GPU does not support BF16")
    assert _OPTIONS["max_autotune"] is True

    forward, inputs = _make_forward(dtype, profile)
    _fill_inputs(inputs, 42)
    snapshots = tuple(value.clone() for value in inputs)
    expected = _reference(forward)
    _assert_bytes(_reference(forward), expected)
    compiled = aot_forward(forward)
    compiled(forward)
    torch.cuda.synchronize()
    _assert_bytes(forward.output, expected)
    _assert_inputs(inputs, snapshots)
    path = tmp_path / "addmm.bin"
    compiled.save_compiled_function(str(path))

    # Restoring must bind the artifact to distinct storage with the same layout.
    fresh, fresh_inputs = _make_forward(dtype, profile)
    _fill_inputs(fresh_inputs, 43)
    fresh_snapshots = tuple(value.clone() for value in fresh_inputs)
    fresh_expected = _reference(fresh)
    with path.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    restored(fresh)
    torch.cuda.synchronize()
    _assert_bytes(fresh.output, fresh_expected)
    _assert_inputs(fresh_inputs, fresh_snapshots)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        restored(fresh)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            restored(fresh)
        torch.cuda.synchronize()
        _assert_bytes(fresh.output, fresh_expected)
        _assert_inputs(fresh_inputs, fresh_snapshots)
        for seed in (44, 45):
            _fill_inputs(fresh_inputs, seed)
            snapshots = tuple(value.clone() for value in fresh_inputs)
            expected = _reference(fresh)
            _assert_bytes(_reference(fresh), expected)
            assert not torch.equal(expected, fresh_expected)
            for _ in range(2):
                fresh.output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert after[key] == before[key], key
                _assert_bytes(fresh.output, expected)
                _assert_inputs(fresh_inputs, snapshots)
    finally:
        graph.reset()
