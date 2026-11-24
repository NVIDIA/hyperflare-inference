# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared mm preserves ATen arithmetic and read-only input storage."""

import os

import pytest
import torch

from ignition import runtime_patches
from ignition.preparation.aot import aot_forward
from ignition.preparation.compilation import _OPTIONS


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


class MatrixMultiplyForward:
    def __init__(self, left, right, output):
        self.left, self.right, self.output = left, right, output

    def __call__(self):
        self.output.copy_(torch.mm(self.left, self.right))


def _make_forward(dtype, profile):
    if profile == "projection-transposed":
        rows, inner, columns = 120, 96, 160
        left = torch.empty((rows, inner), dtype=dtype, device="cuda")
        weight = torch.empty((columns, inner), dtype=dtype, device="cuda")
        inputs = (left, weight)
        right = weight.t()
    else:
        rows, inner, columns = 9, 31, 17
        left_storage = torch.empty((rows, inner * 2), dtype=dtype, device="cuda")
        right_storage = torch.empty((inner * 2, columns * 2), dtype=dtype, device="cuda")
        inputs = (left_storage, right_storage)
        left = left_storage[:, ::2]
        right = right_storage[::2, ::2]
    output = torch.empty((rows, columns), dtype=dtype, device="cuda")
    return MatrixMultiplyForward(left, right, output), inputs


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
    return torch.mm(forward.left, forward.right)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("profile", ["projection-transposed", "matrix-strided"])
@torch.no_grad()
def test_mm_exact_after_restore_and_allocation_free_replay(tmp_path, dtype, profile):
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
    path = tmp_path / "mm.bin"
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
