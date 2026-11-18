# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Attention views and hidden residual layouts preserve convolution replay."""

import os

import pytest
import torch
import torch.nn.functional as F

from ignition import runtime_patches
from ignition.preparation.aot import aot_forward


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


class AttentionLayoutForward:
    def __init__(self, source, branch, weight, bias):
        self.source, self.branch, self.weight, self.bias = source, branch, weight, bias

    def __call__(self):
        source = self.source.transpose(1, 2).reshape(self.branch.shape)
        value = (source + self.branch) / 1.0
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        return F.conv2d(value, self.weight, self.bias, padding=1)


def _make_forward(dtype):
    source_storage = torch.empty(8 * 5 * 7, dtype=dtype, device="cuda")
    source = source_storage.view(1, 5 * 7, 8)
    branch = torch.empty((1, 8, 5, 7), dtype=dtype, device="cuda")
    weight = torch.empty((12, 8, 3, 3), dtype=dtype, device="cuda")
    bias = torch.empty(12, dtype=dtype, device="cuda")
    return AttentionLayoutForward(source, branch, weight, bias), (source_storage, branch, weight, bias)


def _fill(inputs, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for index, value in enumerate(inputs):
        value.normal_(std=.01 if index == 2 else .25, generator=generator)


def _assert_bytes(actual, expected, *, layout=False):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if layout:
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _assert_inputs(inputs, snapshots):
    for value, snapshot in zip(inputs, snapshots, strict=True):
        _assert_bytes(value, snapshot)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.no_grad()
def test_attention_and_scalar_layout_survive_restore_and_replay(tmp_path, dtype):
    runtime_patches.apply()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("assigned GPU does not support BF16")

    forward, inputs = _make_forward(dtype)
    _fill(inputs, 42)
    snapshots = tuple(value.clone() for value in inputs)
    expected = forward()
    assert torch.isfinite(expected).all()
    _assert_bytes(forward(), expected, layout=True)
    compiled = aot_forward(forward)
    actual = compiled(forward)
    torch.cuda.synchronize()
    _assert_bytes(actual, expected, layout=True)
    _assert_inputs(inputs, snapshots)
    artifact = tmp_path / "mixed-layout.bin"
    compiled.save_compiled_function(str(artifact))

    # Original owners remain live so rebinding cannot reuse their storage.
    fresh, fresh_inputs = _make_forward(dtype)
    for original, replacement in zip(inputs, fresh_inputs, strict=True):
        assert original.untyped_storage().data_ptr() != replacement.untyped_storage().data_ptr()
    _fill(fresh_inputs, 43)
    fresh_snapshots = tuple(value.clone() for value in fresh_inputs)
    fresh_expected = fresh()
    _assert_bytes(fresh(), fresh_expected, layout=True)
    with artifact.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    rebound = restored(fresh)
    torch.cuda.synchronize()
    _assert_bytes(rebound, fresh_expected, layout=True)
    _assert_inputs(inputs, snapshots)
    _assert_inputs(fresh_inputs, fresh_snapshots)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warm_output = restored(fresh)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    _assert_bytes(warm_output, fresh_expected, layout=True)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            captured_output = restored(fresh)
        graph.replay()
        torch.cuda.synchronize()
        _assert_bytes(captured_output, fresh_expected, layout=True)
        for seed in (44, 45):
            _fill(fresh_inputs, seed)
            saved = tuple(value.clone() for value in fresh_inputs)
            reference = fresh()
            assert torch.isfinite(reference).all()
            _assert_bytes(fresh(), reference, layout=True)
            assert not torch.equal(reference, fresh_expected)
            for _ in range(2):
                captured_output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert after[key] == before[key], key
                _assert_bytes(captured_output, reference, layout=True)
                _assert_inputs(fresh_inputs, saved)
                _assert_inputs(inputs, snapshots)
    finally:
        graph.reset()
