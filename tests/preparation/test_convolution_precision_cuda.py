# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared convolution preserves native arithmetic and read-only storage."""

import os

import pytest
import torch
import torch.nn.functional as F

from ignition import runtime_patches
from ignition.preparation.aot import aot_forward
from ignition.preparation.compilation import _OPTIONS


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


class ConvolutionForward:
    def __init__(self, source, weight, bias, output, *, stride, padding,
                 dilation, transposed, output_padding, groups):
        self.source, self.weight, self.bias, self.output = source, weight, bias, output
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.transposed, self.output_padding, self.groups = transposed, output_padding, groups

    def __call__(self):
        if self.transposed:
            result = F.conv_transpose2d(self.source, self.weight, self.bias,
                                       self.stride, self.padding, self.output_padding,
                                       self.groups, self.dilation)
        else:
            result = F.conv2d(self.source, self.weight, self.bias,
                             self.stride, self.padding, self.dilation, self.groups)
        self.output.copy_(result)


def _make_forward(dtype, profile):
    stride, padding, dilation = (1, 1), (0, 0), (1, 1)
    transposed, output_padding, groups = False, (0, 0), 1
    memory_format = torch.contiguous_format
    if profile == "pointwise-biased":
        source_shape, weight_shape = (2, 4, 9, 11), (7, 4, 1, 1)
        output_shape, bias_shape = (2, 7, 9, 11), (7,)
    elif profile == "spatial-biased-channels-last":
        source_shape, weight_shape = (2, 8, 11, 13), (12, 8, 3, 3)
        output_shape, bias_shape = (2, 12, 11, 13), (12,)
        padding, memory_format = (1, 1), torch.channels_last
    elif profile == "grouped-dilated-strided":
        source_shape, weight_shape = (2, 8, 13, 30), (12, 4, 3, 3)
        output_shape, bias_shape = (2, 12, 7, 15), None
        stride, padding, dilation, groups = (2, 1), (2, 1), (2, 1), 2
    elif profile == "transposed-output-padding":
        source_shape, weight_shape = (2, 6, 7, 9), (6, 4, 3, 3)
        output_shape, bias_shape = (2, 4, 14, 17), (4,)
        stride, padding, output_padding, transposed = (2, 2), (1, 1), (1, 0), True
    else:
        raise ValueError(f"unknown convolution test profile: {profile}")

    source_storage = torch.empty(source_shape, dtype=dtype, device="cuda",
                                 memory_format=memory_format)
    weight = torch.empty(weight_shape, dtype=dtype, device="cuda",
                         memory_format=memory_format)
    bias = None if bias_shape is None else torch.empty(bias_shape, dtype=dtype, device="cuda")
    source = source_storage[..., ::2] if profile == "grouped-dilated-strided" else source_storage
    output = torch.empty(output_shape, dtype=dtype, device="cuda", memory_format=memory_format)
    inputs = (source_storage, weight) if bias is None else (source_storage, weight, bias)
    return ConvolutionForward(source, weight, bias, output, stride=stride, padding=padding,
                              dilation=dilation, transposed=transposed,
                              output_padding=output_padding, groups=groups), inputs


def _fill_inputs(inputs, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for value in inputs:
        value.normal_(mean=0, std=.25, generator=generator)


def _assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def _assert_inputs(inputs, snapshots):
    # Include backing-storage gaps that the strided source does not read.
    for value, snapshot in zip(inputs, snapshots, strict=True):
        _assert_bytes(value, snapshot)


def _reference(forward):
    if forward.transposed:
        return F.conv_transpose2d(forward.source, forward.weight, forward.bias,
                                  forward.stride, forward.padding, forward.output_padding,
                                  forward.groups, forward.dilation)
    return F.conv2d(forward.source, forward.weight, forward.bias,
                    forward.stride, forward.padding, forward.dilation, forward.groups)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("profile", ["pointwise-biased", "spatial-biased-channels-last",
                                    "grouped-dilated-strided", "transposed-output-padding"])
@torch.no_grad()
def test_convolution_exact_after_restore_and_allocation_free_replay(tmp_path, dtype, profile):
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
    assert torch.isfinite(expected).all()
    _assert_bytes(_reference(forward), expected)
    compiled = aot_forward(forward)
    compiled(forward)
    torch.cuda.synchronize()
    _assert_bytes(forward.output, expected)
    _assert_inputs(inputs, snapshots)
    path = tmp_path / "convolution.bin"
    compiled.save_compiled_function(str(path))

    # Live original buffers make distinct-storage rebinding observable.
    fresh, fresh_inputs = _make_forward(dtype, profile)
    for original, replacement in zip(inputs, fresh_inputs, strict=True):
        assert original.untyped_storage().data_ptr() != replacement.untyped_storage().data_ptr()
    assert forward.output.untyped_storage().data_ptr() != fresh.output.untyped_storage().data_ptr()
    _fill_inputs(fresh_inputs, 43)
    fresh_snapshots = tuple(value.clone() for value in fresh_inputs)
    fresh_expected = _reference(fresh)
    _assert_bytes(_reference(fresh), fresh_expected)
    with path.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    restored(fresh)
    torch.cuda.synchronize()
    _assert_bytes(fresh.output, fresh_expected)
    _assert_inputs(fresh_inputs, fresh_snapshots)
    _assert_inputs(inputs, snapshots)

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
            fresh_snapshots = tuple(value.clone() for value in fresh_inputs)
            expected = _reference(fresh)
            assert torch.isfinite(expected).all()
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
                _assert_inputs(fresh_inputs, fresh_snapshots)
                _assert_inputs(inputs, snapshots)
    finally:
        graph.reset()
