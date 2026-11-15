# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Opaque consumers reuse computed operands and preserve strided mutations through AOT capture."""

import os

import pytest
import torch
import triton
import triton.language as tl

from ignition.preparation.aot import BoundForward, aot_forward
from ignition.preparation.compilation import _OPTIONS

pytestmark = pytest.mark.skipif(os.getenv('FLARE_RUN_CUDA_TESTS') != '1', reason='requires assigned CUDA')
_ADDRESSES = []


@triton.jit
def _copy(X, Y, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i * STRIDE, i < N, other=0), i < N)


@triton.jit
def _increment(X, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i * STRIDE, i < N, other=0).to(tl.float32)
    tl.store(X + i * STRIDE, value + .5, i < N)


@torch.library.custom_op('test_opaque_operand::consume', mutates_args=('output',))
def _consume(source: torch.Tensor, output: torch.Tensor) -> None:
    _ADDRESSES.append((source.data_ptr(), source.stride()))
    _copy[(triton.cdiv(source.numel(), 256),)](source, output, source.numel(), source.stride(0), 256)


@_consume.register_fake
def _fake_consume(source, output):
    pass


@torch.library.custom_op('test_opaque_operand::increment', mutates_args=('source',))
def _increment_source(source: torch.Tensor) -> None:
    _increment[(triton.cdiv(source.numel(), 256),)](source, source.numel(), source.stride(0), 256)


@_increment_source.register_fake
def _fake_increment_source(source):
    pass


class Forward:
    def __init__(self, mode):
        self.mode = mode
        self.source = torch.full(((2 if mode == 'strided' else 1) << 18,), .25,
                                 dtype=torch.float32, device='cuda')
        self.output = torch.empty((16, 1 << 18), dtype=torch.bfloat16, device='cuda')

    def __call__(self):
        shared = (self.source * 2 + .125).bfloat16()
        if self.mode == 'strided':
            shared = shared[1::2]
        for index in range(16):
            if self.mode == 'mutation' and index == 8:
                _increment_source(shared[::2])
            _consume(shared, self.output[index])


@torch.no_grad()
@pytest.mark.parametrize('mode', ['shared', 'strided', 'mutation'])
def test_aot_opaque_consumers_reuse_one_materialization(tmp_path, mode):
    options = dict(_OPTIONS)
    assert options['emulate_precision_casts'] and options['memory_pool'] == 'combined'
    forward = Forward(mode)
    artifact = aot_forward(forward, options=options)
    path = tmp_path / 'forward.bin'
    artifact.save_compiled_function(str(path))
    with path.open('rb') as file:
        artifact = torch.compiler.load_compiled_function(file)
    _ADDRESSES.clear()
    artifact(forward)
    torch.cuda.synchronize()
    addresses = tuple(_ADDRESSES)
    sizes = BoundForward(artifact, forward, False).capture_allocation_sizes
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            artifact(forward)
        for value in (.125, -.25, .5):
            forward.source.fill_(value)
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in ('allocation.all.allocated', 'allocation.all.freed', 'segment.all.allocated', 'segment.all.freed'):
                assert before[name] == after[name]
            expected = (forward.source * 2 + .125).bfloat16()
            if mode == 'strided':
                expected = expected[1::2]
            for index, row in enumerate(forward.output):
                if mode == 'mutation' and index == 8:
                    expected[::2] = (expected[::2].float() + .5).bfloat16()
                torch.testing.assert_close(row, expected, rtol=0, atol=0)
    finally:
        graph.reset()
    assert len(addresses) == 16
    assert all(stride == ((2,) if mode == 'strided' else (1,)) for _, stride in addresses)
    if mode == 'shared':
        result = dict(unique_addresses=len({address for address, _ in addresses}), pool_bytes=sizes)
        assert result['unique_addresses'] == 1, result
        assert sum(sizes) <= 2 * forward.output[0].nbytes, result


class PendingForward:
    def __init__(self):
        self.source = torch.full((1 << 18,), .25, dtype=torch.float32, device='cuda')
        self.zero = torch.zeros(1, device='cuda')
        self.output = torch.empty((3, 1 << 18), device='cuda')
        self.stream = torch.cuda.Stream()
        self.stream_handle = self.stream.cuda_stream

    def __call__(self):
        from tests.preparation import test_graph_local_scratch_cuda as serial
        from tests.preparation.test_graph_local_overlap_cuda import _start
        from ignition.components.projection_overlap import _join

        source = self.source * 2 + .125
        auxiliary, auxiliary_scratch = torch.empty_like(source), torch.empty_like(source)
        _start(source, self.zero, auxiliary, auxiliary_scratch, 2, self.stream_handle)
        main, main_scratch = torch.empty_like(source), torch.empty_like(source)
        serial._prepared_out(self.source, self.zero, main, main_scratch, 4)
        _join(source, main, [auxiliary], auxiliary_scratch, self.stream_handle)
        self.output[0].copy_(auxiliary)
        self.output[1].copy_(main)
        tail, tail_scratch = torch.empty_like(source), torch.empty_like(source)
        serial._prepared_out(self.source, self.output[0, :1], tail, tail_scratch, 8)
        self.output[2].copy_(tail)


@torch.no_grad()
def test_computed_opaque_operand_survives_auxiliary_stream_join(tmp_path):
    from tests.preparation import test_graph_local_overlap_cuda as overlap

    overlap._ASYNC_ADDRESSES.clear()
    forward = PendingForward()
    artifact = aot_forward(forward, options=dict(_OPTIONS))
    path = tmp_path / 'pending.bin'
    artifact.save_compiled_function(str(path))
    with path.open('rb') as file:
        artifact = torch.compiler.load_compiled_function(file)
    artifact(forward)
    torch.cuda.synchronize()
    assert overlap._ASYNC_ADDRESSES
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            artifact(forward)
        for value in (.125, -.25, .5):
            forward.source.fill_(value)
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in ('allocation.all.allocated', 'allocation.all.freed', 'segment.all.allocated', 'segment.all.freed'):
                assert before[name] == after[name]
            for row, expected in zip(forward.output, (4 * value + .25, 4 * value, 40 * value + 2), strict=True):
                torch.testing.assert_close(row, torch.full_like(row, expected), rtol=0, atol=0)
    finally:
        graph.reset()
