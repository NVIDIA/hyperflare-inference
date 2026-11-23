# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Chunk sizing for DMA all-reduce channels and side-stream reduction helpers."""

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def _fake_dma_runtime(max_bytes, *, align_rows=1):
    from b12x.comm import pcie

    class FakeDmaRuntime(pcie.DmaAllReduce):
        def __init__(self):
            self.max_bytes = max_bytes
            self.world_size = 4
            self.group = object()
            self.exchange_group = object()
            self.single_channel = False

        def should_allreduce(self, tensor):
            return tensor.nbytes <= self.max_bytes and tensor.shape[0] % align_rows == 0

        def plan(self, source, *, out):
            return SimpleNamespace(
                query=SimpleNamespace(surface="DmaAllReduce.all_reduce"),
                request=lambda **kwargs: object(),
            )

    return FakeDmaRuntime()


def test_chunked_dma_all_reduce_uses_exact_channel_capacity(monkeypatch):
    from b12x.comm import pcie
    from b12x.comm.pcie import _dma_preparation
    from ignition.components import collectives

    plan = SimpleNamespace(query=SimpleNamespace(surface="DmaAllReduce.all_reduce"),
                           request=lambda **kwargs: object())
    monkeypatch.setattr(_dma_preparation, "query_from_metadata", lambda *_args, **_kwargs: "dma-query")
    monkeypatch.setattr(pcie, "plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])

    # 40,960 BF16 rows x 5,120 columns = 400 MiB across an 80 MiB (8,192-row) DMA channel.
    rows, hidden = 40960, 5120
    channel_rows = 8192
    runtime = _fake_dma_runtime(channel_rows * hidden * 2)
    source = torch.empty((rows, hidden), dtype=torch.bfloat16, device="meta")
    output = torch.empty_like(source)
    chunked = collectives.PreparedChunkedAllReduce(
        source, output, runtime=runtime, session=object(), coordinator=object(), name="engram.reduce"
    )
    assert chunked.bounds == [
        (0, 8192), (8192, 16384), (16384, 24576), (24576, 32768), (32768, 40960)
    ]
    assert len(chunked.parts) == 5


def test_chunked_dma_all_reduce_respects_should_allreduce_alignment(monkeypatch):
    from b12x.comm import pcie
    from b12x.comm.pcie import _dma_preparation
    from ignition.components import collectives

    plan = SimpleNamespace(query=SimpleNamespace(surface="DmaAllReduce.all_reduce"),
                           request=lambda **kwargs: object())
    monkeypatch.setattr(_dma_preparation, "query_from_metadata", lambda *_args, **_kwargs: "dma-query")
    monkeypatch.setattr(pcie, "plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])

    runtime = _fake_dma_runtime(200 * 64 * 2, align_rows=64)
    source = torch.empty((512, 64), dtype=torch.bfloat16, device="meta")
    output = torch.empty_like(source)
    chunked = collectives.PreparedChunkedAllReduce(
        source, output, runtime=runtime, session=object(), coordinator=object(), name="aligned.reduce"
    )
    # 200 rows fit max_bytes, then decrementing to align_rows=64 selects 192-row chunks.
    assert chunked.bounds == [(0, 192), (192, 384), (384, 512)]


def test_chunked_dma_all_reduce_rejects_oversized_single_row(monkeypatch):
    from ignition.components import collectives

    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    runtime = _fake_dma_runtime(1024)
    source = torch.empty((4, 1024), dtype=torch.bfloat16, device="meta")
    output = torch.empty_like(source)
    with pytest.raises(ValueError, match="DMA channel capacity"):
        collectives.PreparedChunkedAllReduce(
            source, output, runtime=runtime, session=object(), coordinator=object(), name="too_wide"
        )


def test_chunked_non_dma_all_reduce_preserves_halving(monkeypatch):
    from ignition.components import collectives

    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])

    plan = SimpleNamespace(query=SimpleNamespace(surface="OneshotAllReduce.all_reduce"),
                           request=lambda **kwargs: object())
    runtime = SimpleNamespace(
        algorithm="oneshot",
        world_size=4,
        single_channel=True,
        exchange_group=object(),
        for_stream=lambda: SimpleNamespace(should_allreduce=lambda t: t.shape[0] <= 128),
        plan=lambda *_args, **_kwargs: plan,
    )
    source = torch.empty((512, 64), dtype=torch.bfloat16, device="meta")
    output = torch.empty_like(source)
    chunked = collectives.PreparedChunkedAllReduce(
        source, output, runtime=runtime, session=object(), coordinator=object(), name="oneshot.reduce"
    )
    assert chunked.bounds == [(0, 128), (128, 256), (256, 384), (384, 512)]


@pytest.mark.parametrize("functional", [False, True])
def test_side_stream_single_rank_copies_without_cuda_stream(functional):
    from ignition.components.collectives import PreparedAllReduce

    source = torch.arange(24, dtype=torch.bfloat16).view(4, 6)
    output = torch.zeros_like(source)
    reduce = PreparedAllReduce(source, output, runtime=None, session=object(), coordinator=None, name="single")
    if functional:
        reduce.functional(session=object())
    pending = reduce.start_side(source, side=SimpleNamespace(stream_handle=1234))
    assert pending.side is None
    result = reduce.join_side(pending)
    assert torch.equal(result, source)
    if functional:
        assert result is not output
    else:
        assert result is output


def test_side_stream_custom_ops_support_fake_tensor_tracing():
    from ignition.components.collectives import _all_reduce_side, _join_side_stream

    with FakeTensorMode():
        src = torch.empty((16, 64), dtype=torch.bfloat16)
        out = torch.empty((16, 64), dtype=torch.bfloat16)
        dep = torch.empty((16, 64), dtype=torch.bfloat16)
        _all_reduce_side(src, out, [dep], 1, 100, 200)
        _join_side_stream(src, out, [dep], 100)
