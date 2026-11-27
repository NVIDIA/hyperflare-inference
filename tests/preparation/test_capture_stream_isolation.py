# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Warmup execution must run on the dedicated warmup stream, never on the
capture stream.

cuBLAS workspaces are cached per (handle, stream); a workspace-allocating
call on the capture stream before capture can seed that stream's cache
with transient-backed memory the first graph then records. These tests
distinguish the two streams and assert every stream context that wraps
warmup work — compiled preparation, serving warmups, and before_run —
enters the warmup stream, while graph capture keeps the capture stream.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

from ignition.preparation import capture


def _capture_streams(monkeypatch):
    """Install stream mocks that record which stream each context used.

    Returns (capture_stream, warm_stream, entered, captured_on):
    entered lists the stream passed to each torch.cuda.stream() context,
    captured_on the stream= argument of each torch.cuda.graph() call.
    """
    stream = SimpleNamespace(wait_stream=lambda value: None, synchronize=lambda: None)
    warm = SimpleNamespace(wait_stream=lambda value: None, synchronize=lambda: None)
    entered, captured_on = [], []

    @contextmanager
    def record_stream(value):
        entered.append(value)
        yield

    @contextmanager
    def record_graph(graph, *, stream, **kwargs):
        captured_on.append(stream)
        yield

    monkeypatch.setattr(capture, "_warmup_stream", lambda device: warm)
    monkeypatch.setattr(capture, "_seed_warmup_workspaces", lambda stream, device: None)
    monkeypatch.setattr(capture.torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(capture.torch.cuda, "stream", record_stream)
    monkeypatch.setattr(capture.torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(capture.torch.cuda, "CUDAGraph", lambda: "graph")
    monkeypatch.setattr(capture.torch.cuda, "graph", record_graph)
    monkeypatch.setattr(capture.torch.distributed, "is_initialized", lambda: False)
    return stream, warm, entered, captured_on


class _Forward:
    cache_hit = False
    capture_allocation_sizes = ()

    def prepare_capture(self):
        return 0

    def retire_after_capture(self):
        pass

    def __call__(self):
        pass


def _session():
    return SimpleNamespace(
        namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext)


def test_compiled_preparation_and_before_run_enter_the_warmup_stream(monkeypatch):
    stream, warm, entered, captured_on = _capture_streams(monkeypatch)

    class Cache:
        def __init__(self, **kwargs):
            pass

        def prepare_many(self, functions, signatures, **kwargs):
            # The caller wraps this in a stream context; the assertion
            # below checks which one it was.
            return {name: _Forward() for name in functions}

    monkeypatch.setattr(capture, "ForwardCache", Cache)
    forward = _Forward()
    graphs, compiled = capture.capture_forwards(
        {"decode": forward}, {"decode": SimpleNamespace(kind="decode")},
        session=_session(), device="cuda:0", stream=stream, pool=None,
        before_run=lambda name: None)

    assert entered and all(value is warm for value in entered), (
        "warmup work entered the capture stream's context")
    assert captured_on and all(value is stream for value in captured_on), (
        "graph capture did not use the capture stream")


def test_serving_warmup_enters_the_warmup_stream_without_compilation(monkeypatch):
    stream, warm, entered, captured_on = _capture_streams(monkeypatch)
    forward = _Forward()
    capture.capture_forwards(
        {"decode": forward}, {"decode": SimpleNamespace(kind="decode")},
        session=_session(), device="cuda:0", stream=stream, pool=None,
        compile_decoder=False)

    assert entered and all(value is warm for value in entered), (
        "the serving warmup did not run entirely on the warmup stream")
    assert captured_on and all(value is stream for value in captured_on)
