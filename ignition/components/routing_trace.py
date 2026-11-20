"""Read-only routing counters captured beside existing model computation.

Each layer owns a histogram in the native executor's selected completion frame.
The graph resets that frame once, then counts only valid routing rows. No expert
ID, route weight, hidden state or model output is changed.
"""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import launch_triton, prepare_triton

BUFFER = "parallax_routes_v1"
HEADER = 4
_ASYNC_HANDLES = {}
_NEXT_HANDLE = 1


def _register_async(fn):
    global _NEXT_HANDLE
    handle = _NEXT_HANDLE
    _NEXT_HANDLE += 1
    _ASYNC_HANDLES[handle] = fn
    return handle


@torch.library.custom_op("flare::routing_trace_reset_async", mutates_args=("output",))
def _reset_async(output: torch.Tensor, frame: torch.Tensor, sequence: torch.Tensor,
                 stream: int, handle: int) -> None:
    main = torch.cuda.current_stream(output.device)
    auxiliary = torch.cuda.ExternalStream(stream, device=output.device) if stream else main
    if stream:
        auxiliary.wait_stream(main)
    with torch.cuda.stream(auxiliary):
        _ASYNC_HANDLES[handle](output, frame, sequence)


@_reset_async.register_fake
def _reset_async_fake(output, frame, sequence, stream, handle):
    pass


@torch.library.custom_op("flare::routing_trace_count_async", mutates_args=("output",))
def _count_async(ids: torch.Tensor, valid_tokens: torch.Tensor, output: torch.Tensor,
                 frame: torch.Tensor, stream: int, handle: int) -> None:
    main = torch.cuda.current_stream(output.device)
    auxiliary = torch.cuda.ExternalStream(stream, device=output.device) if stream else main
    if stream:
        auxiliary.wait_stream(main)
    with torch.cuda.stream(auxiliary):
        _ASYNC_HANDLES[handle](ids, valid_tokens, output, frame)


@_count_async.register_fake
def _count_async_fake(ids, valid_tokens, output, frame, stream, handle):
    pass


@torch.library.custom_op("flare::routing_trace_join", mutates_args=("output",))
def _trace_join(anchor: torch.Tensor, output: torch.Tensor, stream: int) -> None:
    if stream:
        torch.cuda.current_stream(anchor.device).wait_stream(
            torch.cuda.ExternalStream(stream, device=anchor.device))


@_trace_join.register_fake
def _trace_join_fake(anchor, output, stream):
    pass


@triton.jit
def _reset(output, frame, sequence, CELLS: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    selected = tl.load(frame)
    seq = tl.load(sequence)
    values = tl.where(indices % WIDTH == 0, seq, 0)
    tl.store(output + selected * CELLS + indices, values, mask=indices < CELLS)


@triton.jit
def _count(ids, valid_tokens, output, frame, LAYER: tl.constexpr, LAYERS: tl.constexpr,
           EXPERTS: tl.constexpr, TOKENS: tl.constexpr, TOP_K: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tokens = tl.load(valid_tokens)
    valid = (offsets < TOKENS * TOP_K) & (offsets // TOP_K < tokens)
    experts = tl.load(ids + offsets, mask=valid, other=-1)
    ok = valid & (experts >= 0) & (experts < EXPERTS)
    row = output + (tl.load(frame) * LAYERS + LAYER) * (EXPERTS + 4)
    tl.atomic_add(row + 4 + tl.where(ok, experts, 0), 1, mask=ok, sem="relaxed")
    invalid = tl.sum((valid & ~ok).to(tl.int32), axis=0)
    tl.atomic_add(row + 3, invalid.to(tl.int64), sem="relaxed")
    if tl.program_id(0) == 0:
        tl.store(row + 1, tokens)
        tl.store(row + 2, TOP_K)
        tl.atomic_add(row + 3, ((tokens < 0) | (tokens > TOKENS)).to(tl.int64), sem="relaxed")


def validate_output(output, frame, sequence):
    if (output.dtype != torch.int64 or output.ndim != 3 or output.shape[0] != 2
            or not 1 <= output.shape[1] <= 256 or not 5 <= output.shape[2] <= 4100
            or output.shape[1] * output.shape[2] > 1 << 20
            or frame.dtype != torch.int32 or sequence.dtype != torch.int64
            or frame.shape != (1,) or sequence.shape != (1,)
            or any(t.device != output.device or not t.is_contiguous() for t in (output, frame, sequence))):
        raise ValueError("routing trace requires bounded int64[2,layers,4+experts] and native frame/sequence scalars")


def validate_counter(output, layer, ids, valid_tokens):
    experts = output.shape[2] - HEADER
    if (type(layer) is not int or not 0 <= layer < output.shape[1]
            or ids.dtype not in (torch.int32, torch.int64) or ids.ndim != 2 or not 1 <= ids.shape[1] <= experts
            or not 1 <= ids.shape[0] <= 262144 or valid_tokens.dtype != torch.int32
            or valid_tokens.shape != (1,)
            or any(t.device != output.device or not t.is_contiguous() for t in (ids, valid_tokens))):
        raise ValueError("routing counter requires canonical layer, contiguous IDs and a valid-token scalar")


class RoutingTrace:
    def __init__(self, output, frame, sequence):
        validate_output(output, frame, sequence)
        self.output, self.frame, self.sequence = output, frame, sequence
        cells, width = output.shape[1] * output.shape[2], output.shape[2]
        self.reset = prepare_triton(_reset, (output, frame, sequence, cells, width, 256),
                                   (triton.cdiv(cells, 256), 1, 1))
        self.handle = _register_async(
            lambda out, frm, seq, r=self.reset: r.compiled[r.grid](out, frm, seq, *r.args[3:]))

    def counter(self, layer, ids, valid_tokens):
        experts = self.output.shape[2] - HEADER
        validate_counter(self.output, layer, ids, valid_tokens)
        return prepare_triton(_count, (ids, valid_tokens, self.output, self.frame, layer,
            self.output.shape[1], experts, ids.shape[0], ids.shape[1], 256),
            (triton.cdiv(ids.numel(), 256), 1, 1))

    def begin(self, *, stream_handle=None):
        if stream_handle is None:
            self.reset()
        else:
            _reset_async(self.output, self.frame, self.sequence, stream_handle, self.handle)

    def join(self, anchor, *, stream_handle=None):
        if stream_handle is not None:
            _trace_join(anchor, self.output, stream_handle)

    def functional_counter(self, layer, tokens, top_k, *, dtype=torch.int32):
        """Compile the counter before capture without retaining routing scratch."""
        ids = torch.empty((tokens, top_k), dtype=dtype, device=self.output.device)
        valid = torch.empty((1,), dtype=torch.int32, device=self.output.device)
        return FunctionalCounter(self.counter(layer, ids, valid))


class FunctionalCounter:
    def __init__(self, prepared):
        self.kernel, self.compiled = prepared.kernel, prepared.compiled
        self.tail, self.grid, self.options = prepared.args[2:], prepared.grid, prepared.options
        self.handle = _register_async(
            lambda ids, valid, out, frm, p=self: p.compiled[p.grid](ids, valid, out, frm, *p.tail[2:]))

    def __call__(self, ids, valid_tokens, *, stream_handle=None):
        if stream_handle is None:
            launch_triton(self.kernel, self.compiled, (ids, valid_tokens, *self.tail), self.grid, **self.options)
        else:
            _count_async(ids, valid_tokens, self.tail[0], self.tail[1], stream_handle, self.handle)
