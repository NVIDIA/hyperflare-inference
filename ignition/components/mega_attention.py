# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared SM103 fused decode MLA with explicit inter-launch synchronization state."""

from dataclasses import dataclass
import math
from pathlib import Path
import struct

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.mega_attention_kernel import prepare
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.persistent import persistent_zeros
from ignition.preparation.tuning import declare, register_finalizer


class MegaAttentionWorkspace:
    """Epoch-tagged state shared by serialized attention calls on one device.

    The parallel kernel uses tags to distinguish launches without clearing its
    partial results. Preserve the tags and epoch together across graph variants;
    this storage cannot alias a different operator's temporary workspace.
    """

    def __init__(self, capacity: int, *, device):
        if type(capacity) is not int or not 1 <= capacity <= 76:
            raise ValueError("fused decode attention requires a capacity from 1 through 76")
        device = torch.device(device)
        with torch.cuda.device(device):
            if torch.cuda.get_device_capability() != (10, 3):
                raise ValueError("fused decode attention requires SM103")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("attention state must be allocated before capture")
            # Only the parallel mode (up to 16 rows) uses tagged partials.
            words = min(capacity, 16) * 10 * (128 + 128 + 2048 + 1)
            self.tensors = [
                persistent_zeros((2 * words,), device=device, dtype=torch.float32),
                persistent_zeros((1,), device=device, dtype=torch.float32),
                persistent_zeros((1,), device=device, dtype=torch.int32),
                persistent_zeros((2,), device=device, dtype=torch.int32),
            ]
        self.capacity = capacity
        self.device = self.tensors[0].device


def _spec(tensor):
    return None if tensor is None else (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def _validate_operands(query, positions, cos_sin, sink, window, indices, lengths,
                       compressed, compressed_indices, compressed_lengths, workspace):
    rows = query.shape[0]
    operands = (query, positions, cos_sin, sink, window, indices, lengths,
                compressed, compressed_indices, compressed_lengths)
    if any(value.device != workspace.device for value in operands if value is not None):
        raise ValueError("attention operands and synchronization state must share one CUDA device")
    if (query.shape != (rows, 64, 512) or query.dtype != torch.bfloat16
            or not query.is_contiguous() or not 1 <= rows <= workspace.capacity):
        raise ValueError("attention requires contiguous BF16 queries with 64 heads and 512 dimensions")
    if (positions.shape != (rows,) or positions.dtype != torch.int32 or not positions.is_contiguous()
            or cos_sin.ndim != 2 or cos_sin.shape[0] < 1 or cos_sin.shape[1] != 64
            or cos_sin.dtype != torch.float32 or not cos_sin.is_contiguous()
            or sink.shape != (64,) or sink.dtype != torch.float32 or not sink.is_contiguous()):
        raise ValueError("attention requires int32 positions, contiguous FP32 rotary tables and 64 sinks")
    if (compressed is None) != (compressed_indices is None) or (compressed is None) != (compressed_lengths is None):
        raise ValueError("compressed cache, indices and lengths must be supplied together")
    for cache, ids, lens, width in ((window, indices, lengths, 528),
                                   (compressed, compressed_indices, compressed_lengths, 288)):
        if cache is None:
            continue
        if (cache.ndim != 3 or min(cache.shape) <= 0 or cache.shape[2] != width
                or cache.dtype != torch.uint8 or cache.stride()[1:] != (width, 1)
                or cache.stride(0) < cache.shape[1] * width or cache.stride(0) % 512):
            raise ValueError("attention cache requires page-separated payload/scales and a 512-byte aligned page stride")
        if (ids.ndim != 2 or ids.shape[0] != rows or ids.shape[1] < 1 or ids.dtype != torch.int32
                or not ids.is_contiguous() or lens.shape != (rows,) or lens.dtype != torch.int32
                or not lens.is_contiguous()):
            raise ValueError("attention indices and lengths must be contiguous int32 with the query row count")
    slots = indices.shape[1] + (compressed_indices.shape[1] if compressed is not None else 0)
    if slots > 640:
        raise ValueError("fused decode attention supports at most ten 64-key blocks")
    if compressed is None:
        if rows > 24:
            raise ValueError("window-only reference attention supports at most 24 rows")
        mode = -2
    else:
        mode = -1 if rows <= 16 else -3
        if mode == -3 and (indices.shape[1] > 128 or any(
                cache.shape[1] & (cache.shape[1] - 1) for cache in (window, compressed))):
            raise ValueError("cluster attention requires a window of at most 128 slots and power-of-two pages")
    return operands, mode


@dataclass(frozen=True)
class MegaAttentionQuery:
    rows: int
    mode: int
    nsub: int
    scale_log2: float


CONTRACT = native_contract(component_id="flare.mega_attention.decode",
                           query_type=MegaAttentionQuery, tactics=lambda query: (None,))


class _State:
    def __init__(self, query, kernel_cache):
        self.query = query
        self.library = prepare(kernel_cache)

    def run(self, operands, output, scales, state):
        query, positions, cos_sin, sink, window, indices, lengths, compressed, ci, cl = operands
        q = self.query
        self.library.w1_attn_decode(query, positions, cos_sin, sink, q.scale_log2,
            window, indices, lengths, compressed, ci, cl, output, scales, *state,
            None, q.mode, q.nsub, True, 0, None)


@torch.library.custom_op("flare::mega_attention", mutates_args=("output", "scales", "state"))
def _run(query: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
         sink: torch.Tensor, window: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor,
         compressed: torch.Tensor | None, compressed_indices: torch.Tensor | None,
         compressed_lengths: torch.Tensor | None, output: torch.Tensor, scales: torch.Tensor,
         state: list[torch.Tensor], handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, query.device).run(
        (query, positions, cos_sin, sink, window, indices, lengths,
         compressed, compressed_indices, compressed_lengths), output, scales, state)


@_run.register_fake
def _run_fake(query, positions, cos_sin, sink, window, indices, lengths, compressed,
              compressed_indices, compressed_lengths, output, scales, state, handle):
    pass


def _outputs(rows, device):
    output = torch.empty((rows, 8, 4096), device=device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((8, 32, (rows + 3) // 4 * 4), device=device, dtype=torch.int32)
    return output, scales.permute(2, 0, 1)[:rows]


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class PreparedMegaAttention:
    """Declare the reference decode modes; cache contents use the FlashMLA page format.

    Query heads must already have the reference's 16-element permutation. The
    caller supplies in-range rotary positions and cache indices, with lengths
    bounded by each index width. This component does not write or repack caches.
    """

    def __init__(self, query, positions, cos_sin, sink, window, indices, lengths,
                 compressed=None, compressed_indices=None, compressed_lengths=None, *,
                 workspace: MegaAttentionWorkspace, scale: float, kernel_cache: Path, session, name):
        operands, mode = _validate_operands(query, positions, cos_sin, sink, window, indices,
            lengths, compressed, compressed_indices, compressed_lengths, workspace)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("attention scale must be finite and positive")
        rows = query.shape[0]
        sms = torch.cuda.get_device_properties(workspace.device).multi_processor_count
        nsub = (1 if 8 * rows <= sms else (2 if 4 * rows <= sms else 4)) if mode == -2 else 0
        q = MegaAttentionQuery(rows, mode, nsub, _f32(_f32(scale) * _f32(math.log2(math.e))))
        self.specs = tuple(_spec(value) for value in operands)
        self.workspace = workspace
        self.plan = native_plan(contract=CONTRACT, query=q, device=workspace.device,
            scratch_bytes=lambda tactic: 0, materialize=lambda tactic: _State(q, kernel_cache))
        examples = list(operands)

        def prepare_call(state):
            output, scales = _outputs(rows, workspace.device)
            return PreparedCall(run=lambda: state.run(examples, output, scales, workspace.tensors),
                                output=output, owners=(workspace,))

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)

    def __call__(self, query, positions, cos_sin, sink, window, indices, lengths,
                 compressed=None, compressed_indices=None, compressed_lengths=None):
        operands = (query, positions, cos_sin, sink, window, indices, lengths,
                    compressed, compressed_indices, compressed_lengths)
        if tuple(_spec(value) for value in operands) != self.specs:
            raise ValueError("attention operands must match their prepared geometry, dtype, strides and device")
        output, scales = _outputs(self.plan.query.rows, query.device)
        _run(*operands, output, scales, self.workspace.tensors, self.plan.handle)
        return output, scales
