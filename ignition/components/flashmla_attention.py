# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared reference MLA for gathered prefill and paged decode with explicit storage."""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.flashmla_kernel import prepare
from ignition.components.mega_attention import _outputs, _spec
from ignition.components.mega_attention_cache import MegaCacheLayout
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.tuning import declare, register_finalizer


def _validate(query, positions, cos_sin, sink, kv, indices, lengths,
              compressed, compressed_indices, compressed_lengths):
    tensors = (query, positions, cos_sin, sink, kv, indices, lengths,
               compressed, compressed_indices, compressed_lengths)
    if query.device.type != "cuda" or any(t.device != query.device for t in tensors if t is not None):
        raise ValueError("FlashMLA operands must share one CUDA device")
    if (query.ndim != 3 or query.shape[0] < 1 or query.shape[1:] != (64, 512)
            or query.dtype != torch.bfloat16 or not query.is_contiguous()):
        raise ValueError("FlashMLA requires contiguous BF16 queries with 64 heads and 512 dimensions")
    rows = query.shape[0]
    if (positions.shape != (rows,) or positions.dtype != torch.int32 or not positions.is_contiguous()
            or lengths.shape != (rows,) or lengths.dtype != torch.int32 or not lengths.is_contiguous()
            or sink.shape != (64,) or sink.dtype != torch.float32 or not sink.is_contiguous()
            or cos_sin.ndim != 2 or cos_sin.shape[0] < 1 or cos_sin.shape[1] != 64
            or cos_sin.dtype != torch.float32 or not cos_sin.is_contiguous()):
        raise ValueError("FlashMLA requires int32 positions/lengths, 64 FP32 sinks and contiguous FP32 rotary tables")
    if (compressed is None) != (compressed_indices is None) or (compressed is None) != (compressed_lengths is None):
        raise ValueError("compressed cache, indices and lengths must be supplied together")
    prefill = kv.dtype == torch.bfloat16
    if prefill:
        if (kv.ndim != 3 or kv.shape[0] < 1 or kv.shape[1:] != (1, 512)
                or not kv.is_contiguous() or compressed is not None):
            raise ValueError("prefill requires gathered contiguous BF16 KV with one head and 512 dimensions")
        if indices.ndim != 3 or indices.shape[:2] != (rows, 1):
            raise ValueError("prefill indices must have shape [rows, 1, keys]")
    else:
        if kv.ndim != 3:
            raise ValueError("decode requires a page-separated MXFP8 cache")
        MegaCacheLayout(kv.shape[1], 528).validate(kv)
        if indices.ndim != 2 or indices.shape[0] != rows:
            raise ValueError("decode indices must have shape [rows, keys]")
        if compressed is not None:
            if indices.shape[1] % 8:
                raise ValueError("mixed MXFP8/NVFP4 decode requires a window index capacity divisible by eight")
            if compressed.ndim != 3:
                raise ValueError("compressed decode requires a page-separated NVFP4 cache")
            MegaCacheLayout(compressed.shape[1], 288).validate(compressed)
            if (compressed_indices.ndim != 2 or compressed_indices.shape[0] != rows
                    or compressed_indices.shape[1] < 1 or compressed_indices.dtype != torch.int32
                    or not compressed_indices.is_contiguous() or compressed_lengths.shape != (rows,)
                    or compressed_lengths.dtype != torch.int32 or not compressed_lengths.is_contiguous()):
                raise ValueError("compressed indices and lengths require contiguous int32 rows")
    if indices.shape[-1] < 1 or indices.dtype != torch.int32 or not indices.is_contiguous():
        raise ValueError("FlashMLA indices require a positive key capacity and contiguous int32 storage")
    if any(max((*t.shape, *t.stride())) >= 2**31 for t in tensors if t is not None):
        raise ValueError("FlashMLA dimensions and strides must fit signed int32")
    return prefill, tensors


@dataclass(frozen=True)
class FlashMLAQuery:
    rows: int
    prefill: bool
    scale: float


CONTRACT = native_contract(component_id="flare.flashmla.fused", query_type=FlashMLAQuery,
                           tactics=lambda query: (None,))


class _State:
    def __init__(self, query, kernel_cache, library):
        self.query = query
        self.library = prepare(kernel_cache, library)

    def run(self, operands, output, scales, statistics):
        query, positions, cos_sin, sink, kv, indices, lengths, compressed, ci, cl = operands
        if self.query.prefill:
            self.library.prefill(query, kv, indices, lengths, sink, positions, cos_sin,
                                 output, scales, statistics[0], statistics[1], self.query.scale)
        else:
            self.library.decode(query, kv, indices, lengths, sink, positions, cos_sin,
                                compressed, ci, cl, output, scales, statistics[0], self.query.scale)


@torch.library.custom_op("flare::flashmla_fused", mutates_args=("output", "scales", "statistics"))
def _run(query: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
         sink: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor,
         compressed: torch.Tensor | None, compressed_indices: torch.Tensor | None,
         compressed_lengths: torch.Tensor | None, output: torch.Tensor, scales: torch.Tensor,
         statistics: list[torch.Tensor], handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, query.device).run(
        (query, positions, cos_sin, sink, kv, indices, lengths,
         compressed, compressed_indices, compressed_lengths), output, scales, statistics)


@_run.register_fake
def _run_fake(query, positions, cos_sin, sink, kv, indices, lengths, compressed,
              compressed_indices, compressed_lengths, output, scales, statistics, handle):
    pass


def _buffers(query, device):
    output, scales = _outputs(query.rows, device)
    statistics = [torch.empty((query.rows, 64), device=device, dtype=torch.float32)
                  for _ in range(2 if query.prefill else 1)]
    return output, scales, statistics


def _aligned_indices(indices):
    # The kernel vector-loads index rows at 32-byte boundaries. Retain the
    # logical key capacity and lengths while padding only the row storage.
    width = indices.shape[-1]
    padding = -width % 8
    if padding:
        return torch.nn.functional.pad(indices, (0, padding), value=-1)[..., :width]
    return indices


class PreparedFlashMLA:
    """Use the pinned fused kernel with compiler-visible outputs and statistics.

    BF16 KV selects gathered prefill; uint8 pages select paged decode. Query heads
    have the reference's 16-element permutation. Valid rotary positions and cache
    indices must be in range, and lengths must fit their index capacity. Negative
    indices mask keys. The caller owns cache insertion, gathering and index mapping.
    Index rows acquire 32-byte-aligned storage inside the compiled forward;
    padding does not add keys or change their live lengths.
    """

    def __init__(self, query, positions, cos_sin, sink, kv, indices, lengths,
                 compressed=None, compressed_indices=None, compressed_lengths=None, *,
                 scale: float, kernel_cache: Path, library: Path, session, name):
        prefill, operands = _validate(query, positions, cos_sin, sink, kv, indices, lengths,
                                      compressed, compressed_indices, compressed_lengths)
        if torch.cuda.get_device_capability(query.device) != (10, 3):
            raise ValueError("the reference FlashMLA component requires SM103")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("attention scale must be finite and positive")
        q = FlashMLAQuery(query.shape[0], prefill, scale)
        self.specs = tuple(_spec(value) for value in operands)
        device = query.device
        self.plan = native_plan(contract=CONTRACT, query=q, device=device,
            scratch_bytes=lambda tactic: 0,
            materialize=lambda tactic: _State(q, kernel_cache, library))
        examples = list(operands)
        examples[5] = _aligned_indices(indices)

        def prepare_call(state):
            output, scales, statistics = _buffers(q, device)
            return PreparedCall(run=lambda: state.run(examples, output, scales, statistics), output=output)

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)

    def __call__(self, query, positions, cos_sin, sink, kv, indices, lengths,
                 compressed=None, compressed_indices=None, compressed_lengths=None):
        operands = (query, positions, cos_sin, sink, kv, indices, lengths,
                    compressed, compressed_indices, compressed_lengths)
        if tuple(_spec(value) for value in operands) != self.specs:
            raise ValueError("FlashMLA operands must match their prepared geometry, dtype, strides and device")
        output, scales, statistics = _buffers(self.plan.query, query.device)
        _run(query, positions, cos_sin, sink, kv, _aligned_indices(indices), lengths,
             compressed, compressed_indices, compressed_lengths,
             output, scales, statistics, self.plan.handle)
        return output, scales
