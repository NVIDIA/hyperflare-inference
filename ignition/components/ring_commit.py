# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Commit accepted token prefixes from packed staging into multiple byte rings."""

import torch
import triton
import triton.language as tl

from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.persistent import persistent_copy


@triton.jit
def _commit(Staged, Backing, Offsets, Rows, Starts, Lengths, ROW_STRIDE: tl.constexpr,
            STAGED_STRIDE: tl.constexpr, REGION_STRIDE: tl.constexpr,
            ROWS_STRIDE: tl.constexpr, STARTS_STRIDE: tl.constexpr, LENGTHS_STRIDE: tl.constexpr,
            WINDOW: tl.constexpr, STAGED_TOKENS: tl.constexpr,
            PLANES: tl.constexpr, RECORD: tl.constexpr, BLOCK: tl.constexpr):
    item, request, region = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(Lengths + request * LENGTHS_STRIDE)
    start, row = tl.load(Starts + request * STARTS_STRIDE), tl.load(Rows + request * ROWS_STRIDE)
    token = tl.maximum(0, length - WINDOW) + item
    byte = tl.arange(0, BLOCK)
    live = (token < length) & (row > 0) & (byte < RECORD)
    source_byte = tl.full((BLOCK,), 0, tl.int32)
    target_byte = tl.full((BLOCK,), 0, tl.int32)
    prefix: tl.constexpr = 0
    for plane in tl.static_range(len(PLANES)):
        width = PLANES[plane]
        in_plane = (byte >= prefix) & (byte < prefix + width)
        source_byte = tl.where(in_plane,
            prefix * STAGED_TOKENS + (WINDOW + token) * width + byte - prefix, source_byte)
        target_byte = tl.where(in_plane,
            prefix * WINDOW + ((start + token) % WINDOW) * width + byte - prefix, target_byte)
        prefix += width
    value = tl.load(Staged + region * REGION_STRIDE + request * STAGED_STRIDE
                    + source_byte, live, other=0)
    offset = 0 if Offsets is None else tl.load(Offsets + region)
    tl.store(Backing + row * ROW_STRIDE + offset + target_byte, value, live)


class PreparedRingCommit:
    """Commit per-region staging produced inside a speculative forward.

    Admission validates distinct positive destination rows and bounds accepted
    counts by token_capacity. Row zero and unaccepted ring records are preserved.
    The complete backing tensor exposes mutations to forward compilation.
    record_planes describes per-token byte widths stored as consecutive planes
    across each ring or staging page. Omitting it keeps interleaved records.
    staging_alignment pads each staging page without changing plane offsets.
    """

    def __init__(self, backing, offsets, rows, starts, *, token_capacity, window_tokens, record_bytes,
                 staging=None, record_planes=None, staging_alignment=1):
        if (backing.device.type != "cuda" or backing.dtype != torch.uint8 or backing.ndim != 2
                or backing.stride(1) != 1 or backing.stride(0) < backing.shape[1]
                or rows.ndim != 1 or rows.numel() == 0 or starts.shape != rows.shape
                or rows.dtype != torch.int64 or starts.dtype != torch.int64
                or any(value.device != backing.device for value in (rows, starts))):
            raise ValueError("ring commits require byte rows and matching int64 row/position vectors")
        if any(type(value) is not int or value <= 0 for value in (token_capacity, window_tokens, record_bytes)):
            raise ValueError("ring and staging capacities must be positive integers")
        planes = (record_bytes,) if record_planes is None else tuple(record_planes)
        if (not planes or any(type(width) is not int or width <= 0 for width in planes)
                or sum(planes) != record_bytes):
            raise ValueError("ring plane widths must be positive integers summing to the record size")
        if (type(staging_alignment) is not int or staging_alignment <= 0
                or staging_alignment & (staging_alignment - 1)):
            raise ValueError("ring staging alignment must be a positive power of two")
        offsets = tuple(offsets)
        size = window_tokens * record_bytes
        if (not offsets or any(type(offset) is not int or offset < 0 or offset + size > backing.shape[1]
                               for offset in offsets)
                or any(first + size > second for first, second in zip(sorted(offsets), sorted(offsets)[1:]))):
            raise ValueError("ring regions must be distinct non-overlapping fields within each backing row")
        self.backing, self.rows, self.starts = backing, rows, starts
        self.offsets = persistent_copy(torch.tensor(offsets, dtype=torch.int64, device=backing.device))
        record_numeric_tensor(self.offsets)
        from ignition.preparation.scratch import bind_scratch

        staged_bytes = (window_tokens + token_capacity) * record_bytes
        staged_bytes = (staged_bytes + staging_alignment - 1) // staging_alignment * staging_alignment
        shape = (len(offsets), rows.numel(), staged_bytes)
        self.staging_shape, self.rows_shape = shape, tuple(rows.shape)
        self.staging = bind_scratch({"staging": (shape, torch.uint8)}, device=backing.device,
            workspace=None if staging is None else {"staging": staging})["staging"]
        self.staging.zero_()
        self.window_tokens, self.record_bytes = window_tokens, record_bytes
        self.record_planes, self.staged_tokens = planes, window_tokens + token_capacity
        self.grid = (min(token_capacity, window_tokens), rows.numel(), len(offsets))

    def functional(self):
        self.staging = self.rows = self.starts = None
        return self

    def __call__(self, counts, *, staging, rows, starts):
        if counts.shape != self.rows_shape or counts.dtype not in (torch.int32, torch.int64) or counts.device != self.backing.device:
            raise ValueError("accepted counts must match the ring commit's request rows")
        if (tuple(staging.shape) != self.staging_shape or staging.dtype != torch.uint8
                or staging.device != self.backing.device or not staging.is_contiguous()
                or any(value.shape != self.rows_shape or value.dtype != torch.int64
                       or value.device != self.backing.device for value in (rows, starts))):
            raise ValueError("ring commit operands must match prepared staging and request geometry")
        torch.library.wrap_triton(_commit)[self.grid](
            staging, self.backing, self.offsets, rows, starts, counts,
            self.backing.stride(0), staging.stride(1), staging.stride(0),
            rows.stride(0), starts.stride(0), counts.stride(0),
            self.window_tokens, self.staged_tokens, self.record_planes,
            self.record_bytes, triton.next_power_of_2(self.record_bytes))
