# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared writers for page-separated MXFP8 and NVFP4 attention caches."""

from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.mega_cache_kernel import prepare
from ignition.components.mega_cache_kernel.kernels import _rope_quant_insert_nvfp4_kernel
from ignition.components.native_operator import native_contract, native_plan
from ignition.components.ring_commit import _commit
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class MegaCacheLayout:
    """Each page contains all payload rows, then all scale rows, then alignment padding."""

    page_tokens: int
    record_bytes: int

    def __post_init__(self):
        if (type(self.page_tokens) is not int or self.page_tokens <= 0
                or type(self.record_bytes) is not int or self.record_bytes not in (528, 288)):
            raise ValueError("attention cache pages require positive token capacity and 528- or 288-byte records")

    @property
    def payload_bytes(self):
        return 512 if self.record_bytes == 528 else 256

    @property
    def scale_bytes(self):
        return self.record_bytes - self.payload_bytes

    @property
    def page_bytes(self):
        return (self.page_tokens * self.record_bytes + 511) // 512 * 512

    def view(self, storage):
        if (storage.ndim != 2 or storage.dtype != torch.uint8 or storage.stride(1) != 1
                or storage.shape[1] != self.page_bytes or storage.stride(0) < self.page_bytes
                or storage.stride(0) % 512):
            raise ValueError("cache backing must have one aligned byte row per page")
        return storage[:, :self.page_tokens * self.record_bytes].view(-1, self.page_tokens, self.record_bytes)

    def validate(self, cache):
        if (cache.ndim != 3 or cache.shape[0] <= 0
                or cache.shape[1:] != (self.page_tokens, self.record_bytes)
                or cache.dtype != torch.uint8 or cache.stride()[1:] != (self.record_bytes, 1)
                or cache.stride(0) < self.page_bytes or cache.stride(0) % 512):
            raise ValueError("attention cache must preserve its prepared page and scale layout")


@dataclass(frozen=True)
class CacheWriteQuery:
    rows: int
    page_tokens: int
    record_bytes: int
    page_stride: int
    ratio: int
    rotary_rows: int


def _validate(values, positions, cos_sin, cache, slots, query):
    q = query
    MegaCacheLayout(q.page_tokens, q.record_bytes).validate(cache)
    if (values.shape != (q.rows, 512) or values.dtype != torch.bfloat16 or not values.is_contiguous()
            or positions.shape != (q.rows,) or positions.dtype != torch.int64 or not positions.is_contiguous()
            or slots.shape != (q.rows,) or slots.dtype != torch.int64 or not slots.is_contiguous()
            or cos_sin.shape != (q.rotary_rows, 64) or cos_sin.dtype != torch.float32
            or not cos_sin.is_contiguous() or cache.stride(0) != q.page_stride
            or any(t.device != values.device for t in (positions, cos_sin, cache, slots))):
        raise ValueError("cache writer requires prepared contiguous BF16 values, int64 positions/slots and FP32 rotary tables")


def _constants(q):
    return 64, q.page_stride, q.page_tokens, q.ratio, False


_NATIVE_JITS.add(_rope_quant_insert_nvfp4_kernel)


@program_cache(scope="preparation")
def compile_insert(payload, ordinal):
    q = CacheWriteQuery(**dict(payload))
    with torch.cuda.device(ordinal):
        return _rope_quant_insert_nvfp4_kernel.warmup(
            torch.bfloat16, torch.int64, torch.float32, torch.uint8, torch.int64,
            *_constants(q), grid=(q.rows, 1, 1), num_warps=4, launch_pdl=False)


CONTRACT = native_contract(component_id="flare.mega_attention.cache_write",
                           query_type=CacheWriteQuery, tactics=lambda query: (None,))


class _State:
    def __init__(self, query, device, kernel_cache):
        self.query = query
        with torch.cuda.device(device):
            if query.record_bytes == 528:
                self.writer = prepare(kernel_cache)
            else:
                self.writer = load_programs(compile_insert(CONTRACT.encode_query(query), device.index))

    def run(self, values, positions, cos_sin, cache, slots):
        q = self.query
        if q.record_bytes == 528:
            self.writer(values, cache, slots, positions, cos_sin, q.page_tokens, None, True)
        else:
            self.writer[(q.rows, 1, 1)](values, positions, cos_sin, cache, slots, *_constants(q))


@torch.library.custom_op("flare::mega_cache_write", mutates_args=("cache",))
def _write(values: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
           cache: torch.Tensor, slots: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, values.device).run(values, positions, cos_sin, cache, slots)


@_write.register_fake
def _write_fake(values, positions, cos_sin, cache, slots, handle):
    pass


class PreparedMegaCacheWriter:
    """Fuse rotary transformation, quantization and insertion without intermediate cache records.

    Valid slots must be unique within a call and address the supplied cache.
    Negative slots suppress writes. Positions for valid rows address the rotary
    table. NVFP4 ratio two publishes only closed pairs, using the pair's first
    position for rotation. Inputs and page padding are unchanged.
    """

    def __init__(self, values, positions, cos_sin, cache, slots, *, ratio: int,
                 kernel_cache: Path, session, name):
        if cache.ndim != 3:
            raise ValueError("cache writer requires a paged cache view")
        layout = MegaCacheLayout(cache.shape[1], cache.shape[2])
        layout.validate(cache)
        if type(ratio) is not int or ratio not in (1, 2) or (layout.record_bytes == 528 and ratio != 1):
            raise ValueError("MXFP8 windows require ratio one; NVFP4 caches require ratio one or two")
        if values.ndim != 2 or values.shape[0] <= 0 or cos_sin.ndim != 2 or cos_sin.shape[0] <= 0:
            raise ValueError("cache writer values and rotary tables require positive row capacities")
        device = values.device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 3):
            raise ValueError("the reference cache writer requires SM103")
        q = CacheWriteQuery(values.shape[0], layout.page_tokens, layout.record_bytes,
                            cache.stride(0), ratio, cos_sin.shape[0])
        _validate(values, positions, cos_sin, cache, slots, q)
        self.plan = native_plan(contract=CONTRACT, query=q, device=device,
            scratch_bytes=lambda tactic: 0,
            compile_jobs=lambda tactic, detected: () if q.record_bytes == 528 else (CompileJob.create(
                "ignition.components.mega_attention_cache:compile_insert", CONTRACT.encode_query(q), detected.ordinal),),
            materialize=lambda tactic: _State(q, device, kernel_cache))
        examples = [values, positions, cos_sin, cache, slots]

        def prepare_call(state):
            values, positions, cos_sin, cache, slots = examples
            target = torch.empty_strided(cache.shape, cache.stride(), dtype=cache.dtype, device=device)
            return PreparedCall(run=lambda: state.run(values, positions, cos_sin, target, slots), output=target)

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)

    def __call__(self, values, positions, cos_sin, cache, slots):
        _validate(values, positions, cos_sin, cache, slots, self.plan.query)
        _write(values, positions, cos_sin, cache, slots, self.plan.handle)


# Window continuation stores the ring in the same separated-plane format as
# attention pages. Speculative/prefill staging adds suffix slots to that page;
# copying a contiguous 128-record prefix would put scales in the wrong region.
@triton.jit
def _snapshot_window(History, Staged, Rows, HISTORY_STRIDE: tl.constexpr,
                     STAGED_STRIDE: tl.constexpr, STAGED_TOKENS: tl.constexpr):
    request, token = tl.program_id(0), tl.program_id(1)
    row = tl.load(Rows + request)
    byte = tl.arange(0, 1024)
    source = tl.where(byte < 512, token * 512 + byte, 128 * 512 + token * 16 + byte - 512)
    target = tl.where(byte < 512, token * 512 + byte, STAGED_TOKENS * 512 + token * 16 + byte - 512)
    live = (row > 0) & (byte < 528)
    value = tl.load(History + row * HISTORY_STRIDE + source, live, other=0)
    tl.store(Staged + request * STAGED_STRIDE + target, value, live)


def _validate_window_transfer(history, staged, rows):
    MegaCacheLayout(128, 528).validate(history)
    if staged.ndim != 3 or staged.shape[1] <= 128:
        raise ValueError("window staging requires 128 history slots followed by a positive suffix capacity")
    MegaCacheLayout(staged.shape[1], 528).validate(staged)
    if (rows.shape != (staged.shape[0],) or rows.dtype != torch.int64 or not rows.is_contiguous()
            or history.device != staged.device or rows.device != staged.device):
        raise ValueError("window transfers require one contiguous int64 state row per staging page on the same device")


def snapshot_window(history, staged, rows):
    """Copy active continuation rings into staging prefixes; row zero is inactive.

    State rows must index history. Staging suffixes and padding remain unchanged.
    """
    _validate_window_transfer(history, staged, rows)
    torch.library.wrap_triton(_snapshot_window)[(rows.numel(), 128)](
        history, staged, rows, history.stride(0), staged.stride(0), staged.shape[1])


def commit_window(staged, history, rows, starts, counts):
    """Commit only accepted suffix tokens, preserving every rejected token's history.

    Active state rows must be distinct. Counts are in [0, suffix capacity] and
    starts are nonnegative absolute positions; the scheduler and speculative
    commit computation supply those bounds. Only the final 128 accepted tokens
    of a long prefill need to be retained in the continuation ring.
    """
    _validate_window_transfer(history, staged, rows)
    if any(t.shape != rows.shape or t.dtype != torch.int64 or not t.is_contiguous()
           or t.device != rows.device for t in (starts, counts)):
        raise ValueError("window commit requires contiguous int64 start positions and accepted counts")
    torch.library.wrap_triton(_commit)[(min(staged.shape[1] - 128, 128), rows.numel(), 1)](
        staged, history, None, rows, starts, counts, history.stride(0), staged.stride(0), 0,
        rows.stride(0), starts.stride(0), counts.stride(0), 128, staged.shape[1], (512, 16), 528, 1024)
