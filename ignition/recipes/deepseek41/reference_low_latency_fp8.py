# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 low-latency FP8 linear projections for DeepSeek V4.1 reference execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn
import triton
import triton.language as tl
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import Plan, PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare

LOW_LATENCY_FP8_COMPONENT = "flare.sm103.low_latency_fp8"
LOW_LATENCY_FP8_MAX_ROWS = 8


def _low_latency_runner(device):
    from flashinfer.gemm import gemm_base

    major, minor = torch.cuda.get_device_capability(device)
    return gemm_base._cutedsl_low_latency_blockscaled_gemm_runner(major * 10 + minor, True)


@triton.jit
def _deepgemm_sf_to_f8_128x4(
    Src, Dst, src_stride_col,
    M: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = tile_k < K_TILES
    for r in tl.static_range(M):
        val = tl.load(Src + r + tile_k * src_stride_col, mask=mask_k, other=0)
        tl.store(Dst + tile_k * 128 + r * 4, val, mask=mask_k)


_NATIVE_JITS.add(_deepgemm_sf_to_f8_128x4)


@torch.no_grad()
def deepgemm_weight_sf_to_f8_128x4(w_sf_int32: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """Convert DeepGEMM packed int32 UE8M0 weight scales to FlashInfer F8_128x4 uint8 layout."""
    if w_sf_int32.dtype != torch.int32 or in_features % 128 or out_features % 128:
        raise ValueError("low-latency FP8 weight scales require int32 packed UE8M0 and 128-aligned N, K")
    raw = w_sf_int32.contiguous().view(torch.uint8).reshape(out_features, in_features // 32)
    k_blk = in_features // 32
    swizzled = raw.view(out_features // 128, 4, 32, k_blk // 4, 4).permute(0, 3, 2, 1, 4).contiguous()
    return swizzled.view(-1)


@dataclass(frozen=True)
class LowLatencyFP8Query:
    rows: int
    in_features: int
    out_features: int

    def __post_init__(self):
        if (type(self.rows) is not int or not 1 <= self.rows <= LOW_LATENCY_FP8_MAX_ROWS
                or type(self.in_features) is not int or self.in_features <= 0 or self.in_features % 128
                or type(self.out_features) is not int or self.out_features <= 0 or self.out_features % 128):
            raise ValueError("SM103 low-latency FP8 projection requires 1..8 rows and 128-aligned N, K")


@lru_cache(maxsize=None)
def _low_latency_fp8_tactics(query: LowLatencyFP8Query) -> tuple:
    target = torch.device("cuda", torch.cuda.current_device())
    rows, k, n = query.rows, query.in_features, query.out_features
    a = torch.empty((1,), dtype=torch.float8_e4m3fn, device=target).expand(rows, k)
    b = torch.empty((1,), dtype=torch.float8_e4m3fn, device=target).expand(k, n)
    sf = torch.empty((1,), dtype=torch.uint8, device=target)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device="meta")
    valid = tuple(_low_latency_runner(target).get_valid_tactics(
        [a, b, sf, sf, torch.bfloat16, out, None], None))
    if n >= 16384:
        preferred = ((128, 5, 1, 1), (128, 6, 1, 1), (128, 4, 1, 1))
        curated = [t for t in valid if t[2] == 1 and t[3] == 1 and t[0] in (128, 256) and t[1] in (3, 4, 5, 6)]
    elif n >= 4096:
        preferred = ((128, 6, 1, 2), (128, 8, 1, 2), (256, 5, 1, 2))
        curated = [t for t in valid if t[2] == 1 and t[3] in (1, 2) and t[0] in (128, 256) and t[1] in (5, 6, 8, 9)]
    else:
        preferred = ((128, 6, 1, 8), (640, 2, 1, 8), (128, 7, 1, 8), (128, 5, 1, 4))
        curated = [t for t in valid if t[2] == 1 and t[3] in (4, 8) and t[0] in (128, 256, 640) and t[1] in (2, 4, 5, 6, 7)]
    ordered = [t for t in preferred if t in valid] + [t for t in curated if t not in preferred]
    return tuple(dict.fromkeys(ordered)) or valid


LOW_LATENCY_FP8_CONTRACT = native_contract(
    component_id=LOW_LATENCY_FP8_COMPONENT,
    query_type=LowLatencyFP8Query,
    tactics=_low_latency_fp8_tactics,
)


@program_cache(scope="preparation")
def compile_low_latency_fp8_swizzle(payload, ordinal):
    query = LowLatencyFP8Query(**payload["query"])
    k_tiles = query.in_features // 128
    block_k = triton.next_power_of_2(k_tiles)
    with torch.cuda.device(ordinal):
        return (_deepgemm_sf_to_f8_128x4.warmup(
            torch.int32, torch.int32, 16,
            query.rows, k_tiles, block_k,
            grid=(1, 1, 1), num_warps=4,
        ),)


class PreparedLowLatencyFP8:
    def __init__(self, query: LowLatencyFP8Query, tactic, device):
        self.query = query
        self.tactic = tactic
        self.device = torch.device(device)
        self.k_tiles = query.in_features // 128
        self.block_k = triton.next_power_of_2(self.k_tiles)
        self.swizzle, = load_programs(compile_low_latency_fp8_swizzle(
            dict(query=LOW_LATENCY_FP8_CONTRACT.encode_query(query)), self.device.index))
        self.runner = _low_latency_runner(self.device)

    def run(self, x_fp8, x_sf, weight, weight_sf_f8, out, sf_scratch):
        self.swizzle[(1, 1, 1)](
            x_sf, sf_scratch, x_sf.stride(1),
            self.query.rows, self.k_tiles, self.block_k,
        )
        self.runner(
            inputs=[x_fp8, weight.T, sf_scratch.view(torch.uint8), weight_sf_f8, torch.bfloat16, out, None],
            tactic=self.tactic,
        )


def low_latency_fp8_plan(query: LowLatencyFP8Query, device) -> Plan:
    resolved = torch.device(device)
    return native_plan(
        contract=LOW_LATENCY_FP8_CONTRACT,
        query=query,
        device=resolved,
        scratch_bytes=lambda tactic: 0,
        compile_jobs=lambda tactic, detected: (CompileJob.create(
            "ignition.recipes.deepseek41.reference_low_latency_fp8:compile_low_latency_fp8_swizzle",
            dict(query=LOW_LATENCY_FP8_CONTRACT.encode_query(query)), detected.ordinal),),
        materialize=lambda tactic: PreparedLowLatencyFP8(query, tactic, resolved),
    )


@torch.library.custom_op("flare::sm103_low_latency_fp8", mutates_args=("out", "sf_scratch"))
def _low_latency_fp8(
    x_fp8: torch.Tensor, x_sf: torch.Tensor, weight: torch.Tensor, weight_sf_f8: torch.Tensor,
    out: torch.Tensor, sf_scratch: torch.Tensor, plan_handle: int,
) -> None:
    require_prepared(plan_from_handle(plan_handle), LOW_LATENCY_FP8_COMPONENT, x_fp8.device).run(
        x_fp8, x_sf, weight, weight_sf_f8, out, sf_scratch)


@_low_latency_fp8.register_fake
def _fake_low_latency_fp8(x_fp8, x_sf, weight, weight_sf_f8, out, sf_scratch, plan_handle):
    pass


class LowLatencyFP8Linear(nn.Module):
    """Wrap a DeepInfer FP8 ReplicatedLinear to run FlashInfer low-latency GEMM for M <= 8."""

    def __init__(self, base, rows_list, *, session, name: str):
        super().__init__()
        self.base = base
        self.input_size = base.input_size
        self.output_size = base.output_size
        self.use_fp32_output = base.use_fp32_output
        device = base.weight.device
        sf_f8 = persistent_copy(deepgemm_weight_sf_to_f8_128x4(
            base.weight.sf, self.output_size, self.input_size))
        self.register_buffer("weight_sf_f8", sf_f8)
        record_numeric_tensor(self.weight_sf_f8)
        self._plans = {}
        requests = []
        for rows in sorted({r for r in rows_list if 1 <= r <= LOW_LATENCY_FP8_MAX_ROWS}):
            query = LowLatencyFP8Query(rows, self.input_size, self.output_size)
            op_plan = low_latency_fp8_plan(query, device)
            self._plans[rows] = op_plan

            def make_prepare(rows=rows, k=self.input_size, n=self.output_size):
                def prepare(state):
                    aligned_m = ((rows * 4 + 15) // 16) * 16 // 4
                    x_fp8 = torch.zeros((rows, k), dtype=torch.float8_e4m3fn, device=device)
                    x_sf = torch.empty_strided((rows, k // 128), (1, aligned_m), dtype=torch.int32, device=device)
                    x_sf.fill_(0x7F7F7F7F)
                    out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
                    sf_scratch = torch.empty((128 * (k // 128),), dtype=torch.int32, device=device)

                    def produce():
                        x_fp8.zero_()
                        x_sf.fill_(0x7F7F7F7F)

                    return PreparedCall(
                        run=lambda: state.run(x_fp8, x_sf, base.weight, self.weight_sf_f8, out, sf_scratch),
                        produce=produce,
                        output=out,
                        owners=(base.weight, self.weight_sf_f8),
                    )
                return prepare

            prep = make_prepare()
            requests.append(op_plan.request(name=f"{name}.m{rows}", prepare_call=prep, benchmark_call=prep))
        if requests:
            declare(session, tuple(requests))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x, out_acc=None):
        if out_acc is None:
            if isinstance(x, torch.Tensor):
                rows = x.shape[0] if x.ndim == 2 else 0
            else:
                rows = x[0].shape[0] if x[0].ndim == 2 else 0
            plan = self._plans.get(rows)
            if plan is not None:
                if isinstance(x, torch.Tensor):
                    from deepinfer.model_executor.layers.linear import cast_fp8
                    x_fp8, x_sf = cast_fp8(x)
                else:
                    x_fp8, x_sf = x
                if x_fp8.is_contiguous() and x_sf.dtype == torch.int32 and x_sf.stride(0) == 1:
                    out = torch.empty((rows, self.output_size), dtype=torch.bfloat16, device=x_fp8.device)
                    sf_scratch = torch.empty((128 * (self.input_size // 128),), dtype=torch.int32, device=x_fp8.device)
                    _low_latency_fp8(x_fp8, x_sf, self.base.weight, self.weight_sf_f8, out, sf_scratch, plan.handle)
                    return out
        return self.base(x, out_acc=out_acc)
