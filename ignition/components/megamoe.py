# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared TP1 MXFP8/MXFP4 MegaMoE with optional fused FP8 shared experts."""

from dataclasses import dataclass
import math
from pathlib import Path

import torch
import triton
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.megamoe_kernel import initialize
from ignition.components.megamoe_kernel.inputs import _prepare_megamoe_inputs_kernel
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.persistent import persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class MegaMoeGeometry:
    experts: int
    capacity: int
    top_k: int
    hidden: int
    intermediate: int
    shared_experts: int

    def __post_init__(self):
        if (any(type(value) is not int for value in vars(self).values())
                or min(self.experts, self.capacity, self.top_k, self.hidden, self.intermediate) <= 0
                or self.top_k > self.experts or self.hidden % 128 or self.intermediate % 128
                or self.shared_experts not in (0, 1)):
            raise ValueError("MegaMoE requires positive capacities, 128-aligned widths and zero or one shared expert")


class MegaMoeWorkspace:
    """Model-owned protocol storage shared only by serialized, identical geometries.

    Kernel semaphores and ring storage share one native allocation. Initialize it
    once, retain it across graph variants and layers, and expose every launch as
    a mutation. Do not give this buffer to another operator's scratch allocator.
    """

    def __init__(self, geometry, *, device, kernel_cache: Path):
        device = torch.device(device)
        with torch.cuda.device(device):
            if torch.cuda.get_device_capability() != (10, 3):
                raise ValueError("the reference MegaMoE path requires SM103")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("MegaMoE workspace must be prepared before capture")
            self.library = initialize(kernel_cache)
            alignment = self.library._C.get_token_alignment_for_mega_moe()
            self.geometry = MegaMoeGeometry(geometry.experts,
                triton.cdiv(geometry.capacity, alignment) * alignment, geometry.top_k,
                geometry.hidden, geometry.intermediate, geometry.shared_experts)
            g = self.geometry
            nbytes, self.slice_inputs = self.library._C.get_symm_buffer_size_for_mega_moe(
                1, g.experts, g.capacity, g.top_k, g.hidden, g.intermediate,
                "fp8xfp4", "swiglu", g.shared_experts)
            self.buffer = persistent_zeros((nbytes,), dtype=torch.int8, device=device)
            self.views = self.slice_inputs(self.buffer)


@dataclass(frozen=True)
class MegaMoeQuery:
    rows: int
    experts: int
    capacity: int
    top_k: int
    hidden: int
    intermediate: int
    shared_experts: int
    x_sf_stride: tuple
    shared_sf_stride: tuple
    shared_block_m: int
    clamp: float

    def __post_init__(self):
        object.__setattr__(self, "x_sf_stride", tuple(self.x_sf_stride))
        object.__setattr__(self, "shared_sf_stride", tuple(self.shared_sf_stride))


def _validate_inputs(source, ids, probabilities, padding, geometry, device, rows):
    if (source.dtype != torch.bfloat16 or source.shape != (rows, geometry.hidden)
            or not 0 < rows <= geometry.capacity or not source.is_contiguous()
            or ids.shape != (rows, geometry.top_k) or ids.dtype != torch.int32 or not ids.is_contiguous()
            or probabilities.shape != ids.shape or probabilities.dtype != torch.float32
            or not probabilities.is_contiguous() or padding.shape != (rows,)
            or padding.dtype != torch.bool or not padding.is_contiguous()):
        raise ValueError("MegaMoE requires the prepared contiguous BF16 source, int32 routes, FP32 weights and boolean padding")
    if any(t.device != device for t in (source, ids, probabilities, padding)):
        raise ValueError("MegaMoE operands and workspace must share one CUDA device")


def _validate_weights(weights, geometry, *, shared):
    g = geometry
    prefix = () if shared else (g.experts,)
    values_dtype = torch.float8_e4m3fn if shared else torch.int8
    divisor = 1 if shared else 2
    for (values, scales), rows, columns in (
        (weights.gate_up, 2 * g.intermediate, g.hidden),
        (weights.down, g.hidden, g.intermediate),
    ):
        sf_columns = columns // 128
        scale_stride = (1, rows) if shared else (rows * sf_columns, 1, rows)
        if (values.shape != (*prefix, rows, columns // divisor)
                or values.dtype != values_dtype or not values.is_contiguous()
                or scales.shape != (*prefix, rows, sf_columns) or scales.dtype != torch.int32
                or scales.stride() != scale_stride):
            raise ValueError("MegaMoE weights require matching packed values and MN-major UTCCP scales")


def _stage_constants(q):
    return (q.hidden, 1, q.hidden, 1, *q.x_sf_stride, *q.shared_sf_stride,
            q.top_k, 1, q.top_k, 1, 1, q.top_k, 1, q.top_k, 1,
            q.rows, q.hidden, q.top_k, 8 if q.rows >= 64 else 1,
            128, 32, triton.next_power_of_2(q.top_k), q.shared_block_m, True)


_NATIVE_JITS.add(_prepare_megamoe_inputs_kernel)


@program_cache(scope="preparation")
def compile_inputs(payload, ordinal):
    q = MegaMoeQuery(**dict(payload))
    with torch.cuda.device(ordinal):
        return _prepare_megamoe_inputs_kernel.warmup(
            torch.bfloat16, torch.float8_e4m3fn, torch.int32,
            torch.int32 if q.shared_experts else None, torch.int32, torch.float32,
            torch.bool, torch.int64, torch.float32, *_stage_constants(q),
            grid=(1, 1, 1), num_warps=4)


CONTRACT = native_contract(component_id="flare.megamoe.tp1_mxfp8_mxfp4",
                           query_type=MegaMoeQuery, tactics=lambda query: (None,))


class _State:
    def __init__(self, query, workspace):
        self.query, self.workspace = query, workspace
        self.stager = load_programs(compile_inputs(CONTRACT.encode_query(query), workspace.buffer.device.index))

    def run(self, source, ids, probabilities, padding, matrices, output, buffer):
        q = self.query
        x, sf, out_ids, out_weights, _, shared_sf, *_ = self.workspace.slice_inputs(buffer)
        self.stager[(triton.cdiv(q.rows, 8 if q.rows >= 64 else 1), q.hidden // 128, 1)](
            source, x, sf, shared_sf if q.shared_experts else None,
            ids, probabilities, padding, out_ids, out_weights, *_stage_constants(q))
        shared_gate = tuple(matrices[4:6]) if q.shared_experts else None
        shared_down = tuple(matrices[6:8]) if q.shared_experts else None
        self.workspace.library._C.fp8_fp4_mega_moe(
            output, tuple(matrices[:2]), tuple(matrices[2:4]), shared_gate, shared_down,
            None, buffer, [buffer.data_ptr()], 0, q.capacity, q.experts, q.top_k,
            (1, 1, 32), "swiglu", q.clamp, True, 1., 0.)


@torch.library.custom_op("flare::megamoe", mutates_args=("output", "workspace"))
def _run(source: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
         padding: torch.Tensor, matrices: list[torch.Tensor], output: torch.Tensor,
         workspace: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, source.device).run(
        source, ids, probabilities, padding, matrices, output, workspace)


@_run.register_fake
def _run_fake(source, ids, probabilities, padding, matrices, output, workspace, handle):
    pass


class PreparedMegaMoe:
    """Declare fused expert execution with compiler-visible weights and protocol storage."""

    def __init__(self, source, ids, probabilities, padding, routed, shared, *,
                 workspace: MegaMoeWorkspace, clamp: float, session, name):
        g = workspace.geometry
        rows = source.shape[0]
        _validate_inputs(source, ids, probabilities, padding, g, workspace.buffer.device, rows)
        if not math.isfinite(clamp) or clamp <= 0:
            raise ValueError("MegaMoE SwiGLU clamp must be finite and positive")
        if (shared is not None) != bool(g.shared_experts):
            raise ValueError("MegaMoE shared weights must match the workspace geometry")
        _validate_weights(routed, g, shared=False)
        if shared is not None:
            _validate_weights(shared, g, shared=True)
        self.matrices = list((*routed.gate_up, *routed.down))
        if shared is not None:
            self.matrices.extend((*shared.gate_up, *shared.down))
        if any(t.device != workspace.buffer.device for t in
               (source, ids, probabilities, padding, *self.matrices)):
            raise ValueError("MegaMoE operands and workspace must share one CUDA device")
        for tensor in self.matrices:
            record_numeric_tensor(tensor)
        block_m = workspace.library.get_block_m_for_mega_moe(
            1, g.experts, g.capacity, rows, g.top_k, "fp8xfp4") if shared is not None else 1
        q = MegaMoeQuery(rows, g.experts, g.capacity, g.top_k, g.hidden, g.intermediate,
            g.shared_experts, tuple(workspace.views[1].stride()),
            tuple(workspace.views[5].stride()) if shared is not None else (0, 0), block_m, clamp)
        self.workspace = workspace
        self.plan = native_plan(contract=CONTRACT, query=q, device=source.device,
            scratch_bytes=lambda tactic: 0,
            compile_jobs=lambda tactic, device: (CompileJob.create(
                "ignition.components.megamoe:compile_inputs", CONTRACT.encode_query(q), device.ordinal),),
            materialize=lambda tactic: _State(q, workspace))

        examples = [source, ids, probabilities, padding]

        def prepare(state):
            source, ids, probabilities, padding = examples
            output = torch.empty_like(source)
            return PreparedCall(run=lambda: state.run(source, ids, probabilities, padding,
                self.matrices, output, workspace.buffer), output=output,
                owners=(workspace, self.matrices))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))
        register_finalizer(session, examples.clear)

    def __call__(self, source, ids, probabilities, padding):
        _validate_inputs(source, ids, probabilities, padding, self.workspace.geometry,
                         self.workspace.buffer.device, self.plan.query.rows)
        output = torch.empty_like(source)
        _run(source, ids, probabilities, padding, self.matrices, output,
             self.workspace.buffer, self.plan.handle)
        return output
