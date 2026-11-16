# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared grouped projection consuming fused attention FP8 output directly."""

from dataclasses import dataclass
from pathlib import Path

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.megamoe_kernel import initialize
from ignition.components.mega_attention import _spec
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class GroupedProjectionQuery:
    rows: int
    groups: int
    outputs: int
    inputs: int


CONTRACT = native_contract(component_id="flare.mxfp8.grouped_projection",
                           query_type=GroupedProjectionQuery, tactics=lambda query: (None,))


class _State:
    def __init__(self, kernel_cache):
        self.library = initialize(kernel_cache)

    def run(self, values, scales, weight, weight_scales, output):
        self.library.fp8_einsum("bhr,hdr->bhd", (values, scales), (weight, weight_scales),
                               output, recipe=(1, 1, 32))


@torch.library.custom_op("flare::mxfp8_grouped_projection", mutates_args=("output",))
def _run(values: torch.Tensor, scales: torch.Tensor, weight: torch.Tensor,
         weight_scales: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, values.device).run(values, scales, weight, weight_scales, output)


@_run.register_fake
def _run_fake(values, scales, weight, weight_scales, output, handle):
    pass


class PreparedGroupedFP8Projection:
    """Consume packed MN-major scales without decoding or requantizing activations.

    Weights have the consumer's column permutation applied during preparation.
    The BF16 result preserves the reference intermediate before the final dense
    projection. The caller owns that projection and its activation quantization.
    """

    def __init__(self, values, scales, weight, weight_scales, *, kernel_cache: Path, session, name):
        if values.ndim != 3 or weight.ndim != 3:
            raise ValueError("grouped projection requires three-dimensional activations and weights")
        rows, groups, inputs = values.shape
        wg, outputs, wi = weight.shape
        if (min(rows, groups, inputs, outputs) <= 0 or groups != wg or inputs != wi
                or inputs % 128 or outputs % 128 or values.dtype != torch.float8_e4m3fn
                or weight.dtype != values.dtype or not values.is_contiguous() or not weight.is_contiguous()
                or scales.shape != (rows, groups, inputs // 128) or scales.dtype != torch.int32
                or scales.stride() != (1, inputs // 128 * ((rows + 3) // 4 * 4), (rows + 3) // 4 * 4)
                or weight_scales.shape != (groups, outputs, inputs // 128) or weight_scales.dtype != torch.int32
                or weight_scales.stride() != (outputs * (inputs // 128), 1, outputs)
                or any(t.device != values.device for t in (scales, weight, weight_scales))):
            raise ValueError("grouped projection requires contiguous FP8 values and matched packed MN-major scales")
        if values.device.type != "cuda" or torch.cuda.get_device_capability(values.device) != (10, 3):
            raise ValueError("the reference grouped projection requires SM103")
        self.weights = weight, weight_scales
        for tensor in self.weights:
            record_numeric_tensor(tensor)
        self.specs = _spec(values), _spec(scales)
        query = GroupedProjectionQuery(rows, groups, outputs, inputs)
        device = values.device
        self.plan = native_plan(contract=CONTRACT, query=query, device=device, scratch_bytes=lambda tactic: 0,
                                materialize=lambda tactic: _State(kernel_cache))
        examples = [values, scales]

        def prepare_call(state):
            output = torch.empty((rows, groups, outputs), dtype=torch.bfloat16, device=device)
            return PreparedCall(run=lambda: state.run(*examples, *self.weights, output), output=output)

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)

    def __call__(self, values, scales):
        if (_spec(values), _spec(scales)) != self.specs:
            raise ValueError("grouped projection inputs must match their prepared geometry, dtype, strides and device")
        q = self.plan.query
        output = torch.empty((q.rows, q.groups, q.outputs), dtype=torch.bfloat16, device=values.device)
        _run(values, scales, *self.weights, output, self.plan.handle)
        return output
