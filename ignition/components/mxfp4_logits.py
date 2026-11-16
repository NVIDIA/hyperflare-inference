# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared reference SM103 paged MXFP4 query/key scoring."""

from dataclasses import dataclass

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.mega_attention import _spec
from ignition.components.megamoe_kernel.mqa import prepare
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class LogitsQuery:
    rows: int
    pages: int
    page_tokens: int
    columns: int
    sms: int


CONTRACT = native_contract(component_id="flare.mxfp4.paged_logits",
                           query_type=LogitsQuery, tactics=lambda query: (None,))


class _State:
    def __init__(self, kernel_cache):
        self.library = prepare(kernel_cache)

    def run(self, operands, schedule, output):
        self.library.paged(*operands, schedule, output)


@torch.library.custom_op("flare::mxfp4_paged_logits", mutates_args=("schedule", "output"))
def _run(query: torch.Tensor, scales: torch.Tensor, cache: torch.Tensor,
         weights: torch.Tensor, lengths: torch.Tensor, pages: torch.Tensor,
         schedule: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, query.device).run(
        (query, scales, cache, weights, lengths, pages), schedule, output)


@_run.register_fake
def _run_fake(query, scales, cache, weights, lengths, pages, schedule, output, handle):
    pass


def _outputs(query, device):
    stride = (query.columns + 255) // 256 * 256
    # The reference kernel writes complete 256-key tiles, including the tail.
    return (torch.empty((query.sms + 1, 2), dtype=torch.int32, device=device),
            torch.empty((query.rows, stride), dtype=torch.float32, device=device))


class PreparedPagedMXFP4Logits:
    """Score visible keys; the caller masks columns at or beyond each row's length.

    Each query row has its own page table and visible key count, including
    speculative rows of the same request. Cache pages contain separate 64-byte
    packed-value and four-byte UE8M0 scale planes. Admission must bound lengths
    to the declared columns and page IDs to the cache. Empty rows do no work.
    """

    def __init__(self, query, scales, cache, weights, lengths, pages, *, columns,
                 kernel_cache, session, name):
        rows = query.shape[0]
        operands = query, scales, cache, weights, lengths, pages
        if (query.ndim != 3 or query.shape != (rows, 32, 64) or rows <= 0
                or query.dtype != torch.int8 or not query.is_contiguous()
                or scales.shape != (rows, 32) or scales.dtype != torch.int32 or not scales.is_contiguous()
                or weights.shape != (rows, 32) or weights.dtype != torch.float32 or not weights.is_contiguous()
                or lengths.shape != (rows,) or lengths.dtype != torch.int32 or not lengths.is_contiguous()
                or pages.ndim != 2 or pages.shape[0] != rows or pages.shape[1] <= 0
                or pages.dtype != torch.int32 or not pages.is_contiguous()
                or cache.ndim != 3 or cache.shape[0] <= 0 or cache.shape[1] not in (64, 128)
                or cache.shape[2] != 68 or cache.dtype != torch.uint8
                or cache.stride()[1:] != (68, 1) or cache.stride(0) % 512
                or cache.stride(0) < cache.shape[1] * 68
                or type(columns) is not int or not 0 < columns <= pages.shape[1] * cache.shape[1]
                or any(t.device != query.device for t in operands)):
            raise ValueError("MXFP4 logits require packed 32-head queries, planar cache pages, and matching int32 metadata")
        if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device) != (10, 3):
            raise ValueError("the reference MXFP4 logits require SM103")
        self.specs = tuple(map(_spec, operands))
        q = LogitsQuery(rows, pages.shape[1], cache.shape[1], columns,
                        torch.cuda.get_device_properties(query.device).multi_processor_count)
        self.plan = native_plan(contract=CONTRACT, query=q, device=query.device,
            scratch_bytes=lambda tactic: 0, materialize=lambda tactic: _State(kernel_cache))
        examples = list(operands)
        device = query.device

        def prepare_call(state):
            schedule, output = _outputs(q, device)
            return PreparedCall(run=lambda: state.run(examples, schedule, output), output=output)

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)

    def __call__(self, query, scales, cache, weights, lengths, pages):
        operands = query, scales, cache, weights, lengths, pages
        if tuple(map(_spec, operands)) != self.specs:
            raise ValueError("MXFP4 logits operands must match their prepared geometry, dtype, strides and device")
        schedule, output = _outputs(self.plan.query, query.device)
        _run(*operands, schedule, output, self.plan.handle)
        return output[:, :self.plan.query.columns]
