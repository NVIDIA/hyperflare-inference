# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Losslessly gather planar MXFP4 index-cache pages for contiguous scoring."""

from dataclasses import dataclass

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.mega_attention import _spec
from ignition.components.mega_cache_kernel import prepare_gather
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.tuning import declare


@dataclass(frozen=True)
class GatherQuery:
    requests: int
    pages_per_request: int
    page_tokens: int
    page_stride: int
    capacity: int


CONTRACT = native_contract(component_id="flare.mxfp4.index_gather",
                          query_type=GatherQuery, tactics=lambda query: (None,))


class _State:
    def __init__(self, kernel_cache):
        self.gather = prepare_gather(kernel_cache)

    def run(self, cache, pages, offsets, keys, scales):
        self.gather(cache, keys, scales, pages, offsets)


@torch.library.custom_op("flare::mxfp4_index_gather", mutates_args=("keys", "scales"))
def _run(cache: torch.Tensor, pages: torch.Tensor, offsets: torch.Tensor,
         keys: torch.Tensor, scales: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, cache.device).run(cache, pages, offsets, keys, scales)


@_run.register_fake
def _run_fake(cache, pages, offsets, keys, scales, handle):
    pass


def _outputs(q, device):
    return (torch.empty((q.capacity, 64), dtype=torch.int8, device=device),
            torch.empty((q.capacity, 4), dtype=torch.uint8, device=device))


class PreparedMXFP4IndexGather:
    """Copy packed keys and scale bytes without dequantization or rounding.

    Offsets contain request prefix sums starting at zero. They must be
    nondecreasing, end within capacity, and describe lengths covered by each
    request's page table. Live page IDs must address the cache. Empty requests
    read no pages; output after the final offset is undefined. Callers must
    exclude that tail from subsequent scoring intervals.
    """

    def __init__(self, cache, pages, offsets, *, capacity, kernel_cache, session, name):
        if (cache.ndim != 3 or cache.shape[0] <= 0 or cache.shape[1] not in (64, 128)
                or cache.shape[2] != 68 or cache.dtype != torch.uint8
                or cache.stride()[1:] != (68, 1) or cache.stride(0) < cache.shape[1] * 68
                or cache.stride(0) % 512 or pages.ndim != 2 or min(pages.shape) <= 0
                or pages.dtype != torch.int32 or not pages.is_contiguous()
                or offsets.shape != (pages.shape[0] + 1,) or offsets.dtype != torch.int32
                or not offsets.is_contiguous() or type(capacity) is not int or not 0 < capacity < 2**24
                or pages.device != cache.device or offsets.device != cache.device):
            raise ValueError("index gathering requires planar MXFP4 pages, contiguous int32 page tables and request offsets")
        if cache.device.type != "cuda" or torch.cuda.get_device_capability(cache.device) != (10, 3):
            raise ValueError("reference index gathering requires SM103")
        self.specs = tuple(map(_spec, (cache, pages, offsets)))
        q = GatherQuery(*pages.shape, cache.shape[1], cache.stride(0), capacity)
        device = cache.device
        self.plan = native_plan(contract=CONTRACT, query=q, device=device,
            scratch_bytes=lambda tactic: 0, materialize=lambda tactic: _State(kernel_cache))

        def prepare_call(state):
            # Repeated page zero bounds tuning reads independently of live cache mappings.
            backing = torch.zeros((1, q.page_stride), dtype=torch.uint8, device=device)
            fixture = backing.as_strided((1, q.page_tokens, 68), (q.page_stride, 68, 1))
            table = torch.zeros((q.requests, q.pages_per_request), dtype=torch.int32, device=device)
            length = min(q.capacity // q.requests, q.pages_per_request * q.page_tokens)
            bounds = torch.arange(q.requests + 1, dtype=torch.int32, device=device) * length
            keys, scales = _outputs(q, device)
            return PreparedCall(run=lambda: state.run(fixture, table, bounds, keys, scales),
                                output=(keys, scales))

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))

    def __call__(self, cache, pages, offsets):
        if tuple(map(_spec, (cache, pages, offsets))) != self.specs:
            raise ValueError("index gather inputs must match their prepared geometry, dtype, strides and device")
        keys, scales = _outputs(self.plan.query, cache.device)
        _run(cache, pages, offsets, keys, scales, self.plan.handle)
        return keys, scales.view(torch.int32).squeeze(-1)
