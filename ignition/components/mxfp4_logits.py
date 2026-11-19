# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared reference SM103 contiguous and paged MXFP4 query/key scoring."""

from dataclasses import dataclass

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.mega_attention import _spec
from ignition.components.megamoe_kernel.mqa import prepare
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class ContiguousLogitsQuery:
    rows: int
    columns: int


@dataclass(frozen=True)
class LogitsQuery:
    rows: int
    pages: int
    page_tokens: int
    columns: int
    sms: int


@dataclass(frozen=True)
class SparseLogitsQuery:
    rows: int
    pages: int
    page_tokens: int
    blocks: int
    block_tokens: int
    sms: int


CONTRACT = native_contract(component_id="flare.mxfp4.paged_logits",
                           query_type=LogitsQuery, tactics=lambda query: (None,))
SPARSE_CONTRACT = native_contract(component_id="flare.mxfp4.sparse_paged_logits",
                                 query_type=SparseLogitsQuery, tactics=lambda query: (None,))
CONTIGUOUS_CONTRACT = native_contract(component_id="flare.mxfp4.contiguous_logits",
    query_type=ContiguousLogitsQuery, tactics=lambda query: (None,))


class _ContiguousState:
    def __init__(self, kernel_cache):
        self.library = prepare(kernel_cache)

    def run(self, operands, output):
        self.library.contiguous(*operands, output)


@torch.library.custom_op("flare::mxfp4_contiguous_logits", mutates_args=("output",))
def _run_contiguous(query: torch.Tensor, scales: torch.Tensor, keys: torch.Tensor,
                    key_scales: torch.Tensor, weights: torch.Tensor, starts: torch.Tensor,
                    ends: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, query.device).run(
        (query, scales, keys, key_scales, weights, starts, ends), output)


@_run_contiguous.register_fake
def _run_contiguous_fake(query, scales, keys, key_scales, weights, starts, ends, output, handle):
    pass


def _contiguous_output(query, device):
    return torch.empty(((query.rows + 3) // 4 * 4, (query.columns + 511) // 256 * 256),
                       dtype=torch.float32, device=device)


class PreparedContiguousMXFP4Logits:
    """Score contiguous keys with FP32 head weights and reference prefill arithmetic.

    Each query selects a half-open interval in the shared key array. Admission
    must enforce 0 <= starts <= ends <= key capacity; empty intervals do no
    work. Output outside those intervals is undefined and must be masked by
    the consumer. Keys and their four UE8M0 scale bytes are copied losslessly
    from the cache before this operation; the scorer performs no gathering.
    """

    def __init__(self, query, scales, keys, key_scales, weights, starts, ends, *,
                 kernel_cache, session, name):
        rows = query.shape[0] if query.ndim else 0
        columns = keys.shape[0] if keys.ndim else 0
        operands = query, scales, keys, key_scales, weights, starts, ends
        expected = (((rows, 32, 64), torch.int8), ((rows, 32), torch.int32),
                    ((columns, 64), torch.int8), ((columns,), torch.int32),
                    ((rows, 32), torch.float32), ((rows,), torch.int32), ((rows,), torch.int32))
        if (not 0 < rows <= 131072 or not 0 < columns <= 1 << 24
                or any(t.shape != shape or t.dtype != dtype or not t.is_contiguous()
                       or t.device != query.device for t, (shape, dtype) in zip(operands, expected))):
            raise ValueError("contiguous MXFP4 logits require packed 32-head queries, 128-element keys and int32 intervals")
        if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device) != (10, 3):
            raise ValueError("the reference MXFP4 logits require SM103")
        self.specs = tuple(map(_spec, operands))
        q = ContiguousLogitsQuery(rows, columns)
        self.plan = native_plan(contract=CONTIGUOUS_CONTRACT, query=q, device=query.device,
            scratch_bytes=lambda tactic: 0, materialize=lambda tactic: _ContiguousState(kernel_cache))
        examples, device = list(operands), query.device

        def prepare_call(state):
            output = _contiguous_output(q, device)
            return PreparedCall(run=lambda: state.run(examples, output), output=output)

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)

    def __call__(self, query, scales, keys, key_scales, weights, starts, ends):
        operands = query, scales, keys, key_scales, weights, starts, ends
        if tuple(map(_spec, operands)) != self.specs:
            raise ValueError("contiguous MXFP4 operands must match their prepared geometry, dtype, strides and device")
        output = _contiguous_output(self.plan.query, query.device)
        _run_contiguous(*operands, output, self.plan.handle)
        return output[:self.plan.query.rows, :self.plan.query.columns]


class _State:
    def __init__(self, kernel_cache):
        self.library = prepare(kernel_cache)

    def run(self, operands, schedule, output):
        self.library.paged(*operands, schedule, output)


class _SparseState:
    def __init__(self, query, kernel_cache):
        self.library = prepare(kernel_cache)
        self.query = query
        self.sizes = self.library.sparse_sizes(query.rows, query.blocks, query.block_tokens)

    def run(self, operands, metadata, workspace, output):
        self.library.sparse(*operands, metadata, workspace, output, self.query.block_tokens)


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


@torch.library.custom_op("flare::mxfp4_sparse_paged_logits", mutates_args=("metadata", "workspace", "output"))
def _run_sparse(query: torch.Tensor, scales: torch.Tensor, cache: torch.Tensor,
                weights: torch.Tensor, lengths: torch.Tensor, pages: torch.Tensor,
                requests: torch.Tensor, blocks: torch.Tensor, metadata: torch.Tensor,
                workspace: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, query.device).run(
        (query, scales, cache, weights, lengths, pages, requests, blocks), metadata, workspace, output)


@_run_sparse.register_fake
def _run_sparse_fake(query, scales, cache, weights, lengths, pages, requests, blocks,
                     metadata, workspace, output, handle):
    pass


def _outputs(query, device):
    stride = (query.columns + 255) // 256 * 256
    # The reference kernel writes complete 256-key tiles, including the tail.
    return (torch.empty((query.sms + 1, 2), dtype=torch.int32, device=device),
            torch.empty((query.rows, stride), dtype=torch.float32, device=device))


def _sparse_outputs(query, sizes, device):
    stride = (query.blocks * query.block_tokens + 511) // 512 * 512
    # The metadata builder requires zero counters and resets them on completion.
    return (torch.empty((sizes[0],), dtype=torch.uint8, device=device),
            torch.zeros((sizes[1],), dtype=torch.uint8, device=device),
            torch.empty(((query.rows + 1) // 2 * 2, stride), dtype=torch.bfloat16, device=device))


def _validate_inputs(query, scales, cache, weights, lengths, pages, columns, weight_dtype):
    rows = query.shape[0] if query.ndim else 0
    operands = query, scales, cache, weights, lengths, pages
    if (query.ndim != 3 or query.shape != (rows, 32, 64) or rows <= 0
            or query.dtype != torch.int8 or not query.is_contiguous()
            or scales.shape != (rows, 32) or scales.dtype != torch.int32 or not scales.is_contiguous()
            or weights.shape != (rows, 32) or weights.dtype != weight_dtype or not weights.is_contiguous()
            or lengths.shape != (rows,) or lengths.dtype != torch.int32 or not lengths.is_contiguous()
            or pages.ndim != 2 or pages.shape[0] != rows or pages.shape[1] <= 0
            or pages.dtype != torch.int32 or not pages.is_contiguous()
            or pages.stride(0) < pages.shape[1]
            or cache.ndim != 3 or cache.shape[0] <= 0 or cache.shape[1] not in (64, 128)
            or cache.shape[2] != 68 or cache.dtype != torch.uint8
            or cache.stride()[1:] != (68, 1) or cache.stride(0) % 512
            or cache.stride(0) < cache.shape[1] * 68
            or type(columns) is not int or not 0 < columns <= pages.shape[1] * cache.shape[1]
            or any(t.device != query.device for t in operands)):
        raise ValueError("MXFP4 logits require packed 32-head queries, planar cache pages, and matching int32 metadata")
    if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device) != (10, 3):
        raise ValueError("the reference MXFP4 logits require SM103")
    return rows


class PreparedPagedMXFP4Logits:
    """Score visible keys; the caller masks columns at or beyond each row's length.

    Each query row has its own page table and visible key count, including
    speculative rows of the same request. Cache pages contain separate 64-byte
    packed-value and four-byte UE8M0 scale planes. Admission must bound lengths
    to the declared columns and page IDs to the cache. Empty rows do no work.
    """

    def __init__(self, query, scales, cache, weights, lengths, pages, *, columns,
                 kernel_cache, session, name):
        rows = _validate_inputs(query, scales, cache, weights, lengths, pages, columns, torch.float32)
        operands = query, scales, cache, weights, lengths, pages
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


class PreparedSparsePagedMXFP4Logits:
    """Score sorted candidate blocks using reference BF16 head weights and logits.

    Each row's candidate prefix must be unique, increasing and inside its
    visible context, with padding repeating the final valid block. Request IDs
    group consecutive queries; rows of one request must share a page table.
    The caller masks logits outside the valid sparse-token prefix. Empty
    contexts select no keys. Values and scales retain the dense cache layout.
    """

    def __init__(self, query, scales, cache, weights, lengths, pages, requests, blocks,
                 *, block_tokens, kernel_cache, session, name):
        columns = pages.shape[1] * cache.shape[1] if pages.ndim == 2 and cache.ndim == 3 else 0
        rows = _validate_inputs(query, scales, cache, weights, lengths, pages, columns, torch.bfloat16)
        if (rows > 1 << 20 or requests.shape != (rows,) or requests.dtype != torch.int32
                or not requests.is_contiguous() or requests.device != query.device
                or blocks.ndim != 2 or blocks.shape[0] != rows or blocks.dtype != torch.int32
                or blocks.device != query.device or not blocks.is_contiguous()
                or not 0 < blocks.shape[1] <= 4096 or blocks.shape[1] % 4
                or type(block_tokens) is not int or block_tokens not in (8, 16)):
            raise ValueError("sparse logits require int32 request IDs and up to 4096 candidate blocks in multiples of four")
        operands = query, scales, cache, weights, lengths, pages, requests, blocks
        self.specs = tuple(map(_spec, operands))
        q = SparseLogitsQuery(rows, pages.shape[1], cache.shape[1], blocks.shape[1], block_tokens,
                             torch.cuda.get_device_properties(query.device).multi_processor_count)
        self.plan = native_plan(contract=SPARSE_CONTRACT, query=q, device=query.device,
            scratch_bytes=lambda tactic: 0, materialize=lambda tactic: _SparseState(q, kernel_cache))
        examples = list(operands)
        device = query.device

        def prepare_call(state):
            metadata, workspace, output = _sparse_outputs(q, state.sizes, device)
            return PreparedCall(run=lambda: state.run(examples, metadata, workspace, output), output=output)

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)
        register_finalizer(session, lambda: self._finalize(device))

    def _finalize(self, device):
        self.sizes = require_prepared(self.plan, self.plan.component_id, device).sizes

    def __call__(self, query, scales, cache, weights, lengths, pages, requests, blocks):
        operands = query, scales, cache, weights, lengths, pages, requests, blocks
        if tuple(map(_spec, operands)) != self.specs:
            raise ValueError("sparse MXFP4 logits operands must match their prepared geometry, dtype, strides and device")
        metadata, workspace, output = _sparse_outputs(self.plan.query, self.sizes, query.device)
        _run_sparse(*operands, metadata, workspace, output, self.plan.handle)
        q = self.plan.query
        return output[:q.rows, :q.blocks * q.block_tokens]
