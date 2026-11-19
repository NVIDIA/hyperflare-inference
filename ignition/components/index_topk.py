# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared reference selection for dense and candidate-restricted index scores."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.index_topk_kernel import prepare
from ignition.components.mega_attention import _spec
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.tuning import declare, register_finalizer

_deepselect_library = None
_deepselect_lock = Lock()


def _load_deepselect(library):
    global _deepselect_library
    library = Path(library).resolve(strict=True)
    with _deepselect_lock:
        if _deepselect_library is None:
            torch.ops.load_library(str(library))
            _deepselect_library = library
        elif _deepselect_library != library:
            raise ValueError("one preparation process must use one DeepSelect library")


@dataclass(frozen=True)
class TopKQuery:
    rows: int
    columns: int
    implementation: str


CONTRACT = native_contract(component_id="flare.index_topk", query_type=TopKQuery,
                           tactics=lambda query: (None,))


class _State:
    def __init__(self, query, kernel_cache, library):
        self.query = query
        if query.implementation == "deepselect":
            _load_deepselect(library)
        else:
            self.ops = prepare(kernel_cache)

    def run(self, logits, lengths, starts, output, workspace):
        q = self.query
        if q.implementation == "deepselect":
            torch.ops.deep_select.topk(logits, 512, None, lengths, False, False,
                None, output, None, -1, float("-inf"), False, False)
        elif q.implementation == "prefill":
            self.ops.prefill(logits, starts, lengths, output, q.rows, logits.stride(0), 1, 512)
        else:
            getattr(self.ops, q.implementation)(logits, lengths, output, workspace, 512, q.columns)


@torch.library.custom_op("flare::reference_index_topk", mutates_args=("output", "workspace"))
def _run(logits: torch.Tensor, lengths: torch.Tensor, starts: torch.Tensor,
         output: torch.Tensor, workspace: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, logits.device).run(logits, lengths, starts, output, workspace)


@_run.register_fake
def _run_fake(logits, lengths, starts, output, workspace, handle):
    pass


def _outputs(query, device):
    return (torch.zeros(query.rows, device=device, dtype=torch.int32),
            torch.full((query.rows, 512), -1, device=device, dtype=torch.int32),
            torch.empty((1048576 if query.implementation in ("cooperative", "persistent") else 0,),
                        device=device, dtype=torch.uint8))


class PreparedIndexTopK:
    """Select 512 keys with the reference's dtype/phase-specific algorithms.

    Lengths bound each row's live prefix and must not exceed the prepared
    column count. Prefill accepts packed interval starts and returns indices
    relative to those starts. BF16 scores use DeepSelect; FP32 prefill uses per-row
    selection, and FP32 decode uses cooperative selection through 64 rows,
    then persistent selection. Decode launchers receive the prepared capacity
    as their maximum length bound. Selected IDs are ordered descending, with
    -1 padding last. This fixes the consumer's reduction order for a given
    selected set; selection among tied cutoff scores remains kernel-defined.
    """

    def __init__(self, logits, lengths, *, prefill, kernel_cache, library=None, session, name):
        if (logits.ndim != 2 or logits.shape[0] <= 0 or not 0 < logits.shape[1] < 2**23
                or logits.dtype not in (torch.float32, torch.bfloat16) or logits.stride(1) != 1
                or logits.stride(0) * logits.element_size() % 1024
                or lengths.shape != (logits.shape[0],) or lengths.dtype != torch.int32
                or not lengths.is_contiguous() or lengths.device != logits.device):
            raise ValueError("reference index top-K requires aligned score rows and contiguous int32 lengths")
        if logits.device.type != "cuda" or torch.cuda.get_device_capability(logits.device) != (10, 3):
            raise ValueError("reference index top-K requires SM103")
        if logits.dtype == torch.bfloat16:
            if library is None:
                raise ValueError("candidate selection requires the pinned DeepSelect library")
            implementation = "deepselect"
        else:
            implementation = "prefill" if prefill else ("cooperative" if logits.shape[0] <= 64 else "persistent")
        q = TopKQuery(*logits.shape, implementation)
        self.specs = _spec(logits), _spec(lengths)
        device = logits.device
        self.plan = native_plan(contract=CONTRACT, query=q, device=device, scratch_bytes=lambda tactic: 0,
            materialize=lambda tactic: _State(q, kernel_cache, library))
        examples = [logits, lengths]

        def prepare_call(state):
            starts, output, workspace = _outputs(q, device)
            return PreparedCall(run=lambda: state.run(*examples, starts, output, workspace), output=output)

        declare(session, (self.plan.request(name=name, prepare_call=prepare_call),))
        register_finalizer(session, examples.clear)

    def __call__(self, logits, lengths, *, starts=None):
        if (_spec(logits), _spec(lengths)) != self.specs:
            raise ValueError("index top-K inputs must match their prepared geometry, dtype, strides and device")
        default_starts, output, workspace = _outputs(self.plan.query, logits.device)
        if starts is None:
            starts = default_starts
        elif self.plan.query.implementation != "prefill" or _spec(starts) != self.specs[1]:
            raise ValueError("packed starts require FP32 prefill selection and matching int32 bounds")
        _run(logits, lengths, starts, output, workspace, self.plan.handle)
        # Native selection scatters with atomics. Canonicalize IDs before
        # attention so execution order cannot perturb its floating-point sum.
        return output.sort(dim=-1, descending=True).values
