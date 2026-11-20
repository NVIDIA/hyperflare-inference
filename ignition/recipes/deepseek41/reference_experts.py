# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind DeepInfer's router and normalization outputs to prepared expert storage.

This preparation module implements the MLP call used by DeepInfer decoder layers.
Routing arithmetic is DeepGEMM's reference implementation; Flare owns weight
placement and the serialized expert workspace. It is not a serving recipe.
"""

import math

import torch
from torch import nn

from ignition.preparation.persistent import persistent_copy
from ignition.preparation.rebuild_manifest import declare_module_buffers
from ignition.preparation.tensor_metadata import record_numeric_tensor


@torch.library.custom_op("flare::deepseek_reference_gate", mutates_args=("ids", "probabilities"))
def _gate(source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
          image_bias: torch.Tensor | None, valid: torch.Tensor,
          image_mask: torch.Tensor | None, placement: torch.Tensor, counts: torch.Tensor,
          ids: torch.Tensor, probabilities: torch.Tensor, scaling: float) -> None:
    import deep_gemm

    deep_gemm.bf16_mega_gate(source, weight, num_topk=ids.shape[1],
        use_shared_as_routed=False, num_shared_experts=0, routed_scaling_factor=scaling,
        ep_rank=0, scoring_func="sqrtsoftplus", mask=valid, bias=bias,
        image_bias=image_bias if image_mask is not None else None,
        image_token_mask=image_mask, to_physical_map=placement, logical_count=counts,
        out=(ids, probabilities))


@_gate.register_fake
def _gate_fake(source, weight, bias, image_bias, valid, image_mask, placement, counts,
               ids, probabilities, scaling):
    pass


@torch.library.custom_op("flare::deepseek_reference_traced_gate",
                         mutates_args=("ids", "probabilities", "routes"))
def _traced_gate(source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                 image_bias: torch.Tensor | None, valid: torch.Tensor,
                 image_mask: torch.Tensor | None, placement: torch.Tensor, counts: torch.Tensor,
                 ids: torch.Tensor, probabilities: torch.Tensor, routes: torch.Tensor,
                 scaling: float) -> None:
    import deep_gemm

    deep_gemm.bf16_mega_gate(source, weight, num_topk=ids.shape[1],
        use_shared_as_routed=False, num_shared_experts=0, routed_scaling_factor=scaling,
        ep_rank=0, scoring_func="sqrtsoftplus", mask=valid, bias=bias,
        image_bias=image_bias if image_mask is not None else None,
        image_token_mask=image_mask, to_physical_map=placement, logical_count=counts,
        unmapped_topk_idx=routes, out=(ids, probabilities))


@_traced_gate.register_fake
def _traced_gate_fake(source, weight, bias, image_bias, valid, image_mask, placement, counts,
                      ids, probabilities, routes, scaling):
    pass


class ReferenceExperts(nn.Module):
    """Reference MoE composition with a prepared operation for each row capacity.

    ``placement`` maps each canonical expert to one physical row, with host rows
    preceding HBM rows. Its owner may publish updates only between drained graph
    executions. All operations share weights and workspace. Image tokens use
    the checkpoint's image routing bias; masked tokens have no routed demand.
    """

    @classmethod
    def prepare(cls, weights, rows, *, weight, bias, image_bias, scaling,
                scoring_func, clamp, session, name):
        """Declare row variants over one layer's residency-owned weight planes."""
        from ignition.components.megamoe import PreparedMegaMoe

        rows = tuple(rows)
        g = weights.workspace.geometry
        if (not rows or len(set(rows)) != len(rows)
                or any(type(row) is not int or not 0 < row <= g.capacity for row in rows)):
            raise ValueError("reference expert row variants must fit their shared workspace")
        device = weights.workspace.buffer.device
        example = torch.zeros(max(rows), g.hidden, dtype=torch.bfloat16, device=device)
        ids = torch.arange(g.top_k, dtype=torch.int32, device=device).expand(max(rows), -1).contiguous()
        probabilities = torch.full((max(rows), g.top_k), 1 / g.top_k, device=device)
        padding = torch.zeros(max(rows), dtype=torch.bool, device=device)
        operations = {row: PreparedMegaMoe(example[:row], ids[:row], probabilities[:row], padding[:row],
            weights.routed, weights.shared, workspace=weights.workspace, host_tier=weights.host_tier,
            host_shared_first=True, clamp=clamp, session=session, name=f"{name}.{row}") for row in rows}
        return cls(operations, weight=weight, bias=bias, image_bias=image_bias,
                   placement=weights.expert_map.reshape(-1, 1), scaling=scaling, scoring_func=scoring_func)

    def __init__(self, operations, *, weight, bias, image_bias, placement, scaling,
                 scoring_func):
        super().__init__()
        self.operations = dict(operations)
        if not self.operations:
            raise ValueError("reference experts require prepared row capacities")
        first = next(iter(self.operations.values()))
        g = first.workspace.geometry
        device = first.workspace.buffer.device
        canonical_experts = weight.shape[0]
        if not 0 < canonical_experts <= g.experts:
            raise ValueError("reference router must fit the physical expert rows")
        if scoring_func != "sqrtsoftplus" or not math.isfinite(scaling):
            raise ValueError("reference routing requires sqrtsoftplus and finite scaling")
        for rows, op in self.operations.items():
            if (rows != op.plan.query.rows or op.workspace is not first.workspace
                    or len(op.matrices) != len(first.matrices)
                    or any(a is not b for a, b in zip(op.matrices, first.matrices))
                    or op.host_tier is not first.host_tier):
                raise ValueError("reference expert capacities must share weights, tiers and workspace")
            if op.host_tier is not None and g.shared_experts and not op.plan.query.host_shared_first:
                raise ValueError("reference experts require shared-first host accumulation")
        for name, tensor, shape, dtype in (
            ("weight", weight, (canonical_experts, g.hidden), torch.bfloat16),
            ("bias", bias, (canonical_experts,), torch.float32),
            ("placement", placement, (canonical_experts, 1), torch.int32),
        ):
            if (tensor.shape != shape or tensor.dtype != dtype
                    or tensor.device != device or not tensor.is_contiguous()):
                raise ValueError(f"reference router {name} has incompatible geometry, dtype or device")
            self.register_buffer(name, tensor)
        if image_bias is not None and (
                image_bias.shape != bias.shape or image_bias.dtype != bias.dtype
                or image_bias.device != device or not image_bias.is_contiguous()):
            raise ValueError("reference image routing bias must match text routing bias geometry")
        physical = placement.detach().cpu().flatten()
        if (physical.unique().numel() != canonical_experts
                or int(physical.min()) < 0 or int(physical.max()) >= g.experts):
            raise ValueError("reference expert placement must select distinct valid physical rows")
        self.register_buffer("image_bias", image_bias)
        self.register_buffer("counts", persistent_copy(torch.ones(canonical_experts, dtype=torch.int32, device=device)))
        for tensor in (weight, bias, image_bias, placement, self.counts):
            if tensor is not None:
                record_numeric_tensor(tensor)
        self.scaling = float(scaling)

    def _operation(self, rows):
        if rows not in self.operations:
            raise ValueError(f"reference experts have no prepared operation for {rows} rows")
        return self.operations[rows]

    def with_routing_trace(self, trace, layer, *, stream=None, routes_buffer=None, valid_buffer=None):
        """Return a read-only traced view recording canonical expert IDs."""
        if trace is None:
            raise ValueError("routing instrumentation requires a prepared trace")
        return TracedReferenceExperts(self, trace, layer, stream=stream,
                                      routes_buffer=routes_buffer, valid_buffer=valid_buffer)

    def fused_norm_cast_kwargs(self, num_tokens):
        x, sf, _, _, shared_sf, block_m = self._operation(num_tokens).encoded_inputs()
        kwargs = dict(out=(x, sf), fp8_cast_params=dict(fmt="e4m3", num_per_channels=32,
            round_sf=True, use_packed_ue8m0=True, use_tma_aligned_col_major_sf=False),
            extra_sf_layout="mega_moe")
        if shared_sf is not None:
            kwargs.update(extra_sf_out=shared_sf, extra_sf_block_m=block_m)
        return kwargs

    def forward(self, hidden_states, hidden_states_bf16, input_metadata, *, expert_stage=None):
        op = self._operation(hidden_states_bf16.shape[0])
        _, _, ids, probabilities, _, _ = op.encoded_inputs()
        image_mask = getattr(input_metadata, "images_seq_mask", None)
        if image_mask is not None and self.image_bias is None:
            raise ValueError("image routing requires the checkpoint image bias")
        _gate(hidden_states_bf16, self.weight, self.bias, self.image_bias,
              input_metadata.valid_token_mask, image_mask, self.placement, self.counts,
              ids, probabilities, self.scaling)
        return op.encoded(hidden_states_bf16, hidden_states, staged=expert_stage)


class TracedReferenceExperts(nn.Module):
    """Record canonical expert demand beside an immutable ReferenceExperts binding."""

    def __init__(self, base, trace, layer, *, stream=None, routes_buffer=None, valid_buffer=None):
        super().__init__()
        first_op = next(iter(base.operations.values()))
        top_k = first_op.workspace.geometry.top_k
        ids_dtype = first_op.encoded_inputs()[2].dtype
        self.__dict__["base"] = base
        self.counters = {rows: trace.functional_counter(layer, rows, top_k, dtype=ids_dtype)
                         for rows in base.operations}
        self._async_trace = stream is not None
        self.stream_handle = 0 if stream is None else stream.cuda_stream
        self._routes_buffer = routes_buffer
        self._valid_buffer = valid_buffer

    def fused_norm_cast_kwargs(self, num_tokens):
        return self.base.fused_norm_cast_kwargs(num_tokens)

    def forward(self, hidden_states, hidden_states_bf16, input_metadata, *, expert_stage=None):
        base = self.base
        rows = hidden_states_bf16.shape[0]
        op = base._operation(rows)
        _, _, ids, probabilities, _, _ = op.encoded_inputs()
        image_mask = getattr(input_metadata, "images_seq_mask", None)
        if image_mask is not None and base.image_bias is None:
            raise ValueError("image routing requires the checkpoint image bias")
        if self._async_trace and self._routes_buffer is not None:
            routes = self._routes_buffer[:rows]
        else:
            routes = torch.empty_like(ids)
        _traced_gate(hidden_states_bf16, base.weight, base.bias, base.image_bias,
                     input_metadata.valid_token_mask, image_mask, base.placement, base.counts,
                     ids, probabilities, routes, base.scaling)
        stream_handle = self.stream_handle if self._async_trace else None
        if self._async_trace:
            valid_tokens = getattr(input_metadata, "_trace_valid_buffer", None)
            if valid_tokens is None:
                source = getattr(input_metadata, "valid_token_count", None)
                if source is None:
                    source = input_metadata.valid_token_mask.sum(dtype=torch.int32).reshape(1)
                if self._valid_buffer is not None:
                    self._valid_buffer.copy_(source)
                    valid_tokens = self._valid_buffer
                else:
                    valid_tokens = source
                    stream_handle = None
        else:
            valid_tokens = getattr(input_metadata, "valid_token_count", None)
            if valid_tokens is None:
                valid_tokens = input_metadata.valid_token_mask.sum(dtype=torch.int32).reshape(1)
        self.counters[rows](routes, valid_tokens, stream_handle=stream_handle)
        return op.encoded(hidden_states_bf16, hidden_states, staged=expert_stage)


declare_module_buffers(f"{ReferenceExperts.__module__}.{ReferenceExperts.__qualname__}",
                       ("weight", "bias", "image_bias"))
