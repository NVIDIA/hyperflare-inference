# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless weight layouts at the fused attention projection boundaries."""

from dataclasses import dataclass

import torch

from ignition.preparation.persistent import persistent_copy


@dataclass(frozen=True)
class FusedMLAOutputWeights:
    grouped: tuple[torch.Tensor, torch.Tensor]
    dense: object


def _validate(values, scales, shape):
    if (min(shape) <= 0 or values.ndim != len(shape) or tuple(values.shape) != shape
            or values.dtype != torch.float8_e4m3fn or not values.is_contiguous()
            or scales.shape != (*shape[:-1], shape[-1] // 32)
            or scales.dtype != torch.uint8 or not scales.is_contiguous()
            or scales.device != values.device or shape[-1] % 128):
        raise ValueError("fused attention projections require contiguous FP8 weights and per-row native UE8M0 scale bytes")


def query_planes(values, scales, *, heads=64, head_dim=512):
    """Permute output rows in 16-element head chunks, moving their scales with them.

    Input scales are expanded native bytes for each weight row and 32 columns.
    The result is a preparation intermediate for the selected projection packer.
    """
    if (type(heads) is not int or type(head_dim) is not int or heads <= 0
            or head_dim <= 0 or head_dim % 16 or values.ndim != 2):
        raise ValueError("query permutation requires positive heads and a head dimension divisible by sixteen")
    _validate(values, scales, (heads * head_dim, values.shape[1]))
    order = torch.arange(heads * head_dim, device=values.device).view(heads, head_dim // 16, 16)
    order = order.transpose(0, 1).flatten()
    return (values.view(torch.uint8)[order].view(values.dtype), scales[order])


def output_planes(values, scales, *, heads_per_group=8, head_dim=512):
    """Permute input columns in 32-element head chunks without changing quantization.

    Each group is an independent output projection. Per-32-column scale groups
    undergo the same permutation as their value chunks.
    """
    if (type(heads_per_group) is not int or type(head_dim) is not int
            or heads_per_group <= 0 or head_dim <= 0 or head_dim % 32 or values.ndim != 3):
        raise ValueError("output permutation requires positive head groups and a head dimension divisible by thirty-two")
    groups, rows, _ = values.shape
    _validate(values, scales, (groups, rows, heads_per_group * head_dim))
    chunks = torch.arange(heads_per_group * head_dim // 32, device=values.device)
    chunks = chunks.view(heads_per_group, head_dim // 32).t().flatten()
    order = (chunks[:, None] * 32 + torch.arange(32, device=values.device)).flatten()
    return values.view(torch.uint8)[:, :, order].view(values.dtype), scales[:, :, chunks]


def pack_output_planes(values, scales):
    """Pack already permuted grouped weights for direct consumption of attention FP8 output."""
    import deep_gemm

    if values.ndim != 3:
        raise ValueError("grouped output weights require [groups, rows, columns]")
    groups, rows, columns = values.shape
    _validate(values, scales, (groups, rows, columns))
    if rows % 4:
        raise ValueError("grouped output rows must satisfy the four-row scale alignment")
    # Supply native exponent bits to DeepGEMM's scale-layout transform.
    native = (scales.to(torch.int32) << 23).view(torch.float32)
    packed = deep_gemm.transform_sf_into_required_layout(native, rows, columns, (1, 32), groups)
    return persistent_copy(values), persistent_copy(packed)
