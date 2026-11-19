# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference query rotation/quantization and planar index-key cache production."""

import math

import torch

from ignition.components.mega_cache_kernel.indexer import (
    _indexer_k_norm_rope_quant_store_kernel, _query_rope_quant_kernel,
)


def index_cache_view(storage, page_tokens):
    """Expose separated 64-byte values and four-byte scales in aligned pages."""
    if (type(page_tokens) is not int or page_tokens not in (64, 128)
            or storage.ndim != 2 or storage.dtype != torch.uint8 or storage.stride(1) != 1
            or storage.shape[1] != (page_tokens * 68 + 511) // 512 * 512
            or storage.stride(0) < storage.shape[1] or storage.stride(0) % 512):
        raise ValueError("index cache requires 64/128-token pages with 512-byte-aligned storage")
    return storage.as_strided((storage.shape[0], page_tokens, 68), (storage.stride(0), 68, 1))


def _rotary_inputs(values, positions, cos_sin):
    rows = values.shape[0]
    if (values.device.type != "cuda" or values.dtype != torch.bfloat16
            or positions.shape != (rows,) or positions.dtype != torch.int64 or not positions.is_contiguous()
            or cos_sin.ndim != 2 or cos_sin.shape[0] <= 0 or cos_sin.shape[1] != 64
            or cos_sin.dtype != torch.float32 or not cos_sin.is_contiguous()
            or positions.device != values.device or cos_sin.device != values.device):
        raise ValueError("index rotation requires BF16 inputs, contiguous int64 positions and an FP32 cosine/sine table")


def quantize_index_queries(query, positions, cos_sin, weights, *, weight_dtype):
    """Rotate Q, preserve BF16 rounding, quantize MXFP4 and scale per-head weights.

    Positions must index the supplied table. Query scales remain separate from
    the head weights. The sparse scoring consumer requires BF16 weights; the
    full-cache scoring consumer requires FP32 weights.
    """
    _rotary_inputs(query, positions, cos_sin)
    rows = query.shape[0]
    if (query.shape != (rows, 32, 128) or not query.is_contiguous()
            or weights.shape != (rows, 32) or weights.dtype != torch.float32 or not weights.is_contiguous()
            or weights.device != query.device or weight_dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError("index query preparation requires 32 contiguous 128-element heads and FP32 source weights")
    packed = torch.empty((rows, 32, 64), device=query.device, dtype=torch.uint8)
    scales = torch.empty((rows, 32, 4), device=query.device, dtype=torch.uint8)
    folded = torch.empty((rows, 32), device=query.device, dtype=weight_dtype)
    torch.library.wrap_triton(_query_rope_quant_kernel)[(rows, 32)](
        positions, query, query.stride(0), query.stride(1), cos_sin, cos_sin.stride(0), 32,
        packed, packed.stride(0), packed.stride(1), scales, scales.stride(0), scales.stride(1),
        128, 32, weights, weights.stride(0), 128 ** -.5, 32 ** -.5, folded, folded.stride(0),
        num_warps=1)
    return packed.view(torch.int8), scales.view(torch.int32).squeeze(-1), folded


def write_index_keys(values, positions, cos_sin, norm_weight, cache, slots, *, eps, ratio):
    """Normalize/rotate projected keys and publish complete MXFP4 cache records.

    Negative slots and non-closing compression positions perform no write.
    Other slots must be unique, within the cache, and use positions within the
    rotary table. Each page holds packed values followed by scale bytes.
    """
    _rotary_inputs(values, positions, cos_sin)
    rows = values.shape[0]
    if (values.shape != (rows, 128) or values.stride(1) != 1
            or norm_weight.shape != (128,) or norm_weight.dtype not in (torch.bfloat16, torch.float32)
            or not norm_weight.is_contiguous() or norm_weight.device != values.device
            or slots.shape != (rows,) or slots.dtype != torch.int64 or not slots.is_contiguous()
            or slots.device != values.device or cache.ndim != 3 or cache.shape[0] <= 0
            or cache.shape[1] not in (64, 128) or cache.shape[2] != 68
            or cache.dtype != torch.uint8 or cache.device != values.device
            or cache.stride()[1:] != (68, 1) or cache.stride(0) % 512
            or cache.stride(0) < cache.shape[1] * 68
            or type(ratio) is not int or ratio not in (1, 2) or not math.isfinite(eps) or eps <= 0):
        raise ValueError("index-key writing requires 128-element keys, planar MXFP4 pages and matched normalization/slot metadata")
    torch.library.wrap_triton(_indexer_k_norm_rope_quant_store_kernel)[(rows,)](
        values, values.stride(0), positions, norm_weight, eps, cos_sin, cos_sin.stride(0),
        cache, slots, cache.shape[1], HEAD_SIZE=128, ROPE_HEAD_DIM=64, COMPRESS_RATIO=ratio,
        TOKEN_STRIDE=64, SCALE_DIM=4, KV_BLOCK_STRIDE=cache.stride(0), FP8_MAX=448., USE_FP4=True,
        SHUFFLE=False, BLOCK_TILE_SIZE=16, HEAD_TILE_SIZE=16, num_warps=1)
