# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Normalize GLM latent projections and write FP8 latent and index cache records."""

import torch
import triton
import triton.language as tl
from b12x.attention.dsa_indexer._glm_pool_kernels import _write_pool
from triton.language.extra.cuda import libdevice


@triton.jit
def _normalized_bf16(x, weight, width: tl.constexpr, eps: tl.constexpr):
    dim = tl.arange(0, width)
    values = tl.load(x + dim).to(tl.float32)
    inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / width + eps)
    normalized = (values * inverse_rms).to(tl.bfloat16).to(tl.float32)
    return (normalized * tl.load(weight + dim).to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _norm_and_store(projected, query_weight, latent_weight, angles, slots, cache, query,
                    row_stride, page_stride, record_stride, capacity,
                    EPS: tl.constexpr, WRITE_QUERY: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    source = projected + row * row_stride
    if WRITE_QUERY and tl.program_id(1) == 1:
        result = _normalized_bf16(source, query_weight, 2048, EPS)
        tl.store(query + row * 2048 + tl.arange(0, 2048), result)
    else:
        slot = tl.load(slots + row).to(tl.int64)
        if slot >= 0 and slot < capacity:
            record = cache + (slot // 64) * page_stride + (slot % 64) * record_stride
            latent = _normalized_bf16(source + 2048, latent_weight, 512, EPS).to(tl.float32)
            groups = tl.reshape(latent, (4, 128))
            maximum = tl.max(tl.abs(groups), axis=1)
            scales = tl.where(maximum == 0., 1., maximum * (1. / 448.))
            # The b12x GLM latent format uses an approximate reciprocal after computing its FP32 scales.
            inverse = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f",
                                                args=[scales], dtype=tl.float32, is_pure=True, pack=1)
            values = tl.minimum(tl.maximum(groups * inverse[:, None], -448.), 448.)
            tl.store(record.to(tl.pointer_type(tl.float8e4nv)) + tl.arange(0, 512), tl.reshape(values, (512,)))
            tl.store((record + 512).to(tl.pointer_type(tl.float32)) + tl.arange(0, 4), scales)
            pair = 2 * tl.arange(0, 32)
            key = source + 2048 + 512
            a = tl.load(key + pair).to(tl.float32)
            b = tl.load(key + pair + 1).to(tl.float32)
            cos = tl.load(angles + row * 64 + pair)
            sin = tl.load(angles + row * 64 + pair + 1)
            rotated = (record + 528).to(tl.pointer_type(tl.bfloat16)) + pair
            tl.store(rotated, a * cos - b * sin)
            tl.store(rotated + 1, a * sin + b * cos)


def store_glm_latent(projected, query_weight, latent_weight, angles, slots, cache, eps, *, query=None):
    """Write 656-byte active cache records and optionally return the normalized query in caller storage."""
    torch.library.wrap_triton(_norm_and_store)[(projected.shape[0], 2 if query is not None else 1)](
        projected, query_weight, latent_weight, angles, slots, cache, query,
        projected.stride(0), cache.stride(0), cache.stride(1), cache.shape[0] * 64,
        EPS=eps, WRITE_QUERY=query is not None, num_warps=8)


@triton.jit
def _normalize_and_store_index(keys, weights, bias, angles, slots, cache,
                               row_stride, page_stride, capacity, EPS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots + row)
    if slot >= 0 and slot < capacity:
        dim = tl.arange(0, 128)
        x = tl.load(keys + row * row_stride + dim).to(tl.float32)
        centered = x - tl.sum(x, axis=0) / 128
        variance = tl.sum(centered * centered, axis=0) / 128
        normalized = centered * tl.rsqrt(variance + EPS)
        # Round the affine transform to BF16 before exchanging the rotary and unrotated halves.
        scaled = libdevice.mul_rn(normalized, tl.load(weights + dim).to(tl.float32))
        key = (scaled + tl.load(bias + dim).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        stored_dim = dim ^ 64
        partner = tl.gather(key, dim ^ 1, axis=0)
        cos = tl.load(angles + row * 64 + (dim & 62))
        sin = tl.load(angles + row * 64 + (dim & 62) + 1)
        even = (dim & 1) == 0
        a = tl.where(even, key, partner)
        b = tl.where(even, partner, key)
        rotated = tl.where(even, a * cos - b * sin, a * sin + b * cos)
        values = tl.where(dim < 64, rotated, key)
        _write_pool(cache.to(tl.pointer_type(tl.float8e4nv)), cache.to(tl.pointer_type(tl.float32)),
                    slot, values, stored_dim, dim < 128, True,
                    page_stride, 64, 128, 448., HADAMARD=False, POWER_OF_TWO_SCALE=False)


def store_glm_index_key(keys, weights, bias, angles, slots, cache, eps):
    """Normalize and rotate the key into the checkpoint’s planar 64-record index layout."""
    torch.library.wrap_triton(_normalize_and_store_index)[(keys.shape[0],)](
        keys, weights, bias, angles, slots, cache,
        keys.stride(0), cache.stride(0), cache.shape[0] * 64, EPS=eps, num_warps=4)
