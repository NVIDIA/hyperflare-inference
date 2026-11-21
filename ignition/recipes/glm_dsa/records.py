# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Latent cache records and rotary positions."""

import torch
import triton
import triton.language as tl

from .config import LATENT_RANK, LATENT_RECORD_BYTES, ROTARY_DIM

# Record fields as offsets in elements of their own dtype: E4M3 latent, FP32 scales, BF16 rotary key.
_SCALE_OFFSET = LATENT_RANK // 4
_ROTARY_OFFSET = (LATENT_RANK + 4 * (LATENT_RANK // 128)) // 2


@triton.jit
def _write_records(Latent, Rotary, Slots, Values, Scales, Keys, latent_stride, rotary_stride,
                   RECORD: tl.constexpr, SCALE_OFFSET: tl.constexpr, ROTARY_OFFSET: tl.constexpr):
    # One record per row: 512 latent values quantized to E4M3 with one FP32 scale per 128 values, then the
    # rotary key unchanged. A negative slot skips its row.
    row = tl.program_id(0)
    slot = tl.load(Slots + row).to(tl.int64)
    if slot >= 0:
        column = tl.arange(0, 512)
        values = tl.reshape(tl.load(Latent + row * latent_stride + column).to(tl.float32), (4, 128))
        peak = tl.max(tl.abs(values), axis=1)
        scale = tl.where(peak > 0, peak / 448.0, 1.0)
        scaled = tl.minimum(tl.maximum(values / scale[:, None], -448.0), 448.0)
        tl.store(Values + slot * RECORD + column, tl.reshape(scaled, (512,)).to(tl.float8e4nv))
        tl.store(Scales + slot * (RECORD // 4) + SCALE_OFFSET + tl.arange(0, 4), scale)
        pair = tl.arange(0, 64)
        tl.store(Keys + slot * (RECORD // 2) + ROTARY_OFFSET + pair, tl.load(Rotary + row * rotary_stride + pair))


def _launch(kernel, grid, args, **meta):
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    if torch.compiler.is_compiling() or get_proxy_mode() is not None:
        torch.library.wrap_triton(kernel)[grid](*args, **meta)
    else:
        kernel[grid](*args, **meta)


def write_records(latent, key_rotary, slots, cache):
    """Write one 656-byte record per row into `cache` [records..., 656] uint8 at `slots`.

    `latent` is BF16 [rows, 512] and `key_rotary` BF16 [rows, 64], already rotated. The bytes follow
    b12x's `pack_mla_kv_cache_reference`: E4M3 latents, four FP32 group scales, and the BF16 key.
    """
    records = cache.view(-1, LATENT_RECORD_BYTES)
    if (latent.dtype != torch.bfloat16 or key_rotary.dtype != torch.bfloat16 or records.dtype != torch.uint8
            or latent.shape[1] != LATENT_RANK or key_rotary.shape[1] != ROTARY_DIM or not records.is_contiguous()):
        raise ValueError("latent records require BF16 [rows,512] and [rows,64] inputs and contiguous uint8 storage")
    _launch(_write_records, (latent.shape[0],), (latent, key_rotary, slots, records.view(torch.float8_e4m3fn),
            records.view(torch.float32), records.view(torch.bfloat16), latent.stride(0), key_rotary.stride(0)),
            RECORD=LATENT_RECORD_BYTES, SCALE_OFFSET=_SCALE_OFFSET, ROTARY_OFFSET=_ROTARY_OFFSET, num_warps=4)


def rotary_frequencies(base, device):
    """The 32 rotary frequencies of the 64 rotated values, FP32."""
    exponents = torch.arange(0, ROTARY_DIM, 2, dtype=torch.float32, device=device) / ROTARY_DIM
    return 1.0 / (base ** exponents)


def rotary_angles(positions, frequencies):
    """Cosine and sine of every row's angles, FP32 [rows, 32]. Inactive rows use position zero."""
    angles = positions.clamp_min(0).to(torch.float32)[:, None] * frequencies
    return angles.cos(), angles.sin()


def rotate(values, cos, sin):
    """Rotate interleaved pairs of the last axis, BF16 [rows, 64] or [rows, heads, 64].

    Cosine, sine, and each product round to BF16, as they do in the checkpoint's rotary embedding.
    """
    if values.ndim == 3:
        cos, sin = cos[:, None], sin[:, None]
    cos, sin = cos.to(values.dtype), sin.to(values.dtype)
    a, b = values[..., 0::2], values[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)
