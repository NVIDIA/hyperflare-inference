# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import triton
import triton.language as tl


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _rope_quant_insert_nvfp4_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
):
    """V4.1 NVFP4 record: RoPE, then e2m1 with one e4m3 scale per 16 dims.

    The scale is ``amax / 6`` (6 is e2m1's largest magnitude) clamped to the
    e4m3 range, with no per-tensor scale on top.
    """
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)

    # NoPE pairs load (cos, sin) = (1, 0), so the rotation is the identity there.
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * s, odd * c + even * s)
    if SANITIZE_CACHE_NANS:
        rotated = tl.where(rotated == rotated, rotated, 0.0)

    tiles = tl.reshape(rotated, (32, 16))
    amax = tl.max(tl.abs(tiles), 1)
    # 2**-9 is the smallest normal e4m3 magnitude; 448 the largest.
    scale = tl.clamp(amax * (1.0 / 6.0), 0.001953125, 448.0).to(tl.float8e4nv)
    # Round-to-nearest division: Triton's default div.full misplaces values
    # that land exactly on an e2m1 tie.
    scaled = tl.math.div_rn(tiles, tl.reshape(scale.to(tl.float32), (32, 1)))
    lo, hi = tl.split(tl.reshape(tl.reshape(scaled, (512,)), (256, 2)))

    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    tl.store(
        page + (slot % CACHE_BLOCK) * 256 + tl.arange(0, 256),
        _fp32x2_to_fp4x2(lo, hi),
    )
    scales = page + CACHE_BLOCK * 256 + (slot % CACHE_BLOCK) * 32
    tl.store(scales + tl.arange(0, 32), scale.to(tl.uint8, bitcast=True))
