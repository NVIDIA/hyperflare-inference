# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling per-token four-or-six NVFP4 quantization arithmetic.

Adapted from vLLM PR #69028, ops/nvfp4.py at b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0:
https://github.com/vllm-project/vllm/blob/b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0/vllm/models/inkling2/nvidia/ops/nvfp4.py
Flare retains the Blackwell hardware E2M1 conversion path.
"""

import triton
import triton.language as tl

FP32_MAX = tl.constexpr(3.4028234663852886e38)

@triton.jit
def _pack_roundtrip(x):
    low, high = tl.split(x)
    packed, halves = tl.inline_asm_elementwise(
        "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, $3, $2; "
        "cvt.u32.u8 $0, b; cvt.rn.f16x2.e2m1x2 $1, b; }",
        "=r,=r,f,f",
        [low, high],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    low = (halves & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True)
    high = (halves >> 16).to(tl.uint16).to(tl.float16, bitcast=True)
    return packed.to(tl.uint8), tl.join(low, high).to(tl.float32)


@triton.jit
def _block_error(x, reconstructed):
    groups: tl.constexpr = x.shape[0]
    error = tl.abs(reconstructed - x)
    for width in tl.static_range(3, -1, -1):
        low, high = tl.split(tl.reshape(error, (groups, 1 << width, 2)))
        error = low + high
    return tl.reshape(error, (groups,))


@triton.jit
def _block_scales(block_max, encoder_scale, decoder_scale):
    """The 6.0- and 4.0-based block scales of blocks with max |x| `block_max`
    and their inverses."""
    scale6 = tl.minimum(block_max * (encoder_scale * (1.0 / 6.0)), 448.0)
    scale4 = tl.minimum(block_max * (encoder_scale * 0.25), 448.0)
    scale6 = scale6.to(tl.float8e4nv).to(tl.float32)
    scale4 = scale4.to(tl.float8e4nv).to(tl.float32)
    inverse6 = tl.minimum(tl.div_rn(1.0, scale6 * decoder_scale), FP32_MAX)
    inverse4 = tl.minimum(tl.div_rn(1.0, scale4 * decoder_scale), FP32_MAX)
    return scale6, scale4, inverse6, inverse4


@triton.jit
def _quantize_block(x, encoder_scale, decoder_scale, outer_scale, BLOCK: tl.constexpr):
    x = tl.reshape(x, (BLOCK // 16, 16))
    scale6, scale4, inverse6, inverse4 = _block_scales(
        tl.max(tl.abs(x), 1), encoder_scale, decoder_scale
    )
    packed6, rounded6 = _pack_roundtrip(
        tl.reshape(x * inverse6[:, None], (BLOCK // 16, 8, 2))
    )
    packed4, rounded4 = _pack_roundtrip(
        tl.reshape(x * inverse4[:, None], (BLOCK // 16, 8, 2))
    )
    reconstructed6 = tl.reshape(rounded6, (BLOCK // 16, 16)) * scale6[:, None]
    reconstructed4 = tl.reshape(rounded4, (BLOCK // 16, 16)) * scale4[:, None]
    error6 = _block_error(x, reconstructed6 * outer_scale)
    error4 = _block_error(x, reconstructed4 * outer_scale)
    choose4 = error4 < error6
    packed = tl.reshape(tl.where(choose4[:, None], packed4, packed6), (BLOCK // 2,))
    scales = tl.where(choose4, scale4, scale6).to(tl.float8e4nv)
    return packed, scales
