# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import triton
import triton.language as tl


@triton.jit
def _dequantize_and_gather_k_mxfp8_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    head_dim: tl.constexpr,  # 512
    scale_dim: tl.constexpr,  # 16
    quant_block: tl.constexpr,  # 32
    cache_block_size: tl.constexpr,
    block_stride: tl.constexpr,
    use_fnuz: tl.constexpr = False,
):
    """Gather V4.1 MXFP8 rows into a bf16 workspace, dequantizing in place."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    d = tl.arange(0, head_dim)
    s = tl.arange(0, scale_dim)
    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # int64: physical_block_idx * block_stride can exceed 2^31.
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        token_data_ptr = cache_block_ptr + pos_in_block * head_dim
        token_scale_ptr = (
            cache_block_ptr + cache_block_size * head_dim + pos_in_block * scale_dim
        )

        x_uint8 = tl.load(token_data_ptr + d)
        if use_fnuz:
            x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
        else:
            x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        tiles = tl.reshape(x_fp8.to(tl.float32), (scale_dim, quant_block))

        # UE8M0: scale = 2^(stored_value - 127).
        encoded = tl.load(token_scale_ptr + s)
        scale = tl.exp2(encoded.to(tl.float32) - 127.0)
        dequant = tl.reshape(tiles * tl.reshape(scale, (scale_dim, 1)), (head_dim,))

        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1
        tl.store(output_row_ptr + d, dequant.to(tl.bfloat16))



@triton.jit
def _dequantize_and_gather_k_nvfp4_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    head_dim: tl.constexpr,  # 512
    scale_dim: tl.constexpr,  # 32
    quant_block: tl.constexpr,  # 16
    cache_block_size: tl.constexpr,
    block_stride: tl.constexpr,
):
    """Gather and dequantize V4.1 NVFP4 rows into a bf16 workspace."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    packed_bytes: tl.constexpr = head_dim // 2
    d = tl.arange(0, head_dim)
    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        physical_block_idx = tl.load(
            block_table_ptr + batch_idx * max_blocks_per_seq + block_in_seq
        )
        page = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        packed = tl.load(
            page + pos_in_block * packed_bytes + tl.arange(0, packed_bytes)
        )
        # Even element in the low nibble, odd in the high nibble.
        codes = tl.interleave((packed & 0xF).to(tl.int32), (packed >> 4).to(tl.int32))
        # e2m1 magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
        mag_code = codes & 7
        e = (mag_code >> 1).to(tl.float32)
        m = (mag_code & 1).to(tl.float32)
        mag = tl.where(mag_code < 2, m * 0.5, (1.0 + m * 0.5) * tl.exp2(e - 1.0))
        vals = tl.where(codes >= 8, -mag, mag)

        sf = tl.load(
            page
            + cache_block_size * packed_bytes
            + pos_in_block * scale_dim
            + tl.arange(0, scale_dim)
        )
        scale = sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tiles = tl.reshape(vals, (scale_dim, quant_block))
        dequant = tl.reshape(tiles * tl.reshape(scale, (scale_dim, 1)), (head_dim,))

        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1
        tl.store(output_row_ptr + d, dequant.to(tl.bfloat16))
