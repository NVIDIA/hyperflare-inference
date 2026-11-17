# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference candidate block selection and sparse-column translation kernels."""

import triton
import triton.language as tl

_INT32_BIG = tl.constexpr(2**31 - 1)


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where(
        (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
        float("inf"),
        reduced,
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)

@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )

@triton.jit(do_not_specialize_on_alignment=["end_ptr"])
def _expand_candidates_kernel(
    cand_ptr,
    cand_stride,
    ks_ptr,
    ks_stride,
    ke_ptr,
    ke_stride,
    si_ptr,
    si_stride,
    end_ptr,
    end_stride,
    K: tl.constexpr,
    S: tl.constexpr,
    RATIO: tl.constexpr,
    SBK: tl.constexpr,
):
    """One program per row: expand + filter + sort candidate blocks.

    Each row's valid sparse blocks come out sorted ascending (no dedup pass:
    production candidates are unique, and the kernel tolerates repeats),
    padded by repeating the last valid block — DeepGEMM's padding convention.
    ``end`` counts valid sparse-token columns; valid columns form a prefix
    because block ids are sorted.
    """
    row = tl.program_id(0)
    ks = tl.load(ks_ptr + row * ks_stride)
    ke = tl.load(ke_ptr + row * ke_stride)
    cols = tl.arange(0, K)
    cand = tl.load(cand_ptr + row * cand_stride + cols)  # K is pow2, no mask
    if RATIO == 2:
        cand = tl.interleave(cand * RATIO, cand * RATIO + 1)

    # Keep only blocks inside the row's KV range [0, ke - ks); the lower
    # bound is free since candidates are non-negative.
    kv_len = tl.maximum(ke - ks, 0)
    max_block = (kv_len + SBK - 1) // SBK
    valid = (cand >= 0) & (cand < max_block)
    vals = tl.where(valid, cand, _INT32_BIG)
    vals = tl.sort(vals)

    # Anchor at the row's ks rounded down to a sparse-block boundary
    # (identity for the paged path, where ks == 0).
    base = ks // SBK
    si = tl.where(vals != _INT32_BIG, vals, 0) + base
    n_valid = tl.sum(valid.to(tl.int32))
    last_rel = tl.max(tl.where(valid, cand, -1))
    pad_val = tl.where(n_valid > 0, last_rel + base, base)
    tl.store(
        si_ptr + row * si_stride + tl.arange(0, S),
        tl.where(tl.arange(0, S) < n_valid, si, pad_val),
    )

    # Valid sparse-token column count; only the last block can be partial.
    # Absolute start of the last block: (base + last_rel) * SBK + ks % SBK,
    # which is exactly last_rel * SBK + ks.
    last_start = last_rel * SBK + ks
    fill = tl.minimum(tl.maximum(ke - last_start, 1), SBK)
    end = tl.where(n_valid > 0, (n_valid - 1) * SBK + fill, 0)
    tl.store(end_ptr + row * end_stride, end)

@triton.jit
def _sparse_topk_remap_kernel(
    col_ptr,
    col_stride,
    si_ptr,
    si_stride,
    ks_ptr,
    ks_stride,
    out_ptr,
    out_stride,
    k,
    width,
    SBK: tl.constexpr,
    K_POW2: tl.constexpr,
):
    """Remap sparse top-k columns to request-local positions.

    Column ``j * SBK + o`` scores the token at
    ``(si[row, j] - ks // SBK) * SBK + o``; columns outside ``[0, width)``
    (including DeepSelect's sentinel for NaN rows) become -1.
    """
    row = tl.program_id(0)
    ks_base = tl.load(ks_ptr + row * ks_stride) // SBK
    cols = tl.arange(0, K_POW2)
    c = tl.load(col_ptr + row * col_stride + cols, mask=cols < k, other=-1)
    valid = (c >= 0) & (c < width)
    cc = tl.where(valid, c, 0)
    blocks = tl.load(si_ptr + row * si_stride + cc // SBK)
    pos = (blocks - ks_base) * SBK + cc % SBK
    tl.store(out_ptr + row * out_stride + cols, tl.where(valid, pos, -1), mask=cols < k)
