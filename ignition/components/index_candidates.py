# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Select context blocks and translate sparse score columns to logical tokens."""

import torch
import triton

from ignition.components.mega_cache_kernel.candidates import (
    _block_scores_kernel, _store_candidates_kernel,
    _expand_candidates_kernel, _sparse_topk_remap_kernel,
)


def _lengths(lengths, rows, device):
    if (lengths.shape != (rows,) or lengths.dtype != torch.int32
            or not lengths.is_contiguous() or lengths.device != device):
        raise ValueError("candidate bounds require one contiguous int32 length per row")


def select_blocks(logits, lengths, *, count, block_tokens):
    """Select blocks by maximum score, always retaining the newest live block.

    Lengths bound request-local prefixes in logits. Empty blocks are excluded;
    the result contains unique block IDs with -1 padding. Selection retains
    the reference's torch.topk tie behavior.
    """
    if (logits.ndim != 2 or logits.device.type != "cuda" or logits.stride(1) != 1
            or logits.dtype != torch.float32 or logits.shape[0] <= 0 or logits.shape[1] <= 0
            or type(count) is not int or count <= 0 or count & (count - 1)
            or type(block_tokens) is not int or block_tokens not in (8, 16)):
        raise ValueError("candidate selection requires FP32 score rows, power-of-two capacity and 8/16-token blocks")
    rows, width = logits.shape
    _lengths(lengths, rows, logits.device)
    blocks = triton.cdiv(width, block_tokens)
    scores = torch.empty((rows, blocks), dtype=logits.dtype, device=logits.device)
    output = torch.empty((rows, count), dtype=torch.int32, device=logits.device)
    torch.library.wrap_triton(_block_scores_kernel)[(rows, triton.cdiv(blocks, 128))](
        logits, None, lengths, scores, *logits.stride(), 0, lengths.stride(0),
        width, blocks, block_tokens, False, 1, 128)
    top = scores.topk(min(count, blocks), dim=-1)
    torch.library.wrap_triton(_store_candidates_kernel)[(rows, triton.cdiv(count, 256))](
        top.values, top.indices, output, *output.stride(), top.values.shape[1], count, 256)
    return output


def expand_blocks(candidates, lengths, *, block_tokens, sparse_tokens):
    """Sort candidate blocks into the reference scorer's repeated-tail layout.

    Candidates must be unique nonnegative block IDs or -1. Blocks beyond each
    row's visible prefix are removed. The returned lengths count valid sparse
    score columns, including a partial last block; padding repeats the last
    live block and is excluded by those lengths.
    """
    if (candidates.ndim != 2 or candidates.device.type != "cuda"
            or candidates.dtype != torch.int32 or candidates.stride(1) != 1
            or candidates.shape[0] <= 0 or not 0 < candidates.shape[1] <= 4096
            or candidates.shape[1] & (candidates.shape[1] - 1)
            or type(block_tokens) is not int or type(sparse_tokens) is not int
            or sparse_tokens not in (8, 16) or block_tokens not in (sparse_tokens, 2 * sparse_tokens)):
        raise ValueError("sparse candidates require int32 power-of-two rows and an integral 1/2 block ratio")
    rows, count = candidates.shape
    _lengths(lengths, rows, candidates.device)
    ratio = block_tokens // sparse_tokens
    starts = torch.zeros_like(lengths)
    blocks = torch.empty((rows, count * ratio), dtype=torch.int32, device=candidates.device)
    ends = torch.empty_like(lengths)
    torch.library.wrap_triton(_expand_candidates_kernel)[(rows,)](
        candidates, candidates.stride(0), starts, 1, lengths, 1,
        blocks, blocks.stride(0), ends, 1, K=count, S=count * ratio,
        RATIO=ratio, SBK=sparse_tokens, num_warps=16)
    return blocks, ends


def remap_columns(columns, blocks, *, sparse_tokens):
    """Translate sparse selected columns to request-local token positions.

    Invalid columns remain -1. Selection must exclude repeated padding blocks
    using the lengths returned by expand_blocks before calling this function.
    """
    if (columns.ndim != 2 or columns.device.type != "cuda" or columns.dtype != torch.int32
            or columns.stride(1) != 1 or columns.shape[0] <= 0 or columns.shape[1] <= 0
            or blocks.ndim != 2 or blocks.shape[0] != columns.shape[0] or blocks.shape[1] <= 0
            or blocks.dtype != torch.int32 or blocks.stride(1) != 1 or blocks.device != columns.device
            or type(sparse_tokens) is not int or sparse_tokens not in (8, 16)):
        raise ValueError("sparse remapping requires matched int32 column/block rows and 8/16-token blocks")
    rows, count = columns.shape
    starts = torch.zeros(rows, dtype=torch.int32, device=columns.device)
    output = torch.empty(columns.shape, dtype=torch.int32, device=columns.device)
    torch.library.wrap_triton(_sparse_topk_remap_kernel)[(rows,)](
        columns, columns.stride(0), blocks, blocks.stride(0), starts, 1,
        output, output.stride(0), count, blocks.shape[1] * sparse_tokens,
        SBK=sparse_tokens, K_POW2=triton.next_power_of_2(count))
    return output
