# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Warm resident projection weights on a prepared CUDA stream."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefetch_weight_l2(words, word_count, sums, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    block = tl.program_id(0)
    grid = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    partial = tl.full((BLOCK,), 0, tl.int64)
    for base in range(block * BLOCK * UNROLL, word_count, grid * BLOCK * UNROLL):
        for step in tl.static_range(UNROLL):
            indices = base + step * BLOCK + lanes
            partial += tl.load(words + indices, mask=indices < word_count, other=0)
    # Storing one sum per block keeps every weight read live.
    tl.store(sums + block, tl.sum(partial, axis=0))


def prefetch_weight_l2(words: torch.Tensor, sums: torch.Tensor, programs: int) -> None:
    """Read a contiguous int64 view; the caller supplies one int64 slot per block."""
    _prefetch_weight_l2[(programs,)](words, words.numel(), sums, BLOCK=2048, UNROLL=8, num_warps=16)


@torch.library.custom_op("flare::prefetch_weights", mutates_args=("sums",))
def _prefetch(anchor: torch.Tensor, weights: list[torch.Tensor], sums: torch.Tensor,
              programs: int, stream: int) -> None:
    auxiliary = torch.cuda.ExternalStream(stream, device=anchor.device)
    auxiliary.wait_stream(torch.cuda.current_stream(anchor.device))
    with torch.cuda.stream(auxiliary):
        for weight in weights:
            prefetch_weight_l2(weight, sums, programs)


@_prefetch.register_fake
def _fake_prefetch(anchor, weights, sums, programs, stream):
    pass


@torch.library.custom_op("flare::join_weight_prefetch", mutates_args=())
def join_weight_prefetch(anchor: torch.Tensor, sums: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(anchor.device).wait_stream(torch.cuda.ExternalStream(stream, device=anchor.device))


@join_weight_prefetch.register_fake
def _fake_join(anchor, sums, stream):
    pass


# A CUDA stream join has an ordering effect even when the model never consumes
# the values written to sums. Its input keeps that storage live until the join.
join_weight_prefetch.register_effect(torch.library.EffectType.ORDERED)


class PreparedWeightPrefetch:
    """Bind contiguous BF16 projection storage and the stream that owns the reads."""

    def __init__(self, weights, programs, stream):
        self.words = [w.view(-1).view(torch.int64) for w in weights
                      if isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16
                      and w.is_contiguous() and w.numel() % 4 == 0 and w.storage_offset() % 4 == 0]
        self.programs = programs
        self.stream = stream
        self.stream_handle = stream.cuda_stream

    def __call__(self, anchor, sums):
        if self.words:
            _prefetch(anchor, self.words, sums, self.programs, self.stream_handle)
