# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose reference query preparation, phase-specific scoring and selection."""

import torch

from ignition.components.index_candidates import select_blocks, expand_blocks, remap_columns, mask_blocks
from ignition.components.index_topk import PreparedIndexTopK
from ignition.components.mxfp4_index_inputs import index_cache_view, quantize_index_queries
from ignition.components.mxfp4_logits import (
    PreparedPagedMXFP4Logits, PreparedSparsePagedMXFP4Logits, PreparedContiguousMXFP4Logits,
)
from ignition.preparation.examples import declaration_tensor


def _page_rows(pages):
    # A singleton slice of an expanded table is contiguous with row stride zero.
    # The sparse scheduler also uses that stride as the logical page bound.
    if pages.stride(0) < pages.shape[1]:
        return pages.clone(memory_format=torch.contiguous_format)
    return pages.contiguous()


class PreparedReferenceIndexer:
    """Select 512 logical context tokens, optionally publishing or using blocks.

    Candidate state contains request-local block IDs. Decode scores paged
    cache storage; prefill scores shared, losslessly gathered keys in FP32
    and applies candidate masks before per-row selection.
    """

    def __init__(self, query, cache, pages, lengths, *, page_tokens, cos_sin,
                 candidate_mode, candidate_count, block_tokens, prefill,
                 kernel_cache, library, session, name, examples=None, prefill_keys=None):
        if candidate_mode not in ("none", "publish", "restrict"):
            raise ValueError("index candidate mode must be none, publish or restrict")
        if (query.ndim != 3 or query.shape[1:] != (32, 128) or query.dtype != torch.bfloat16
                or candidate_count != 2048 or block_tokens != 8):
            raise ValueError("reference index composition requires 32 query heads and 2048 eight-token candidate blocks")
        self.cache = index_cache_view(cache, page_tokens)
        self.cos_sin = cos_sin
        self.candidate_mode, self.candidate_count, self.block_tokens = candidate_mode, candidate_count, block_tokens
        self.prefill = prefill
        if prefill != (prefill_keys is not None):
            raise ValueError("reference prefill requires contiguous key and scale examples")
        sparse = candidate_mode == "restrict" and not prefill
        self.weight_dtype = torch.bfloat16 if sparse else torch.float32
        rows, device = query.shape[0], query.device

        def example(role, shape, dtype):
            return declaration_tensor(examples, "reference_index." + role, shape, dtype, device)

        packed = example("packed", (rows, 32, 64), torch.int8)
        scales = example("scales", (rows, 32), torch.int32)
        weights = example("weights", (rows, 32), self.weight_dtype)
        if not prefill:
            pages = _page_rows(pages)
        if prefill:
            keys, key_scales = prefill_keys
            columns = keys.shape[0]
            starts = example("starts", (rows,), torch.int32)
            self.scoring = PreparedContiguousMXFP4Logits(packed, scales, keys, key_scales,
                weights, starts, lengths, kernel_cache=kernel_cache, session=session, name=name + ".scores")
        elif sparse:
            blocks = example("blocks", (rows, candidate_count), torch.int32)
            requests = example("requests", (rows,), torch.int32)
            self.scoring = PreparedSparsePagedMXFP4Logits(packed, scales, self.cache,
                weights, lengths, pages, requests, blocks, block_tokens=block_tokens,
                kernel_cache=kernel_cache, session=session, name=name + ".scores")
            columns = candidate_count * block_tokens
        else:
            columns = pages.shape[1] * page_tokens
            self.scoring = PreparedPagedMXFP4Logits(packed, scales, self.cache,
                weights, lengths, pages, columns=columns, kernel_cache=kernel_cache,
                session=session, name=name + ".scores")
        alignment = 512 if sparse else 256
        stride = (columns + alignment - 1) // alignment * alignment + (256 if prefill else 0)
        scores = example("scores", (rows, stride), self.weight_dtype)[:, :columns]
        self.selection = PreparedIndexTopK(scores, lengths, prefill=prefill,
            kernel_cache=kernel_cache, library=library, session=session, name=name + ".topk")
        self.plans = (self.scoring.plan, self.selection.plan)

    def __call__(self, query, weights, pages, lengths, positions, requests, *, candidates=None,
                 prefill_keys=None, starts=None):
        packed, scales, weights = quantize_index_queries(query, positions, self.cos_sin,
            weights.float().contiguous(), weight_dtype=self.weight_dtype)
        if self.prefill:
            if prefill_keys is None or starts is None:
                raise ValueError("reference prefill requires gathered keys and packed interval starts")
            scores = self.scoring(packed, scales, *prefill_keys, weights, starts, lengths)
            if self.candidate_mode == "restrict":
                if candidates is None or candidates.shape != (query.shape[0], self.candidate_count):
                    raise ValueError("restricted prefill requires the published candidate block rows")
                mask_blocks(scores, lengths, candidates, block_tokens=self.block_tokens, starts=starts)
            selected = self.selection(scores, lengths, starts=starts)
            if self.candidate_mode == "publish":
                blocks = select_blocks(scores, lengths, count=self.candidate_count,
                                       block_tokens=self.block_tokens, starts=starts)
                return selected, blocks, (blocks >= 0).sum(-1, dtype=torch.int32)
            return selected
        pages = _page_rows(pages)
        if self.candidate_mode == "restrict":
            if candidates is None or candidates.shape != (query.shape[0], self.candidate_count):
                raise ValueError("restricted index scoring requires the published candidate block rows")
            blocks, ends = expand_blocks(candidates, lengths,
                block_tokens=self.block_tokens, sparse_tokens=self.block_tokens)
            scores = self.scoring(packed, scales, self.cache, weights, lengths, pages,
                                  requests.to(torch.int32).contiguous(), blocks)
            selected = self.selection(scores, ends)
            return remap_columns(selected, blocks, sparse_tokens=self.block_tokens)
        scores = self.scoring(packed, scales, self.cache, weights, lengths, pages)
        selected = self.selection(scores, lengths)
        if self.candidate_mode == "publish":
            blocks = select_blocks(scores, lengths, count=self.candidate_count, block_tokens=self.block_tokens)
            return selected, blocks, (blocks >= 0).sum(-1, dtype=torch.int32)
        return selected
