# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose fused MLA over separated cache planes and explicit request indices."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ignition.components.flashmla_attention import PreparedFlashMLA
from ignition.components.mega_attention import MegaAttentionWorkspace, PreparedMegaAttention
from ignition.components.mega_attention_cache import MegaCacheLayout
from ignition.components.mega_cache_kernel.gather import (
    _dequantize_and_gather_k_mxfp8_kernel,
    _dequantize_and_gather_k_nvfp4_kernel,
)
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor


@dataclass(frozen=True)
class FusedMLAPreparation:
    """Model-owned rotary tables and synchronization state shared by captured variants."""

    cos_sin: tuple[torch.Tensor, ...]
    workspace: MegaAttentionWorkspace
    kernel_cache: Path
    library: Path
    deepselect_library: Path | None = None

    @classmethod
    def create(cls, frequencies, context, *, kernel_cache, library, deepselect_library=None):
        frequencies = tuple(frequencies)
        if (type(context) is not int or context <= 0 or not frequencies
                or any(t.shape != (32,) or t.dtype != torch.float32
                       or t.device != frequencies[0].device for t in frequencies)):
            raise ValueError("fused MLA requires positive context and matching 32-element FP32 rotary frequencies")
        tables = []
        positions = torch.arange(context, device=frequencies[0].device, dtype=torch.float32)
        for frequency in frequencies:
            angles = positions[:, None] * frequency[None, :]
            table = persistent_copy(torch.cat((angles.cos(), angles.sin()), -1))
            record_numeric_tensor(table)
            tables.append(table)
        return cls(tuple(tables), MegaAttentionWorkspace(76, device=frequencies[0].device),
                   Path(kernel_cache), Path(library).resolve(strict=True),
                   None if deepselect_library is None else Path(deepselect_library).resolve(strict=True))


def gather_cache(cache, lengths, pages, *, output):
    """Gather logical request prefixes with the reference cache dequantization kernels.

    Lengths are in [0, output.shape[1]]; page entries address cache pages. The caller
    supplies those admission bounds. Output rows are contiguous BF16 vectors;
    request strides may include other key regions. Only live prefixes are written.
    """
    if cache.ndim != 3:
        raise ValueError("cache gathering requires a separated paged cache")
    layout = MegaCacheLayout(cache.shape[1], cache.shape[2])
    layout.validate(cache)
    if output.ndim != 3:
        raise ValueError("cache gather output must have request, token, and channel dimensions")
    capacity = output.shape[1]
    if (capacity <= 0 or pages.ndim != 2
            or pages.dtype != torch.int32 or not pages.is_contiguous()
            or lengths.shape != (pages.shape[0],) or lengths.dtype != torch.int32
            or not lengths.is_contiguous() or lengths.device != cache.device or pages.device != cache.device
            or capacity > pages.shape[1] * layout.page_tokens
            or pages.shape[0] * capacity * 512 >= 2**31):
        raise ValueError("cache gather capacity and contiguous int32 request metadata must fit the page table and reference offsets")
    if (output.shape != (pages.shape[0], capacity, 512) or output.dtype != torch.bfloat16
            or output.device != cache.device or output.stride(2) != 1 or output.stride(1) != 512
            or output.stride(0) < capacity * 512
            or (pages.shape[0] - 1) * output.stride(0) + capacity * 512 >= 2**31):
        raise ValueError("cache gather output requires disjoint BF16 request regions and int32-safe row offsets")
    kernel = (_dequantize_and_gather_k_mxfp8_kernel if layout.record_bytes == 528
              else _dequantize_and_gather_k_nvfp4_kernel)
    workers = max(128, min(8192, (capacity + 7) // 8))
    options = {"use_fnuz": False} if layout.record_bytes == 528 else {}
    torch.library.wrap_triton(kernel)[(pages.shape[0], workers)](
        output, output.stride(0), output.stride(1), cache, lengths, pages, 0, None,
        pages.shape[1], 512, layout.scale_bytes, 32 if layout.record_bytes == 528 else 16,
        layout.page_tokens, cache.stride(0), **options)
    return output


def physical_indices(logical, page_table, page_tokens):
    """Translate selected logical keys; negative entries stay masked."""
    page = torch.gather(page_table, 1, (logical.long().clamp_min(0) // page_tokens).clamp_max(page_table.shape[1] - 1))
    return torch.where(logical >= 0, page * page_tokens + logical % page_tokens, -1).to(torch.int32)


def combined_indices(window, window_lengths, selected, selected_lengths):
    """Pack selected keys followed by visible window keys without holes in the live prefix."""
    offsets = torch.arange(selected.shape[1] + window.shape[1], device=window.device)[None, :]
    selected_part = torch.gather(selected, 1, offsets.expand(window.shape[0], -1).clamp_max(selected.shape[1] - 1))
    window_part = torch.gather(window, 1, (offsets - selected_lengths[:, None]).clamp(0, window.shape[1] - 1))
    lengths = selected_lengths + window_lengths
    ids = torch.where(offsets < selected_lengths[:, None], selected_part, window_part)
    return torch.where(offsets < lengths[:, None], ids, -1).to(torch.int32), lengths


class PreparedFusedMLA:
    """Select reference kernels from physical geometry before capture.

    Small eligible decode calls use MegaAttention. Other decode calls use
    paged FlashMLA; prefill gathers BF16 keys for FlashMLA's sparse prefill.
    Query heads carry the reference permutation. Outputs are packed FP8 planes.
    """

    def __init__(self, query, positions, sink, window, indices, lengths, metadata, *,
                 preparation, rotary, mode, compressed=None, selected=None, selected_lengths=None,
                 session, name, examples=None):
        self.preparation, self.cs = preparation, preparation.cos_sin[rotary]
        self.compressed, self.mode = compressed, mode
        self.requests, self.window_tokens = metadata.starts.numel(), window.shape[1]
        self.compressed_tokens = 0 if compressed is None else compressed.shape[1]
        self.compressed_capacity = metadata.page_table.shape[1] * self.compressed_tokens
        self.prefill = mode == "prefill"
        if self.prefill:
            self.key_capacity = self.compressed_capacity + self.window_tokens
            kv = declaration_tensor(examples, "fused_mla.kv", (self.requests * self.key_capacity, 1, 512),
                                    torch.bfloat16, query.device)
            width = indices.shape[1] + (0 if selected is None else selected.shape[1])
            ids = declaration_tensor(examples, "fused_mla.indices", (query.shape[0], 1, width),
                                     torch.int32, query.device).fill_(-1)
            self.operator = PreparedFlashMLA(query, positions, self.cs, sink, kv, ids, lengths,
                scale=512 ** -.5, kernel_cache=preparation.kernel_cache, library=preparation.library,
                session=session, name=name)
        else:
            ci = None if selected is None else torch.full_like(selected, -1)
            rows = query.shape[0]
            eligible = ((compressed is None and rows <= 24) or (compressed is not None and rows <= 16)
                or (compressed is not None and rows <= 76 and indices.shape[1] <= 128
                    and not (window.shape[1] & (window.shape[1] - 1))
                    and not (compressed.shape[1] & (compressed.shape[1] - 1))))
            options = dict(scale=512 ** -.5, kernel_cache=preparation.kernel_cache, session=session, name=name)
            if eligible:
                self.operator = PreparedMegaAttention(query, positions, self.cs, sink, window, indices, lengths,
                    compressed, ci, selected_lengths, workspace=preparation.workspace, **options)
            else:
                self.operator = PreparedFlashMLA(query, positions, self.cs, sink, window, indices, lengths,
                    compressed, ci, selected_lengths, library=preparation.library, **options)
        self.sink = sink

    def __call__(self, query, positions, window, indices, lengths, metadata, *, selected=None, selected_lengths=None):
        if not self.prefill:
            ci = None if selected is None else physical_indices(selected, metadata.token_page_table, self.compressed_tokens)
            return self.operator(query, positions, self.cs, self.sink, window, indices, lengths,
                                 self.compressed, ci, selected_lengths)
        window_pages = torch.arange(self.requests, device=query.device, dtype=torch.int32).view(-1, 1)
        window_lengths = torch.full((self.requests,), self.window_tokens, device=query.device, dtype=torch.int32)
        kv = torch.empty((self.requests, self.key_capacity, 512), device=query.device, dtype=torch.bfloat16)
        gather_cache(window, window_lengths, window_pages, output=kv[:, self.compressed_capacity:])
        ids = torch.where(indices >= 0,
            metadata.requests[:, None] * self.key_capacity + self.compressed_capacity + indices % self.window_tokens, -1)
        if self.compressed is not None:
            ratio = 128 // self.compressed_tokens
            counts = ((metadata.starts + metadata.lengths) // ratio).to(torch.int32)
            gather_cache(self.compressed, counts, metadata.page_table, output=kv[:, :self.compressed_capacity])
            ci = torch.where(selected >= 0, selected + metadata.requests[:, None] * self.key_capacity, -1)
            ids, lengths = combined_indices(ids, lengths, ci, selected_lengths)
        return self.operator(query, positions, self.cs, self.sink, kv.view(-1, 1, 512), ids.to(torch.int32).unsqueeze(1), lengths)
