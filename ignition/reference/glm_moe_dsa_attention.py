# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent arithmetic diagnostics for group-scaled GLM sparse attention.

The SM120 eight-head decoder quantizes absorbed queries in 128-value groups,
uses two E4M3 terms for probability/value products, and rounds normalized
split outputs to BF16 before combining them. These operations are separate
from quantization of the latent cache. The reference evaluates these operations independently on supplied
operands. Its numerical conclusions apply to this attention operation.
"""

from dataclasses import dataclass

import torch

from .glm_moe_dsa_cache import quantize_index


@dataclass(frozen=True)
class SparseAttentionArithmetic:
    """Exact-operand FP64 scores and explicit lower-precision reduction results.

    `dense` uses the independently rounded FP8 query and decoded cache with
    FP64 attention arithmetic. `split_partials` contains the two-term FP8 PV
    results before BF16 storage. `combined` includes that storage rounding
    and FP32 combination. Rounding that result to BF16 gives `output`.
    `split_probabilities` gives each split's contribution to the softmax.
    """

    scores: torch.Tensor
    dense: torch.Tensor
    split_partials: torch.Tensor
    split_probabilities: torch.Tensor
    combined: torch.Tensor
    output: torch.Tensor


@torch.no_grad()
def sparse_attention_arithmetic(query, records, selected, *, chunks_per_split, softmax_scale):
    """Evaluate one row's selected records without importing native arithmetic.

    The BF16 `query` has shape `[heads, 576]` with 64 trailing rotary coordinates.
    `records` contains 656-byte latent records. `selected` preserves the
    native candidate order and may contain -1 padding. A chunk contains 64
    candidates. The caller supplies the prepared plan's chunks per split because
    split geometry determines where intermediate results round to BF16.

    Score dots use FP64 on stored E4M3 values and FP32 scales. The softmax
    recurrence, two-term PV, and split combination use PyTorch FP32 arithmetic.
    This models their rounding boundaries, not the native instruction order
    or approximate exponent/logarithm instructions. Those differences require
    a separate arithmetic bound.
    """
    if query.ndim != 2 or query.shape[1] != 576 or query.dtype != torch.bfloat16:
        raise ValueError("GLM attention reference requires BF16 [heads, 576] queries")
    if records.ndim != 2 or records.shape[1] != 656 or records.dtype != torch.uint8:
        raise ValueError("GLM attention reference requires 656-byte latent records")
    if selected.ndim != 1 or selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("GLM attention reference requires a vector of selected record indices")
    if type(chunks_per_split) is not int or chunks_per_split < 1:
        raise ValueError("GLM attention reference requires positive chunks per split")
    if not isinstance(softmax_scale, (int, float)) or not 0 < softmax_scale < float("inf"):
        raise ValueError("GLM attention reference requires a finite positive softmax scale")
    if not len(records) or not len(selected) or bool(((selected < -1) | (selected >= len(records))).any()):
        raise ValueError("GLM attention selections must name existing records or -1 padding")
    if len({query.device, records.device, selected.device}) != 1:
        raise ValueError("GLM attention reference operands must share a device")

    valid = selected >= 0
    selected_records = records[selected.clamp_min(0).long()]
    keys = selected_records[:, :512].contiguous().view(torch.float8_e4m3fn).float().reshape(-1, 4, 128)
    key_scales = selected_records[:, 512:528].contiguous().view(torch.float32)
    rotary = selected_records[:, 528:].contiguous().view(torch.bfloat16)
    qbytes, qscales = quantize_index(query[:, :512].reshape(-1, 4, 128), query=True)
    qvalues = qbytes.view(torch.float8_e4m3fn).double()
    scores = torch.einsum("hgd,sgd->hsg", qvalues, keys.double())
    scores = (scores * qscales.double()[:, None, :] * key_scales.double()[None]).sum(-1)
    scores += query[:, 512:].double() @ rotary.double().T
    scores *= softmax_scale
    scores[:, ~valid] = -torch.inf
    latent = (keys.double() * key_scales.double()[..., None]).flatten(1)
    dense = (scores.softmax(-1) @ latent if bool(valid.any())
             else torch.zeros((len(query), 512), dtype=torch.float64, device=query.device))
    partials, partitions = [], []
    split_width = 64 * chunks_per_split
    for split_start in range(0, len(selected), split_width):
        peak = torch.full((len(query), 1), -torch.inf, dtype=torch.float32, device=query.device)
        denominator = torch.zeros_like(peak)
        numerator = torch.zeros((len(query), 512), dtype=torch.float32, device=query.device)
        for start in range(split_start, min(split_start + split_width, len(selected)), 64):
            end = min(start + 64, len(selected))
            if not bool(valid[start:end].any()):
                continue
            values = scores[:, start:end].float()
            combined_peak = torch.maximum(peak, values.amax(-1, keepdim=True))
            rescale = (peak - combined_peak).exp()
            probabilities = (values - combined_peak).exp()
            numerator *= rescale
            denominator = denominator * rescale + probabilities.sum(-1, keepdim=True)
            for group in range(4):
                weighted = probabilities * key_scales[start:end, group][None]
                scale = weighted.abs().amax(-1, keepdim=True).clamp_min(1e-10)
                scale *= torch.tensor(1 / 448, dtype=torch.float32, device=query.device)
                normalized = (weighted * scale.reciprocal()).clamp(-448, 448)
                high = normalized.to(torch.float8_e4m3fn).float()
                low = (normalized - high).to(torch.float8_e4m3fn).float()
                contribution = high @ keys[start:end, group] + low @ keys[start:end, group]
                numerator[:, group * 128:(group + 1) * 128] += contribution * scale
            peak = combined_peak
        partials.append(numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny))
        partitions.append((peak + denominator.log()).flatten())
    partials = torch.stack(partials, 1)
    split_probabilities = (torch.stack(partitions, 1).softmax(-1) if bool(valid.any())
                           else torch.zeros((len(query), len(partitions)), device=query.device))
    combined = (partials.bfloat16().float() * split_probabilities[..., None]).sum(1)
    return SparseAttentionArithmetic(scores, dense, partials, split_probabilities, combined, combined.bfloat16())


def fp32_dot_allowance(exact, absolute_product_sum, terms, *, bf16_output):
    """Elementwise worst-case FP32 dot-product error and an optional BF16 final rounding.

    The bound is gamma(n) times the sum of absolute products, with
    gamma(n) = n*u/(1-n*u) and u = 2**-23. One FP32 ulp per operation covers
    directed and nearest rounding. PTX leaves MMA accumulation order and rounding
    unspecified. Applying this bound requires independent
    validation that the selected operator uses FP32 accumulation with at most
    that rounding error. BF16 round-to-nearest contributes
    u = 2**-8 times the bounded accumulated value. The bound requires finite, normal or zero
    operands and intermediates, with FP32 accumulation until the final round.
    """
    if type(terms) is not int or not 0 < terms < 2**23:
        raise ValueError("FP32 dot bounds require a positive term count below 2**23")
    if exact.shape != absolute_product_sum.shape or bool((absolute_product_sum < 0).any()):
        raise ValueError("FP32 dot bounds require matching values and nonnegative product sums")
    unit = 2**-23
    allowance = (terms * unit / (1 - terms * unit)) * absolute_product_sum
    return allowance * (1 + 2**-8) + exact.abs() * 2**-8 if bf16_output else allowance
