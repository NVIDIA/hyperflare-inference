# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Conditional arithmetic bounds for SM120 GLM attention on stored operands.

The reference center is FP64 attention after independent FP8 query rounding.
The allowance includes FP32 reductions, approximate transcendental operations,
two-term E4M3 probability/value products, and BF16 split and final storage.
It assumes validated FP32 dot accumulations with at most one ulp of rounding
per operation. PTX leaves MMA rounding unspecified, so the selected operator
must separately satisfy that contract. This is not a whole-model tolerance.
"""

from dataclasses import dataclass
import math

import torch

from .glm_moe_dsa_cache import quantize_index


@dataclass(frozen=True)
class AttentionAllowance:
    expected: torch.Tensor
    absolute: torch.Tensor
    probability_relative: torch.Tensor
    score_absolute: torch.Tensor
    packing_absolute: torch.Tensor


def two_term_e4m3_allowance(values, scale):
    """Bound E4M3(x) + E4M3(x - E4M3(x)), followed by rescaling.

    Each round-to-nearest E4M3 conversion contributes at most |x|/16 +
    2**-10, including subnormals. Applying it twice gives |x|/256 plus
    (17/16)*2**-10. The bound covers the two rounds when the supplied scaled
    values lie within the finite E4M3 range.
    """
    return values.abs() / 256 + scale.abs() * ((17 / 16) * 2**-10)


@torch.no_grad()
def sparse_attention_allowance(query, records, selected, *, chunks_per_split, softmax_scale):
    """Return an elementwise bound for a row with eight heads and 64-key chunks.

    The error center consumes only independent query rounding and stored cache
    values. No native outputs or observed distances enter the allowance.
    `selected` preserves chunk membership and uses -1 for padding. The
    executed plan supplies `chunks_per_split`. Positive scale and finite normal FP32
    intermediate assumptions are checked where they enter the calculation.
    """
    if query.shape != (8, 576) or query.dtype != torch.bfloat16:
        raise ValueError("attention allowance requires eight BF16 queries of width 576")
    if records.ndim != 2 or records.shape[1] != 656 or records.dtype != torch.uint8:
        raise ValueError("attention allowance requires 656-byte latent records")
    if selected.ndim != 1 or selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("attention allowance requires integer selected record indices")
    if not len(selected) or bool(((selected < -1) | (selected >= len(records))).any()):
        raise ValueError("attention allowance selections must name stored records or padding")
    if type(chunks_per_split) is not int or not 1 <= chunks_per_split <= 32:
        raise ValueError("attention allowance supports one through 32 chunks per split")
    if softmax_scale != 1 / 16:
        raise ValueError("GLM attention allowance requires the checkpoint softmax scale 1/16")
    live = selected >= 0
    data = records[selected.clamp_min(0).long()]
    raw = data[:, :512].contiguous().view(torch.float8_e4m3fn).double().reshape(-1, 4, 128)
    scales = data[:, 512:528].contiguous().view(torch.float32).double()
    rotary = data[:, 528:].contiguous().view(torch.bfloat16).double()
    if not bool(torch.isfinite(query).all() and torch.isfinite(raw).all()
                and torch.isfinite(rotary).all() and torch.isfinite(scales).all()
                and (scales > 0).all()):
        raise ValueError("attention allowance requires finite operands and positive cache scales")
    qbytes, qscale = quantize_index(query[:, :512].reshape(8, 4, 128), query=True)
    q = qbytes.view(torch.float8_e4m3fn).double()
    factor = qscale.double()[:, None, :] * scales[None]
    groups = torch.einsum("hgd,sgd->hsg", q, raw) * factor
    group_magnitude = torch.einsum("hgd,sgd->hsg", q.abs(), raw.abs()) * factor
    rope = query[:, 512:].double() @ rotary.T
    rope_magnitude = query[:, 512:].double().abs() @ rotary.abs().T
    scores = (groups.sum(-1) + rope) * softmax_scale
    latent = (raw * scales[..., None]).flatten(1)
    if not bool(live.any()):
        zero = torch.zeros((8, 512), dtype=torch.float64, device=query.device)
        return AttentionAllowance(zero, zero, zero[:, 0], scores.zero_(), zero)
    unit = 2**-23
    gamma = lambda count: count * unit / (1 - count * unit)
    # Each score combines four 128-term E4M3 dots and one 64-term rotary dot.
    # Allow for two scales per group, the sum, and conversion to base 2.
    score_error = ((gamma(128) + gamma(12)) * group_magnitude.sum(-1)
                   + (gamma(64) + gamma(12)) * rope_magnitude) * softmax_scale
    score_error = score_error * (1 + gamma(3)) + gamma(3) * scores.abs()
    scores[:, ~live] = -torch.inf
    probabilities = scores.softmax(-1)
    expected = probabilities @ latent
    magnitude = probabilities @ latent.abs()
    maximum = scores[:, live].abs().amax(-1)
    span = scores[:, live].amax(-1) - scores[:, live].amin(-1)
    score_error_max = score_error[:, live].amax(-1)
    count = int(live.sum())
    chunks = chunks_per_split
    splits = math.ceil(len(selected) / (64 * chunks))
    # A probability traverses at most two warp/block exponentials and one
    # rescaling for every later chunk. ex2.approx differs by at most two ulps
    # from the rounded value: 2.5 FP32 ulps covers that and the final rounding.
    exponent_calls = chunks + 2
    exp_relative = 2.5 * unit
    log_probability = (2 * score_error_max + exponent_calls * unit * span
                       + exponent_calls * math.log1p(exp_relative))
    sum_error = gamma(min(count, 64 * chunks) + 2 * chunks)
    # PTX lg2 has absolute error 2**-22 on (0.5, 2), and relative error
    # 2**-22 outside that range. The unnormalized sum is between 1 and
    # the split width. An FP32 addition determines the stored base-2 log partition.
    log_error = (2**-22 * max(1, math.log2(64 * chunks)) * math.log(2)
                 + unit * (maximum + math.log(64 * chunks)))
    partition_error = score_error_max + log_probability + math.log1p(sum_error) + log_error
    merge_sum_error = gamma(splits)
    # Normalize within a split, then normalize the split weights. Each ratio
    # includes the positive-sum and reciprocal errors. The exponential factors
    # bound numerator and denominator perturbations in opposite directions.
    within = (log_probability * 2).exp() * (1 + sum_error) / (1 - sum_error) * (1 + unit)
    merge = (2 * partition_error + unit * (2 * maximum + math.log(count))).exp()
    merge *= (1 + exp_relative)**2 * (1 + merge_sum_error) / (1 - merge_sum_error) * (1 + unit)
    probability_error = within * merge - 1
    packing = torch.zeros_like(expected)
    for start in range(0, len(selected), 64):
        end = min(start + 64, len(selected))
        weighted = probabilities[:, start:end, None] * scales[None, start:end]
        # A scale floor before normalization can contribute at most this
        # absolute floor afterward, since the unnormalized denominator >= 1.
        scale = weighted.amax(1, keepdim=True).clamp_min(1e-10) / 448
        error = two_term_e4m3_allowance(weighted, scale)
        packing += torch.einsum("hsg,sgd->hgd", error, raw[start:end].abs()).flatten(1)
    # HIGH and LOW each use a 64-term dot. Multiplications, additions,
    # recurrence rescaling, and normalization add at most six operations
    # per chunk along a contribution's path.
    pv_error = gamma(134 * chunks + 4)
    total = probability_error[:, None] * magnitude
    total += (1 + probability_error[:, None]) * (packing + pv_error * (magnitude + packing))
    partial_magnitude = torch.zeros_like(expected)
    for start in range(0, len(selected), 64 * chunks):
        end = min(start + 64 * chunks, len(selected))
        partial_magnitude += (probabilities[:, start:end] @ latent[start:end]).abs()
    total += 2**-8 * (partial_magnitude + total)
    total += gamma(2 * splits + 2) * (partial_magnitude + total)
    total = total * (1 + 2**-8) + expected.abs() * 2**-8
    # Flushing FP32 subnormal exponentials to zero contributes less than
    # count*tiny times the largest finite value. This term is normally tiny.
    total += count * torch.finfo(torch.float32).tiny * latent.abs().amax(0)[None]
    if not bool(torch.isfinite(total).all()):
        raise ValueError("attention allowance requires finite error propagation")
    return AttentionAllowance(expected, total, probability_error, score_error, packing)
