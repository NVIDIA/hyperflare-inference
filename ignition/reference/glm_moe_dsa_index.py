# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent FP64 scoring of GLM's stored FP8 index queries and keys."""

import torch

from .glm_moe_dsa_cache import quantize_index


def index_score_interval(query, head_weights, key_bytes, key_scales):
    """Return the score and an elementwise FP32 arithmetic allowance for every query/key pair.

    The query holds BF16 vectors after rotary embedding. The head weights
    are the FP32 projection before head normalization. Keys use one E4M3
    vector and one FP32 scale per token. Products of E4M3 values are exact
    in FP32. The allowance assumes at most one FP32 ulp per accumulation,
    including the 128-term dot and the weighted reduction over 32 heads.
    ReLU has Lipschitz constant one, including at zero.
    """
    if (query.ndim != 3 or query.shape[1:] != (32, 128) or query.dtype != torch.bfloat16
            or head_weights.dtype != torch.float32 or head_weights.shape != query.shape[:2]
            or key_bytes.ndim != 2 or key_bytes.shape[1] != 128 or key_bytes.dtype != torch.uint8
            or key_scales.shape != key_bytes.shape[:1] or key_scales.dtype != torch.float32):
        raise ValueError("GLM scoring requires 32 BF16 query heads, FP32 gates and 128-channel FP8 keys")
    if not bool(torch.isfinite(head_weights).all() and torch.isfinite(key_scales).all() and (key_scales > 0).all()):
        raise ValueError("GLM index head weights must be finite and key scales finite and positive")
    qbytes, qscales = quantize_index(query, query=True)
    q = qbytes.view(torch.float8_e4m3fn).double().flatten(0, 1)
    k = key_bytes.view(torch.float8_e4m3fn).double()
    if not bool(torch.isfinite(k).all()):
        raise ValueError("GLM index key encodings must be finite")
    dots = (q @ k.T).view(len(query), 32, -1)
    magnitudes = (q.abs() @ k.abs().T).view_as(dots)
    gates = head_weights.double() * qscales.double() / 64
    weighted = dots.relu() * gates[..., None]
    scale = key_scales.double()
    scores = weighted.sum(1) * scale

    unit = 2**-23

    def gamma(n):
        return n * unit / (1 - n * unit)

    # The additional products fold the query scale and head normalization
    # into each gate, then apply the key scale to the per-token sum.
    error = gamma(128) * (magnitudes * gates.abs()[..., None]).sum(1)
    error = (error * (1 + gamma(40)) + gamma(40) * weighted.abs().sum(1)) * scale
    return scores, error * (1 + unit) + unit * scores.abs()


def valid_index_cutoff(records, lengths, positions, scores, allowance):
    """Check padding, visibility, uniqueness and every retained-versus-omitted score inequality."""
    rows, width = records.shape
    if (scores.ndim != 2 or scores.shape[0] != rows or scores.shape != allowance.shape
            or lengths.shape != (rows,) or positions.shape != (rows,) or scores.shape[1] == 0):
        raise ValueError("GLM cutoff check requires matching score bounds and row extents")
    count = scores.shape[1]
    live = torch.arange(width, device=records.device)[None] < lengths[:, None]
    visible = torch.arange(count, device=records.device)[None] <= positions[:, None]
    expected = (positions + 1).clamp(0, width)
    ordered = records.masked_fill(~live, count).sort(1).values
    repeated = ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] < count)).any(1)
    structural = ((lengths == expected) & (positions >= -1) & (positions < count) & ~repeated
                  & torch.where(live, (records >= 0) & (records <= positions[:, None]), records == -1).all(1))
    # Padding has a separate scatter destination, so it cannot mask token zero.
    keep = torch.zeros((rows, count + 1), dtype=torch.bool, device=records.device)
    keep.scatter_(1, torch.where(live, records.clamp(0, count), count).long(), True)
    keep = keep[:, :count]
    retained_upper = (scores + allowance).masked_fill(~keep, torch.inf).amin(1)
    omitted_lower = (scores - allowance).masked_fill(~visible | keep, -torch.inf).amax(1)
    finite = torch.isfinite(scores).all(1) & torch.isfinite(allowance).all(1) & (allowance >= 0).all(1)
    return structural & finite & (retained_upper >= omitted_lower)
