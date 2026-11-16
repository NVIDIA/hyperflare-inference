# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling2 draft feature mixing, rotary positions, and committed context selection."""

from dataclasses import dataclass

import torch


def mix_features(features, alpha, weight, eps):
    mixed = torch.zeros_like(features[:, 0], dtype=torch.float32)
    for index in range(features.shape[1]):
        mixed = mixed + features[:, index].float() * alpha[index].float()
    scale = torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
    return (mixed * scale * weight.float()).bfloat16()


def block_inputs(anchor, mask, block):
    """Expand one normalized first-position embedding per request into its block of learned masks."""
    return torch.cat((anchor[:, None], mask.expand(anchor.shape[0], block - 1, -1)), 1).flatten(0, 1)


def rotary_table(positions, inverse):
    """Return FP32 cosines and sines of each row's position, shared by every rotation of those rows."""
    angle = positions.clamp_min(0).float()[:, None] * inverse[None]
    return angle.cos()[:, None], angle.sin()[:, None]


def rotate(values, table):
    cosine, sine = table
    left, right = values.float().chunk(2, -1)
    return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)


def rotary(values, positions, inverse):
    return rotate(values, rotary_table(positions, inverse))


@dataclass(frozen=True)
class StagingPlan:
    """Where each slot of a request's private attention pages comes from.

    The plan depends on request metadata only, so one plan serves the key and value histories of
    every draft layer with the same history geometry. A slot holds committed history
    (`history[rows, slots]`), a row of the block being drafted (`values[indices]`), or zero.
    `lengths` is each request's visible key count.
    """

    rows: torch.Tensor
    slots: torch.Tensor
    indices: torch.Tensor
    committed: torch.Tensor
    valid: torch.Tensor
    lengths: torch.Tensor
    page_tokens: int


def window_plan(metadata, window, pages, *, page_tokens=128):
    """Plan the same committed suffix of a `window`-position ring before every query of a noncausal block."""
    m = metadata
    offsets = torch.arange(pages * page_tokens, device=m.starts.device)[None]
    prefix = m.starts.clamp(max=window)[:, None]
    positions = m.starts[:, None] - prefix + offsets
    return StagingPlan(m.state_rows[:, None], positions.remainder(window),
        (m.query_start_loc[:-1, None] + offsets - prefix).clamp(0, m.positions.numel() - 1),
        offsets < prefix, (m.lengths[:, None] > 0) & (offsets < prefix + m.lengths[:, None]),
        torch.where(m.lengths > 0, m.starts.clamp(max=window).int() + m.lengths, 0), page_tokens)


def tail_plan(metadata, page_tokens, pages):
    """Plan the committed partial page followed by the block's rows in private tail pages."""
    m = metadata
    rows = m.starts.numel()
    offsets = torch.arange(pages * page_tokens, device=m.starts.device)[None]
    prefix = m.starts.remainder(page_tokens)[:, None]
    columns = (m.starts // page_tokens).clamp(max=m.page_table.shape[1] - 1)
    requests = torch.arange(rows, device=m.starts.device)
    source_pages = m.page_table[requests, columns].long().clamp_min(0)
    return StagingPlan(source_pages[:, None], offsets.remainder(page_tokens),
        (m.query_start_loc[:-1, None] + offsets - prefix).clamp(0, m.positions.numel() - 1),
        offsets < prefix, (m.lengths[:, None] > 0) & (offsets < prefix + m.lengths[:, None]),
        torch.where(m.lengths > 0, m.starts.int() + m.lengths, 0), page_tokens)


def stage(history, values, plan):
    """Copy the planned history and block rows into private attention pages without changing either source."""
    staged = torch.where(plan.valid[:, :, None, None],
        torch.where(plan.committed[:, :, None, None], history[plan.rows, plan.slots], values[plan.indices]), 0)
    return staged.view(-1, plan.page_tokens, *values.shape[1:])


def accepted_rows(metadata, counts, *, window=None):
    offset = metadata.positions - metadata.starts[metadata.requests]
    count = counts[metadata.requests]
    live = (metadata.positions >= 0) & (offset < count) & (offset >= 0)
    if window is not None:
        live = live & (offset >= (count - window).clamp_min(0))
    return live


def context_slots(metadata, counts, *, window=None, page_tokens=128):
    live = accepted_rows(metadata, counts, window=window)
    if window is not None:
        slot = metadata.state_rows[metadata.requests] * window + metadata.positions.remainder(window)
    else:
        columns = metadata.positions.clamp_min(0) // page_tokens
        rows = torch.arange(metadata.positions.numel(), device=metadata.positions.device)
        page = metadata.token_page_table[rows, columns.clamp(max=metadata.token_page_table.shape[1] - 1)]
        live = live & (page >= 0) & (columns < metadata.token_page_table.shape[1])
        slot = page.long() * page_tokens + metadata.positions.remainder(page_tokens)
    return torch.where(live, slot, -1)
