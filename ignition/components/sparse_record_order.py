# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sort causal sparse positions per row and address selected records through its page table."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sort_records(logical, positions, pages, records, lengths, logical_row, logical_col,
                  position_stride, page_row, page_col, num_pages, WIDTH: tl.constexpr,
                  BLOCK: tl.constexpr, PAGE_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    position = tl.load(logical + row * logical_row + lane * logical_col, mask=lane < WIDTH, other=-1)
    limit = tl.load(positions + row * position_stride)
    page_id = position // PAGE_SIZE
    live = (position >= 0) & (position <= limit) & (page_id < num_pages)
    table = pages + row * page_row
    physical = tl.load(table + page_id * page_col, mask=live, other=-1)
    live = live & (physical >= 0)
    # Unassigned pages may occur between live pages. Discard them before sorting so padding stays last.
    ordered = tl.sort(tl.where(live, position, 2147483647), descending=False)
    count = tl.sum(live.to(tl.int32), 0)
    page = tl.load(table + (ordered // PAGE_SIZE) * page_col, mask=lane < count, other=0)
    result = tl.where(lane < count, page * PAGE_SIZE + ordered % PAGE_SIZE, -1)
    tl.store(records + row * WIDTH + lane, result, mask=lane < WIDTH)
    tl.store(lengths + row, count)


def ordered_sparse_records(logical, positions, pages, page_size):
    """Return physical records in logical order with valid counts for each query row."""
    rows, width = logical.shape
    records = torch.empty((rows, width), dtype=torch.int32, device=logical.device)
    lengths = torch.empty(rows, dtype=torch.int32, device=logical.device)
    torch.library.wrap_triton(_sort_records)[(rows,)](
        logical, positions, pages, records, lengths, logical.stride(0), logical.stride(1), positions.stride(0),
        pages.stride(0), pages.stride(1), pages.shape[1], width, triton.next_power_of_2(width), page_size,
        num_warps=8)
    return records, lengths
