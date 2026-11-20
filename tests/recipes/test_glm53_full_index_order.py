# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Selected sparse records retain a causal token order independent of top-k enumeration."""

import torch

from ignition.recipes.glm53_full.indexer import selected_records


def test_selection_enumeration_cannot_change_attention_partitions():
    # Logical pages use nonmonotonic physical addresses. Ordering by physical
    # record would make the arithmetic depend on the cache allocator.
    pages = torch.tensor([[7, 2, 9], [4, -1, -1]], dtype=torch.int32)
    positions = torch.tensor([130, -1])
    logical = torch.tensor([[129, 3, 64, 65, -1, 180], [0, 2, -1, 1, 4, -1]], dtype=torch.int32)
    expected = torch.tensor([[7 * 64 + 3, 2 * 64, 2 * 64 + 1, 9 * 64 + 1, -1, -1],
                             [-1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    for permutation in (torch.arange(6), torch.arange(5, -1, -1), torch.tensor([3, 5, 0, 4, 1, 2])):
        selection = selected_records(logical[:, permutation], positions, pages)
        assert torch.equal(selection.records, expected)
        assert torch.equal(selection.lengths, torch.tensor([4, 0], dtype=torch.int32))
