# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Physical cache mappings and mixed-key prefixes preserve request boundaries."""

import torch

from ignition.components.fused_mla import combined_indices, physical_indices


def test_selected_indices_follow_each_requests_page_mapping():
    logical = torch.tensor([[0, 63, 64, 127, -1], [1, 65, -1, -1, -1]], dtype=torch.int32)
    pages = torch.tensor([[5, 2], [3, 7]], dtype=torch.int32)
    actual = physical_indices(logical, pages, 64)
    assert actual.tolist() == [[320, 383, 128, 191, -1], [193, 449, -1, -1, -1]]


def test_mixed_key_prefix_does_not_hide_window_keys_behind_unused_selected_slots():
    selected = torch.tensor([[10, 11, 12, -1], [-1, -1, -1, -1], [20, -1, -1, -1]], dtype=torch.int32)
    window = torch.tensor([[100, 101, -1], [200, -1, -1], [-1, -1, -1]], dtype=torch.int32)
    ids, lengths = combined_indices(window, torch.tensor([2, 1, 0], dtype=torch.int32),
                                    selected, torch.tensor([3, 0, 1], dtype=torch.int32))
    assert lengths.tolist() == [5, 1, 1]
    assert ids.tolist() == [[10, 11, 12, 100, 101, -1, -1],
                            [200, -1, -1, -1, -1, -1, -1],
                            [20, -1, -1, -1, -1, -1, -1]]
