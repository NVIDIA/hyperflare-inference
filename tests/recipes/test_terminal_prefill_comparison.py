# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The state diagnostic must report divergence even when output lengths differ."""

import pytest
import torch

from qualify.deepseek41_terminal_prefill import differences


def test_state_comparison_preserves_exact_bytes_and_scalar_counters():
    state = dict(counter=torch.tensor(1), window=torch.tensor([0., -0.]))
    assert differences(state, {name: value.clone() for name, value in state.items()}) == {}
    changed = dict(counter=torch.tensor(2), window=torch.tensor([0., 0.]))
    assert differences(state, changed) == dict(counter=1, window=1)


def test_state_comparison_reports_different_token_counts_without_broadcasting():
    result = differences(dict(tokens=torch.tensor([1, 2])), dict(tokens=torch.tensor([1, 2, 3])))
    assert result['tokens']['reference_shape'] == [2]
    assert result['tokens']['candidate_shape'] == [3]
    with pytest.raises(AssertionError, match='different declared regions'):
        differences(dict(tokens=torch.tensor([1])), {})
