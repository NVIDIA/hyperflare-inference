# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Check GLM FP8 index scores and cutoff decisions with hand-computed values."""

import torch

from ignition.reference.glm_moe_dsa_index import index_score_interval, valid_index_cutoff


def test_key_scale_changes_the_winning_set_and_visibility_limits_eligible_keys():
    query = torch.zeros((1, 32, 128), dtype=torch.bfloat16)
    query[0, 0, 0] = 2
    gates = torch.zeros((1, 32))
    gates[0, 0] = 32
    keys = torch.zeros((4, 128), dtype=torch.uint8)
    keys[:, 0] = torch.tensor([0x38, 0x40, 0xB8, 0x38])  # E4M3 values 1, 2, -1, 1.
    scales = torch.tensor([1., .25, 1., 1.])
    scores, allowance = index_score_interval(query, gates, keys, scales)
    assert scores.tolist() == [[1., .5, 0., 1.]]

    def accepted(records, length, position, score=scores, error=allowance):
        return valid_index_cutoff(torch.tensor([records]), torch.tensor([length]),
                                  torch.tensor([position]), score, error).item()

    assert accepted([0, 3], 2, 3)
    assert accepted([3], 1, 3) and accepted([0], 1, 3)  # Both keys attain the maximum.
    assert not accepted([0, 1], 2, 3)
    assert not accepted([0, 3], 2, 2)
    assert not accepted([0, 0], 2, 3)
    assert accepted([0, -1], 1, 0) and accepted([-1, -1], 0, -1)
    assert not accepted([0, 0], 1, 0)

    scales[1] = 1.
    wrong_scores, wrong_allowance = index_score_interval(query, gates, keys, scales)
    assert wrong_scores.tolist() == [[1., 2., 0., 1.]]
    assert not accepted([0, 3], 2, 3, wrong_scores, wrong_allowance)
