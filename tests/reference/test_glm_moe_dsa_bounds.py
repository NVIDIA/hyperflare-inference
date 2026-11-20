# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Rejection controls for independent GLM arithmetic intervals."""

import pytest
import torch

from ignition.reference.glm_moe_dsa_bounds import (
    projection_interval,
    rms_norm_interval,
    rounded_bf16_interval,
    sigmoid_relative_allowance,
    swiglu_interval,
)


def inside(value, bounds):
    lower, upper = bounds
    return bool(((lower <= value) & (value <= upper)).all())


def test_projection_interval_preserves_fp32_output_precision():
    source = torch.tensor([[1., 1.]], dtype=torch.bfloat16)
    weight = torch.tensor([[1., 2**-8]], dtype=torch.bfloat16)
    bounds = projection_interval(source, weight, bf16_output=False)
    exact = torch.tensor([[1 + 2**-8]], dtype=torch.float64)
    assert inside(exact, bounds)
    assert not inside(exact.bfloat16().double(), bounds)
    assert not inside(exact * 1.2, bounds)


def test_rms_interval_rejects_wrong_variance():
    source = torch.tensor([[1., -2., 3., -4.]], dtype=torch.bfloat16)
    weight = torch.tensor([.5, 1., -1.5, 2.], dtype=torch.bfloat16)
    bounds = rms_norm_interval(source, weight, 1e-6)
    normalized = (source.double() / (7.5 + 1e-6)**.5).bfloat16()
    expected = (normalized.double() * weight.double()).bfloat16().double()
    assert inside(expected, bounds)
    wrong = (source.double() / (7.5 * 1.2 + 1e-6)**.5).bfloat16()
    assert not inside((wrong.double() * weight.double()).bfloat16().double(), bounds)


def test_swiglu_interval_handles_signs_and_rejects_omitted_activation():
    gate = torch.tensor([-2., -1., 0., 1., 2.], dtype=torch.float64)
    up = torch.tensor([-1., 2., 3., 4., -2.], dtype=torch.float64)
    bounds = swiglu_interval(gate, gate, up, up)
    activation = (gate / (1 + torch.exp(-gate))).bfloat16().double()
    assert inside((activation * up).bfloat16().double(), bounds)
    assert not inside((gate * up).bfloat16().double(), bounds)


def test_interval_domains_are_explicit():
    with pytest.raises(ValueError, match="finite and ordered"):
        rounded_bf16_interval(torch.tensor([2.]), torch.tensor([1.]))
    with pytest.raises(ValueError, match="finite inputs"):
        sigmoid_relative_allowance(torch.tensor([81.]))
