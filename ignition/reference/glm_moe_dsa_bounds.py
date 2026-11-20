# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent intervals for GLM BF16 projections, RMS normalization, and SwiGLU.

These references bound individual operations on fixed supplied operands.
They require finite normal FP32 intermediates and the independently validated
one-ulp FP32 accumulation contract described by `fp32_dot_allowance`.
They do not turn local bounds into a whole-model distribution guarantee.
"""

import math

import torch

from .glm_moe_dsa_attention import fp32_dot_allowance


def rounded_bf16_interval(lower, upper):
    """Round both endpoints under the monotone BF16 round-to-nearest operation."""
    if bool((lower > upper).any()) or not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
        raise ValueError("BF16 interval endpoints must be finite and ordered")
    return lower.bfloat16().double(), upper.bfloat16().double()


def projection_interval(source, weight, *, bf16_output):
    """Bound a matrix product with BF16 operands and FP32 accumulation."""
    source, weight = source.double(), weight.double()
    exact = source @ weight.T
    magnitude = source.abs() @ weight.abs().T
    error = fp32_dot_allowance(exact, magnitude, source.shape[-1], bf16_output=False)
    bounds = exact - error, exact + error
    return rounded_bf16_interval(*bounds) if bf16_output else bounds


def rms_norm_interval(source, weight, eps):
    """Bound FP32 RMS statistics, a BF16 normalized value, and BF16 weighted output.

    BF16 squares are exact in FP32. gamma(n+2) covers the sum, division by
    width, and addition of epsilon. PTX rsqrt.approx's relative error is at
    most 2**-22.9. Allow one FP32 ulp for the multiplication before BF16 storage.
    Multiplying two BF16 values is exact in FP32 before the final BF16 round.
    """
    source, weight = source.double(), weight.double()
    unit = 2**-23
    count = source.shape[-1] + 2
    gamma = count * unit / (1 - count * unit)
    variance = source.square().mean(-1, keepdim=True)
    error = gamma * variance + unit * eps
    low_variance, high_variance = variance + eps - error, variance + eps + error
    if not eps > 0 or bool((low_variance <= 0).any()):
        raise ValueError("RMS interval requires a positive bounded variance")
    low_inverse = high_variance.rsqrt() * (1 - 2**-22.9)
    high_inverse = low_variance.rsqrt() * (1 + 2**-22.9)
    a, b = source * low_inverse, source * high_inverse
    lower, upper = torch.minimum(a, b), torch.maximum(a, b)
    error = unit * torch.maximum(lower.abs(), upper.abs())
    lower, upper = rounded_bf16_interval(lower - error, upper + error)
    a, b = lower * weight, upper * weight
    return rounded_bf16_interval(torch.minimum(a, b), torch.maximum(a, b))


def sigmoid_relative_allowance(values):
    """Bound FP32 sigmoid built from a base-2 exponential and reciprocal.

    The argument's base conversion takes two FP32 rounding operations.
    ex2.approx contributes at most 2.5 ulps relative to the exact result. The subsequent
    addition, reciprocal, and output rounding add three one-ulp operations.
    This finite-range contract excludes exponential overflow.
    """
    values = values.double()
    if not bool(torch.isfinite(values).all()) or bool((values.abs() > 80).any()):
        raise ValueError("sigmoid allowance requires finite inputs within [-80, 80]")
    unit = 2**-23
    gamma2 = 2 * unit / (1 - 2 * unit)
    gamma3 = 3 * unit / (1 - 3 * unit)
    exponential = torch.expm1(gamma2 * values.abs()) + 2.5 * unit
    return exponential / (1 - exponential) + gamma3


def swiglu_interval(gate_lower, gate_upper, up_lower, up_upper):
    """Bound SiLU with BF16 activation and product rounding over projection intervals.

    |SiLU'(x)| <= 1 + 1/e, since |x|*sigmoid(x)*(1-sigmoid(x)) <= 1/e.
    That global derivative bound includes the nonmonotone negative region.
    Endpoint rounding also covers a projection close to a BF16 midpoint.
    """
    center = (gate_lower + gate_upper) / 2
    radius = (gate_upper - gate_lower) / 2
    exact = center * center.sigmoid()
    maximum = torch.maximum(gate_lower.abs(), gate_upper.abs())
    sigmoid_error = sigmoid_relative_allowance(maximum)
    activation_error = (1 + 1/math.e) * radius + maximum * sigmoid_error
    activation_error += 2**-23 * (maximum + activation_error)
    lower, upper = rounded_bf16_interval(exact - activation_error, exact + activation_error)
    products = torch.stack((lower * up_lower, lower * up_upper, upper * up_lower, upper * up_upper))
    return rounded_bf16_interval(products.amin(0), products.amax(0))
