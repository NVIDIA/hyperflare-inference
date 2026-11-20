# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Prepared low-rank message controllers for repeated Inkling decoder passes.

Adapted from Inkling2MessageController in vLLM PR #69028, loop.py at
b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0:
https://github.com/vllm-project/vllm/blob/b3c0ff75424ea9393ce2dbfd2e912bb19f7323f0/vllm/models/inkling2/nvidia/loop.py
Flare binds the projections during preparation and preserves the FP32 blend.
"""

from dataclasses import dataclass

import torch

from ignition.components.linear import UnquantizedProjection
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.rebuild_manifest import weight_container


STREAMS = ("attn", "k", "v", "mlp")


@weight_container(("down", "gain", "forget"))
@dataclass(frozen=True)
class ControllerWeights:
    down: torch.Tensor
    gain: torch.Tensor
    forget: torch.Tensor


class PreparedController:
    def __init__(self, weights, norm, *, tokens, eps, session, name, examples):
        self.norm, self.eps = norm, eps
        width, rank = weights.gain.shape
        def example(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "inkling.controller." + role, shape, dtype, norm.device)
        pair = example("pair", (tokens, 2 * width))
        hidden = example("hidden", (tokens, rank))
        output = example("output", (tokens, width), torch.float32)
        self.down = UnquantizedProjection(pair, weights.down, hidden, session=session,
            name=name + ".down").functional(session=session)
        self.gain = UnquantizedProjection(hidden, weights.gain, output, session=session,
            name=name + ".gain").functional(session=session)
        self.forget = UnquantizedProjection(hidden, weights.forget, output, session=session,
            name=name + ".forget").functional(session=session)

    def __call__(self, dest, message):
        pair = torch.stack((dest, message), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        hidden = self.down(pair.to(torch.bfloat16).flatten(1))
        blend = (1.0 - torch.tanh(self.forget(hidden))) * dest + torch.tanh(self.gain(hidden)) * message
        normalized = blend * torch.rsqrt(blend.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.norm.float()).to(torch.bfloat16)
