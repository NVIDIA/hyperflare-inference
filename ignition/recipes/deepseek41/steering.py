# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared activation-steering constants for DeepSeek V4.1 decoder layers."""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch
from safetensors.torch import load

from ignition.preparation.persistent import persistent_copy


def steered_identity(base_identity: str, sidecar_sha256: bytes | None) -> str:
    """Fold an optional steering sidecar SHA-256 into a 64-hex model identity."""
    if sidecar_sha256 is None:
        return base_identity
    return hashlib.sha256(bytes.fromhex(base_identity) + sidecar_sha256).hexdigest()


@dataclass(frozen=True)
class LayerSteering:
    alpha: float
    direction: torch.Tensor
    attention_gain_delta: torch.Tensor


@dataclass(frozen=True)
class SteeringSidecar:
    layers: Mapping[int, LayerSteering]
    sha256: bytes

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "SteeringSidecar":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"steering sidecar is not a file: {path}")
        raw_bytes = path.read_bytes()
        digest = hashlib.sha256(raw_bytes).digest()
        tensors = load(raw_bytes)
        expected = {
            f"layers.{layer}.{name}"
            for layer in range(4, 37)
            for name in ("alpha", "attn_row_gain", "direction")
        }
        if set(tensors) != expected:
            missing = sorted(expected - set(tensors))
            extra = sorted(set(tensors) - expected)
            raise ValueError(
                f"steering sidecar tensor inventory differs from layers 4 through 36: "
                f"missing={missing}, extra={extra}"
            )
        layers = {}
        for layer in range(4, 37):
            prefix = f"layers.{layer}"
            alpha_tensor = tensors[f"{prefix}.alpha"]
            direction = tensors[f"{prefix}.direction"]
            gain = tensors[f"{prefix}.attn_row_gain"]
            if alpha_tensor.dtype != torch.float32 or alpha_tensor.shape != torch.Size([]):
                raise ValueError(f"{prefix}.alpha must be a scalar FP32 tensor")
            if direction.dtype != torch.bfloat16 or direction.shape != (5120,):
                raise ValueError(f"{prefix}.direction must be BF16[5120]")
            if gain.dtype != torch.bfloat16 or gain.shape != (5120,):
                raise ValueError(f"{prefix}.attn_row_gain must be BF16[5120]")
            alpha = float(alpha_tensor.item())
            direction_fp32 = direction.float()
            gain_fp32 = gain.float()
            norm = float(direction_fp32.norm().item())
            if (not math.isfinite(alpha) or alpha <= 0.0
                    or not bool(torch.isfinite(direction_fp32).all())
                    or not bool(torch.isfinite(gain_fp32).all())
                    or abs(norm - 1.0) >= 1e-3):
                raise ValueError(f"{prefix} contains non-finite, non-positive-alpha, or non-unit-norm steering values")
            layers[layer] = LayerSteering(
                alpha=alpha,
                direction=persistent_copy(direction_fp32.to(device=device, non_blocking=False).contiguous()),
                attention_gain_delta=persistent_copy(
                    (gain_fp32 - 1.0).to(device=device, non_blocking=False).contiguous()
                ),
            )
        return cls(MappingProxyType(layers), digest)


def steer_attention(output: torch.Tensor, steering: LayerSteering,
                    token_scale: torch.Tensor) -> torch.Tensor:
    """Apply scaled directional ablation and row-gain rescaling to replicated attention output."""
    direction = steering.direction
    projected = output.float()
    dot = (projected @ direction).unsqueeze(1)
    projected = projected - token_scale * steering.alpha * dot * direction.unsqueeze(0)
    projected = projected * (
        1.0 + token_scale * steering.attention_gain_delta.unsqueeze(0)
    )
    steered = projected.to(output.dtype)
    return torch.where(token_scale == 0.0, output, steered)


def steer_shared_expert(output: torch.Tensor, steering: LayerSteering,
                        token_scale: torch.Tensor) -> torch.Tensor:
    """Apply scaled directional ablation to one rank's shared-expert partial before reduction."""
    direction = steering.direction
    projected = output.float()
    dot = (projected @ direction).unsqueeze(1)
    steered = (
        projected - token_scale * steering.alpha * dot * direction.unsqueeze(0)
    ).to(output.dtype)
    return torch.where(token_scale == 0.0, output, steered)
