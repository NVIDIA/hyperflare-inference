# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling2 patch uploads and generation-scoped image feature publication."""

from dataclasses import dataclass

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication

FEATURES = 16384
PATCH_CAPACITIES = (64, 256, 1024, 4096, 16384, 65536)


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_num_features: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        patches = max(PATCH_CAPACITIES)
        return cls(torch.zeros((patches, 1200), dtype=torch.float32, device=device),
                   torch.zeros((patches + 1, 2), dtype=torch.int64, device=device),
                   torch.zeros((patches, 4), dtype=torch.int64, device=device),
                   torch.zeros((patches, 4), dtype=torch.float32, device=device),
                   torch.zeros(2, dtype=torch.int32, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device),
                   torch.full((1,), -1, dtype=torch.int32, device=device),
                   torch.zeros(1, dtype=torch.int64, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class PreparedVisionGeneration:
    def __init__(self, encoder, inputs, state, completion, frame, sequence):
        self.encoder, self.inputs, self.state = encoder, inputs, state
        self.publish = PreparedFeaturePublication(state, inputs, completion, frame, sequence)

    def __call__(self):
        self.forward()

    def forward(self):
        from .vision import VisionMetadata

        e, i, s = self.encoder, self.inputs, self.state
        r, capacity = s.requests, e.capacity
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, features, offset = i.vision_num_patches, i.vision_num_features, i.vision_feature_offset
        current = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        current &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (count > 0) & (count <= capacity) & (count % 4 == 0) & (features == count // 4)
        valid &= (offset >= 0) & (offset + features <= s.arena.shape[1])
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count + 1)
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        patch_live = torch.arange(capacity, device=count.device) < count
        row_live = torch.arange(capacity + 1, device=count.device) < count + 1
        pixels, positions = i.vision_pixel_values[:capacity], i.vision_position_ids[:capacity + 1]
        indices = i.vision_interpolation_indices[:capacity]
        weights = i.vision_interpolation_weights[:capacity]
        valid &= (~patch_live[:, None] | torch.isfinite(pixels)).all()
        valid &= (positions[0] == 0).all()
        valid &= (~row_live[1:, None] | ((positions[1:] > 0) & (positions[1:] <= count))).all()
        valid &= (~patch_live[:, None] | ((indices >= 0) & (indices < 1024))).all()
        valid &= (~patch_live[:, None] | (torch.isfinite(weights) & (weights >= 0) & (weights <= 1))).all()
        valid &= (~patch_live | ((weights.sum(-1) - 1).abs() <= 1e-6)).all()
        errors = (~valid).to(torch.int32)
        m = VisionMetadata(torch.where(valid & row_live[:, None], positions, 0),
            torch.where(valid & patch_live[:, None], indices, 0),
            torch.where(valid & patch_live[:, None], weights, 0),
            torch.where(valid, i.vision_cu_seqlens, 0), torch.where(valid, count, 0),
            torch.where(valid, features, 0))
        source = pixels.to(torch.bfloat16)
        source.masked_fill_(~(valid & patch_live[:, None]), 0)
        output = e(source, m)
        feature_live = torch.arange(capacity // 4, device=count.device) < features
        finite = (~feature_live[:, None] | torch.isfinite(output)).all()
        errors = errors | (~finite).int() * 64
        self.publish(output, m.num_features, errors)
        return output, errors
