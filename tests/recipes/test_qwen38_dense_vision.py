# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense Qwen vision preparation uses the Qwen3.5 feature width."""

import torch

from ignition.recipes.qwen38_dense.contract import VISION
from ignition.recipes.qwen38_dense.vision import VISION_WIDTH, VisionScratch


def test_dense_vision_scratch_uses_qwen3_5_feature_width():
    assert VISION_WIDTH == 5120
    assert VISION["patch_capacities"] == [256, 1024, 4096, 16384]
    scratch = VisionScratch.allocate(256, torch.device("cpu"))
    assert scratch.patches.shape == (256, 1536)
    assert scratch.merger_intermediate.shape == (64, 4608)
    assert scratch.output.shape == (64, 5120)
