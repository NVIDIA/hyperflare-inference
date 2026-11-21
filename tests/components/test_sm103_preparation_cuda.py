# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP1 bootstrap constructs shared protocol storage with target and draft capacities."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.expert_placement import ExpertPlacement
from ignition.recipes.deepseek41.bootstrap import PreparationConfig
from ignition.recipes.deepseek41.config import ModelConfig
from ignition.recipes.deepseek41.sm103 import SM103Config

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def test_target_and_draft_share_only_matching_protocol_geometries(tmp_path, megamoe_core_storage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    flashmla, deepselect = os.environ.get("FLARE_FLASHMLA_LIBRARY"), os.environ.get("FLARE_DEEPSELECT_LIBRARY")
    if not snapshot or not flashmla or not deepselect:
        pytest.skip("requires the checkpoint and prepared FlashMLA/DeepSelect libraries")
    model = ModelConfig.load(Path(snapshot))
    target = ExpertPlacement(384, tuple(range(160, 384)), tuple(range(256)))
    draft = ExpertPlacement(128, (), tuple(range(128)))
    options = SM103Config(tuple((f"layers.{i}.ffn", target) for i in range(model.layers)) +
        tuple((f"mtp.{i}.ffn", draft) for i in range(model.draft.layers)), Path(flashmla), Path(deepselect))
    config = PreparationConfig(devices=(0,), sm103=options, kernel_cache=tmp_path,
        context=8192, prefill_chunk_size=256, max_active=32)
    layouts, attention = options.prepare(model, config, device=torch.device("cuda:0"),
                                         storage=megamoe_core_storage)
    assert len(layouts) == 43
    target_workspace = layouts["layers.0.ffn"].workspace
    draft_workspace = layouts["mtp.0.ffn"].workspace
    assert target_workspace is not draft_workspace
    assert all(layouts[f"layers.{i}.ffn"].workspace is target_workspace for i in range(40))
    assert all(layouts[f"mtp.{i}.ffn"].workspace is draft_workspace for i in range(3))
    assert target_workspace.geometry.experts == 480
    assert target_workspace.geometry.capacity >= 256
    assert draft_workspace.geometry.experts == 128
    assert draft_workspace.geometry.capacity >= 160
    assert target_workspace.buffer.data_ptr() != draft_workspace.buffer.data_ptr()
    assert all(table.shape == (8192, 64) for table in attention.cos_sin)
    assert all(torch.isfinite(table).all() for table in attention.cos_sin)
    assert all(layout.storage is megamoe_core_storage for layout in layouts.values())
