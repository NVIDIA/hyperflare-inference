# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest

from ignition.preparation import inspect_qwen38_checkpoint
from ignition.preparation.checkpoint import CheckpointError
from ignition.recipes.qwen38 import create_model_tree
from ignition.recipes.qwen38.checkpoint_schema import TRANSFORMERS_VERSION


def test_exact_checkpoint_constructs_canonical_meta_tree_without_cuda():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    snapshot = Path("/data/checkpoint-releases/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD")
    if not snapshot.exists():
        pytest.skip("exact Qwen3.8 checkpoint is not cached")
    initialized = torch.cuda.is_initialized()
    model = create_model_tree(inspect_qwen38_checkpoint(snapshot))
    assert type(model).__name__ == "Qwen4ExpForConditionalGeneration"
    assert len(model.model.language_model.layers) == 48
    assert len(model.model.visual.blocks) == 27
    assert model.model.language_model.layers[1].ple is not None
    assert model.model.language_model.layers[3].self_attn.head_dim == 256
    assert sum(p.numel() for p in model.parameters()) == 177392830576
    assert all(p.device.type == "meta" and not p.requires_grad for p in model.parameters())
    assert all(p.device.type == "meta" for p in model.buffers())
    assert not model.training
    assert torch.cuda.is_initialized() == initialized


def test_model_tree_requires_the_pinned_transformers_release(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = SimpleNamespace(snapshot=tmp_path, metadata_sha256={"config.json": "0" * 64})
    with pytest.raises(CheckpointError, match="config.json changed"):
        create_model_tree(manifest)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0")
    with pytest.raises(RuntimeError, match=f"Qwen preparation requires Transformers {TRANSFORMERS_VERSION}, found 0.0.0"):
        create_model_tree(manifest)


def test_required_transformers_release_matches_the_prepare_pin():
    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    pins = [d.split("==")[1] for d in pyproject["project"]["optional-dependencies"]["prepare"]
            if d.startswith("transformers")]
    assert pins == [TRANSFORMERS_VERSION]
