# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Small's preparation identity and strict checkpoint validation."""

import hashlib
import json
from pathlib import Path

import pytest

from ignition.recipes.inkling2.config import ModelConfig, TextConfig


@pytest.fixture
def small():
    return json.loads((Path(__file__).parent / "fixtures/inkling2_small_nvfp4_config.json").read_text())


def test_small_preparation_identity(small):
    model = ModelConfig.from_dict(small)
    assert type(model.text) is TextConfig
    assert model.text.cache_config is model.text
    identity = {"text": model.text.__dict__ | {"local_layers": sorted(model.text.local_layers)},
        "quantization": model.quantization, "headers": ["00" * 32], "world_size": 4,
        "decoder_layers": 48, "contract": "inkling2.paged-text.v1"}
    # Computed with the NVFP4 geometry fixture on upstream 25b49519.
    assert hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest() == (
        "f176fc401bd5c041af8487a122f2bd4e2629a54a547cdc4c7324f786f66c0e65")


@pytest.mark.parametrize("flag", ["pre_norm_pre_affine", "final_norm_pre_affine"])
def test_small_requires_explicit_affine_flags(small, flag):
    del small["text_config"][flag]
    with pytest.raises(ValueError, match=flag):
        ModelConfig.from_dict(small)


def test_small_rejects_parameter_groups_without_nvfp4(small):
    small["quantization_config"] = {"model": {"parameter_groups": [{"tensors": ["weight"]}]}}
    with pytest.raises(ValueError, match="NVFP4"):
        ModelConfig.from_dict(small)


def test_small_does_not_load_mini_precision_sidecar(small, tmp_path):
    del small["quantization_config"]
    (tmp_path / "config.json").write_text(json.dumps(small))
    (tmp_path / "hf_quant_config.json").write_text('{"quant_algo": "NVFP4"}')
    with pytest.raises(ValueError, match="NVFP4"):
        ModelConfig.load(tmp_path)


def test_small_rejects_loop_geometry(small):
    small["text_config"]["loop_config"] = dict(core_start=6, core_end=18, num_passes=3, gate_rank=64)
    with pytest.raises(ValueError, match="unsupported fields: loop_config"):
        ModelConfig.from_dict(small)


def test_small_rejects_mini_expert_geometry(small):
    small["text_config"].update(hidden_size=2304, intermediate_size=1152, n_routed_experts=128)
    with pytest.raises(ValueError, match="geometry"):
        ModelConfig.from_dict(small)
