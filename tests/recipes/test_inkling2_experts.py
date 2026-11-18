# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inkling2 static expert calibration and tensor-parallel loading geometry."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.inkling2.experts import _expert_scale_vectors, load_experts


def _sources(prefixes):
    return {
        f"{prefix}.{projection}.{suffix}": torch.tensor(value)
        for prefix in prefixes for projection in ("up_proj", "gate_proj", "down_proj")
        for suffix, value in (("weight_scale_2", .25), ("input_scale", .5))
    }


@pytest.mark.parametrize("rank", range(4))
def test_expert_loading_declares_shared_a4_a16_owner_and_exact_tp_partition(rank):
    prefixes = tuple(f"model.language_model.layers.47.mlp.experts.{expert}" for expert in range(384))
    loader = SimpleNamespace(_sources=_sources(prefixes), load_fp4_experts=lambda names, **options: (names, options))
    names, options = load_experts(loader, 47, shard=TensorParallelShard(rank, 4))
    assert names == prefixes
    assert options["intermediate_range"] == (rank * 576, (rank + 1) * 576)
    assert options["hidden_size"] == 4608 and options["intermediate_size"] == 2304
    assert options["activation_mode"] == "a4" and options["activation"] == "situ"
    assert options["a16_max_tokens"] == 64
    assert [vector.tolist() for vector in options["scale_vectors"]] == [[.25] * 384] * 2 + [[2.] * 384] * 2


@pytest.mark.parametrize("invalid", ("gate_weight", "gate_input", "shared_input", "zero", "nan", "dtype", "overflow"))
def test_expert_calibration_rejects_incompatible_static_scales(invalid):
    prefixes = ("expert0", "expert1")
    sources = _sources(prefixes)
    if invalid == "gate_weight":
        sources["expert0.gate_proj.weight_scale_2"] = torch.tensor(.125)
    elif invalid == "gate_input":
        sources["expert0.gate_proj.input_scale"] = torch.tensor(.25)
    elif invalid == "shared_input":
        for projection in ("gate_proj", "up_proj"):
            sources[f"expert1.{projection}.input_scale"] = torch.tensor(.25)
    elif invalid == "dtype":
        sources["expert0.down_proj.input_scale"] = torch.tensor(.5, dtype=torch.bfloat16)
    else:
        sources["expert0.down_proj.input_scale"] = torch.tensor({"zero": 0., "nan": float("nan"), "overflow": 1.e-40}[invalid])
    with pytest.raises(ValueError):
        _expert_scale_vectors(sources, prefixes)
