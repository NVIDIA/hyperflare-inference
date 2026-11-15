# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling preparation preserves the declared TP4 and DSpark capacities."""

from dataclasses import replace

import pytest

from ignition.cli import _parser, preparation
from ignition.recipes import ResolvedModel
from ignition.recipes.inkling2.recipe import RECIPE
from tests.cli.test_planning import derive, gpu


def test_inkling_defaults_cover_tp4_seven_drafts_and_vision():
    assert RECIPE.tensor_parallel == (4,)
    assert (RECIPE.context, RECIPE.concurrency, RECIPE.prefill_chunk) == (1048576, 8, 4096)
    assert (RECIPE.max_images, RECIPE.image_bytes) == (4, 32 << 20)
    model = ResolvedModel(RECIPE, RECIPE.checkpoints[0])
    args = _parser().parse_args(["export", model.checkpoint.alias])
    speculation = preparation.speculation(args, model)
    assert (speculation.draft_tokens, speculation.token_capacity, speculation.graph_budget) == (7, 64, 64)
    assert speculation.packed_verification
    assert speculation.widths == tuple(range(1, 9))


def test_inkling_adapter_preserves_uncompiled_untuned_qualification(monkeypatch, tmp_path):
    from ignition.recipes.inkling2 import bootstrap

    devices = tuple(gpu(i, 95) for i in range(4))
    plan = replace(derive(model=RECIPE, gpus=devices, selected=(0, 1, 2, 3), concurrency=8),
                   autotune_effort="off")
    captured = []
    configuration = bootstrap.PreparationConfig

    class ConfigurationCaptured(Exception):
        pass

    def capture_config(**kwargs):
        captured.append(configuration(**kwargs))
        raise ConfigurationCaptured

    monkeypatch.setattr(bootstrap, "PreparationConfig", capture_config)
    with pytest.raises(ConfigurationCaptured):
        RECIPE.prepare(tmp_path, plan, compile_forwards=False, kernel_cache=None, progress=None)
    config = captured.pop()
    assert config.devices == (0, 1, 2, 3)
    assert config.autotune_effort == "off"
    assert not config.compile_decoder
    assert config.context == 1048576
    assert config.max_active == 8
    assert config.drafting and config.vision
    assert config.speculative_widths == tuple(range(1, 9))
    assert config.speculative_token_capacity == 64
    assert config.speculative_graph_budget == 64
