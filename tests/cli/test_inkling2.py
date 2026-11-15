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
    assert RECIPE.max_concurrency == 32
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


def test_inkling_declares_quantized_proposal_heads_with_checkpoint_precision_as_default():
    assert RECIPE.draft_head_quantizations == ("nvfp4", "mxfp8")
    assert RECIPE.markov_head_quantizations == ("fp8",)
    assert RECIPE.speculation.draft_head_quantization is None
    assert RECIPE.speculation.markov_head_quantization is None
    model = ResolvedModel(RECIPE, RECIPE.checkpoints[0])
    args = _parser().parse_args(["export", model.checkpoint.alias, "--draft-head-quantization", "nvfp4",
                                 "--markov-head-quantization", "fp8"])
    speculation = preparation.speculation(args, model)
    assert (speculation.draft_head_quantization, speculation.markov_head_quantization) == ("nvfp4", "fp8")
    assert speculation.draft_tokens == 7


@pytest.mark.parametrize("draft_head,markov_head", [(None, None), ("nvfp4", "fp8"), ("mxfp8", None), (None, "fp8")])
def test_inkling_adapter_passes_proposal_head_formats_to_preparation(monkeypatch, tmp_path, draft_head, markov_head):
    from ignition.recipes.inkling2 import bootstrap

    devices = tuple(gpu(i, 95) for i in range(4))
    speculation = replace(RECIPE.speculation, draft_head_quantization=draft_head,
                          markov_head_quantization=markov_head)
    plan = derive(model=RECIPE, gpus=devices, selected=(0, 1, 2, 3), concurrency=8, speculation=speculation)
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
    assert (config.draft_head_quantization, config.markov_head_quantization) == (draft_head, markov_head)


def test_inkling_plan_rejects_an_nvfp4_recurrent_correction_head():
    from ignition.errors import UsageError

    devices = tuple(gpu(i, 95) for i in range(4))
    speculation = replace(RECIPE.speculation, markov_head_quantization="nvfp4")
    with pytest.raises(UsageError, match="--markov-head-quantization nvfp4"):
        derive(model=RECIPE, gpus=devices, selected=(0, 1, 2, 3), concurrency=8, speculation=speculation)


@pytest.mark.parametrize("arguments,message", [
    (dict(draft_head_quantization="fp8"), "draft head quantization"),
    (dict(markov_head_quantization="nvfp4"), "correction head quantization"),
    (dict(draft_head_quantization="nvfp4", drafting=False), "positive draft depth"),
    (dict(markov_head_quantization="fp8", speculative_widths=(1,)), "positive draft depth"),
])
def test_inkling_preparation_config_validates_proposal_head_formats(arguments, message):
    from ignition.recipes.inkling2.bootstrap import PreparationConfig

    with pytest.raises(ValueError, match=message):
        PreparationConfig(**arguments)
