# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 preparation admits complete explicit placements and counts physical copies."""

from dataclasses import replace

import pytest

from ignition.components.expert_placement import ExpertPlacement
from ignition.recipes.deepseek41.bootstrap import PreparationConfig, _collective_runtimes
from ignition.recipes.deepseek41.sm103 import SM103Config
from .test_deepseek41_pool import config


@pytest.fixture
def model():
    base = config()
    return replace(base, text={**base.text, "n_routed_experts": 384, "moe_intermediate_size": 2304,
        "num_nextn_predict_layers": 3, "dspark_block_size": 8, "dspark_noise_token_id": 1,
        "dspark_target_layer_ids": [2, 10, 20], "dspark_markov_rank": 256,
        "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3,
        "vocab_size": 129280, "compress_ratios": base.text["compress_ratios"] + [0, 0, 0]})


@pytest.fixture
def options(tmp_path):
    # These CPU tests validate admission and accounting, not library loading.
    library = tmp_path / "library.so"
    library.touch()
    target = ExpertPlacement(384, tuple(range(160, 384)), tuple(range(256)))
    draft = ExpertPlacement(128, (), tuple(range(128)))
    placements = tuple((f"layers.{i}.ffn", target) for i in range(40))
    placements += tuple((f"mtp.{i}.ffn", draft) for i in range(3))
    return SM103Config(placements, library, library)


def test_physical_accounting_includes_duplicate_host_and_hbm_copies(model, options):
    options.validate_model(model, dspark=True)
    per_expert = 3 * 5120 * 2304 * 17 // 32
    assert options.expert_bytes(model, dspark=True) == {
        "host": 40 * 224 * per_expert, "hbm": (40 * 256 + 3 * 128) * per_expert}
    assert options.placements[0][1].physical_rows == 480
    assert options.to_dict()["placements"]["layers.0.ffn"]["hbm"] == list(range(256))


@pytest.mark.parametrize("mutation", ["missing_target", "missing_draft", "wrong_count", "extra_draft"])
def test_incomplete_or_mismatched_model_coverage_fails_before_allocation(model, options, mutation):
    entries = dict(options.placements)
    if mutation == "missing_target":
        del entries["layers.0.ffn"]
    elif mutation == "missing_draft":
        del entries["mtp.0.ffn"]
    elif mutation == "wrong_count":
        entries["layers.0.ffn"] = ExpertPlacement(383, (), tuple(range(383)))
    with pytest.raises(ValueError, match="every enabled"):
        replace(options, placements=tuple(entries.items())).validate_model(model, dspark=mutation != "extra_draft")


def test_tp1_requires_explicit_local_sm103_configuration(options, tmp_path):
    with pytest.raises(ValueError, match="distinct GPUs"):
        PreparationConfig(devices=(0,))
    with pytest.raises(ValueError, match="explicit libraries"):
        PreparationConfig(devices=(0,), sm103=options)
    with pytest.raises(ValueError, match="distinct GPUs"):
        PreparationConfig(devices=(0, 1, 2, 3), sm103=options, kernel_cache=tmp_path)
    with pytest.raises(ValueError, match="one local GPU"):
        PreparationConfig(devices=(0,), hosts=("gracie",), sm103=options, kernel_cache=tmp_path)
    with pytest.raises(ValueError, match="expanded native expert scales"):
        PreparationConfig(devices=(0,), sm103=options, kernel_cache=tmp_path, expert_scale_compression="csf")
    configuration = PreparationConfig(devices=(0,), sm103=options, kernel_cache=tmp_path)
    assert configuration.draft_tokens == 5
    assert _collective_runtimes(configuration, None, None, coordinator=None,
        core_storage=None, verification_samples=configuration.decode_tokens) == (None,) * 5


@pytest.mark.parametrize("experts,host,hbm", [(4, (0, 1), (2,)), (4, (0, 1), (1, 2, 4)),
    (4, (0, 0, 1), (2, 3)), (4, (), (0, 1, 2, True)), (4, (0, 1, 2, 3), ())])
def test_invalid_physical_expert_rows_are_rejected(experts, host, hbm):
    with pytest.raises(ValueError, match="complete canonical coverage"):
        ExpertPlacement(experts, host, hbm)
