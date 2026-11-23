# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 preparation geometry, page and state layout, and declared costs before loading weights."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.glm53_full.bootstrap import PreparationConfig
from ignition.recipes.glm53_full.recipe import RECIPE

SNAPSHOT = os.environ.get("FLARE_GLM53_FULL_SNAPSHOT")
requires_snapshot = pytest.mark.skipif(not SNAPSHOT, reason="requires FLARE_GLM53_FULL_SNAPSHOT")
WORLD = 8


@pytest.fixture(scope="module")
def config():
    from ignition.recipes.glm53_full.config import ModelConfig

    return ModelConfig.load(Path(SNAPSHOT))


def test_default_preparation_backs_one_chunk_and_two_pages_per_request():
    config = PreparationConfig()
    assert len(config.devices) == WORLD
    assert config.preparation_pages == 4096 // 256 + 2 * 8
    assert PreparationConfig(max_active=3, prefill_chunk_size=512).preparation_pages == 2 + 2 * 4
    assert config.tuning_state_rows == 9
    assert PreparationConfig(max_active=1).tuning_state_rows == 3


@pytest.mark.parametrize("values", [
    {"devices": ()}, {"devices": tuple(range(4))}, {"devices": (0,) * 8}, {"devices": (-1, *range(1, 8))},
    {"context": 513}, {"context": 2097152}, {"prefill_chunk_size": 4097},
    {"context": 256, "prefill_chunk_size": 512}, {"max_active": 65},
    {"snapshot_budget_bytes": 0}, {"headroom_bytes": -1}, {"prompt_logprobs": True},
])
def test_invalid_prepared_geometry_rejects_before_gpu_use(values):
    with pytest.raises((ValueError, TypeError)):
        PreparationConfig(**values)


@requires_snapshot
def test_pages_hold_a_latent_record_in_every_layer_and_an_index_key_in_selecting_layers(config):
    from ignition.recipes.glm53_full.pool import ContinuationPool, page_bytes, page_specs

    specs = dict(page_specs(config))
    selecting = [layer for layer in range(config.layers) if config.selects(layer)]
    assert len(selecting) == 21 and len(specs) == config.layers + len(selecting)
    assert page_bytes(config) == 256 * (78 * 656 + 21 * 132)
    pool = ContinuationPool.for_model(config, 3, 2, "cpu")
    assert pool.latent(5).shape == (8, 64, 656) and pool.index(5) is None
    assert pool.latent(6).shape == (8, 64, 656) and pool.index(6).shape == (8, 64 * 132)
    # A continuation row holds one partial page of every region beside the hidden state and token count.
    for name, (nbytes,) in specs.items():
        assert pool.region(name).shape == (3, nbytes)
    assert pool.last_hidden.shape == (3, 6144) and pool.last_hidden.dtype == torch.bfloat16
    assert pool.region("consumed_tokens").shape == (3, 1)
    for value in pool.pages.values():
        value.fill_(1)
    pool.backing.fill_(1)
    pool.clear_for_preparation()
    assert not pool.backing.count_nonzero() and not any(value.count_nonzero() for value in pool.pages.values())


@requires_snapshot
def test_declared_costs_equal_the_prepared_layout(config):
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.glm53_full.pool import continuation_layout, page_bytes
    from ignition.recipes.glm53_full.weights import tensor_partitions

    costs = RECIPE.costs(WORLD, None)
    layout = continuation_layout(config)
    assert costs.page_bytes == page_bytes(config)
    assert costs.state_row_bytes == layout.row_bytes and layout.row_bytes % 256 == 0
    assert RECIPE.page_tokens == 256 and RECIPE.tensor_parallel == (WORLD,)
    manifest = inspect_indexed_checkpoint(Path(SNAPSHOT))
    checkpoint, = RECIPE.checkpoints
    assert checkpoint.stored_bytes == sum(tensor.nbytes for tensor in manifest.tensors.values())
    assert checkpoint.draft_bytes == sum(tensor.nbytes for name, tensor in manifest.tensors.items()
                                         if name.startswith(f"model.layers.{config.draft_layer}."))

    def rank_bytes(rank):
        parts = tensor_partitions(config, manifest.tensors, TensorParallelShard(rank, WORLD), draft=False)
        return sum(tensor.nbytes if parts[name].axis is None
                   else tensor.nbytes // tensor.shape[parts[name].axis] * (parts[name].stop - parts[name].start)
                   for name, tensor in manifest.tensors.items() if parts[name] is not None)

    # The planner divides stored bytes and replicated copies evenly; ranks differ only by the head's last rows.
    planned = RECIPE.weights_bytes(checkpoint, WORLD, None)
    assert planned * WORLD == sum(rank_bytes(rank) for rank in range(WORLD))
    assert 0 <= rank_bytes(0) - planned < 1 << 20


def test_frontend_syntax_is_that_of_glm53_flash_without_rejected_media_tokens():
    """Helios frontend tests of this recipe use the `glm53` contract fixture with its rejected tokens cleared."""
    from ignition.recipes.glm53.contract import FRONTEND as flash
    from ignition.recipes.glm53_full.contract import FRONTEND

    assert FRONTEND == {**flash, "rejected_tokens": []}


def test_model_config_validates_attention_scale_and_positive_scalars():
    from ignition.recipes.glm53_full.config import ARCHITECTURE, ModelConfig, _REQUIRED

    valid = {
        **_REQUIRED,
        "architectures": [ARCHITECTURE],
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
        "num_hidden_layers": 2,
        "indexer_types": ["full", "shared"],
        "mlp_layer_types": ["dense", "sparse"],
        "rms_norm_eps": 1e-5,
        "routed_scaling_factor": 2.5,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                }
            },
        },
    }
    parsed = ModelConfig.from_dict(valid)
    assert parsed.qk_head_dim == 256 and parsed.norm_eps == 1e-5 and parsed.routed_scale == 2.5
    for bad in (
        {**valid, "qk_head_dim": 128},
        {k: v for k, v in valid.items() if k != "qk_head_dim"},
        {**valid, "rms_norm_eps": 0.0},
        {**valid, "rms_norm_eps": True},
        {**valid, "routed_scaling_factor": -1.0},
        {**valid, "routed_scaling_factor": None},
    ):
        with pytest.raises(ValueError):
            ModelConfig.from_dict(bad)


def test_single_request_index_tables_expand_without_copying():
    from ignition.recipes.glm53_full.indexer import index_tables, selected_records

    block_table = torch.tensor([[2, -1, 5]], dtype=torch.int32)
    requests = torch.zeros(4096, dtype=torch.int64)
    tables, row_pages = index_tables(block_table, requests)
    assert row_pages.shape == (4096, 12) and row_pages.stride(0) == 0
    assert row_pages.untyped_storage().data_ptr() == tables.untyped_storage().data_ptr()
    logical = torch.tensor([[65, 2, 260]], dtype=torch.int32)
    positions = torch.tensor([400], dtype=torch.int64)
    selection = selected_records(logical, positions, row_pages[:1])
    assert selection.records.tolist() == [[8 * 64 + 2, 9 * 64 + 1, -1]]
    assert selection.lengths.tolist() == [2]

