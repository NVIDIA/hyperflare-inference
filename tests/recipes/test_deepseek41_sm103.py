# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 preparation admits complete explicit placements and counts physical copies."""

from dataclasses import replace
import json
import struct
import subprocess
import sys

import pytest

from ignition.components.expert_placement import ExpertPlacement
from ignition.recipes.deepseek41.bootstrap import PreparationConfig, _collective_runtimes
from ignition.recipes.deepseek41.sm103 import SM103Config
from .test_deepseek41_pool import config
from .test_deepseek41_csf_config import configuration


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


@pytest.fixture
def snapshot(tmp_path, configuration):
    """Small synthetic payloads exercise header accounting, not model execution."""
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "config.json").write_text(json.dumps(configuration))
    tensors = {
        "layers.0.attn.wq.weight": ("F8_E4M3", 1024),
        "layers.0.attn.wq.scale": ("U8", 32),
        "layers.0.ffn.experts.0.w1.weight": ("U8", 16),
        "layers.0.engram.embed.weight": ("U8", 32),
        "mtp.0.attn.wkv.weight": ("F8_E4M3", 1024),
    }
    offset, header = 0, {}
    for name, (dtype, size) in tensors.items():
        header[name] = dict(dtype=dtype, shape=[size], data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return path


def test_configuration_roundtrip_resolves_libraries_and_preserves_row_order(options, tmp_path):
    value = options.to_dict()
    value["flashmla_library"] = value["deepselect_library"] = "library.so"
    path = tmp_path / "sm103.json"
    path.write_text(json.dumps(value))
    assert SM103Config.load(path) == options


@pytest.mark.parametrize("tracing", [False, True])
def test_single_rank_recipe_passes_placement_without_starting_peers(options, snapshot, tmp_path, monkeypatch, tracing):
    from types import SimpleNamespace
    from ignition import planning
    from ignition.recipes.deepseek41 import bootstrap
    from ignition.recipes.deepseek41.recipe import RECIPE, usage
    from ignition.recipes.declaration import RankUsage
    from ignition.preparation import rank_group

    recipe, checkpoint = options.planning_inputs(RECIPE, RECIPE.checkpoints[0], snapshot, dspark=True)
    gpu = planning.Gpu(0, "GB300", "GPU-test", 256 << 30, 250 << 30, (10, 3))
    plan = replace(planning.plan(recipe, checkpoint, (gpu,), selected=(0,), concurrency=1,
        reserve_bytes=0, speculation=recipe.speculation, prefill_chunk=256, parallax_routing=tracing), sm103=options)
    prepared = SimpleNamespace(memory_report=dict(pages=64, free_device_bytes=1 << 30))
    storage = object()
    calls = []

    def prepare_rank(path, config, *, rank, group, progress, core_storage):
        calls.append((path, config, rank, group, core_storage))
        return prepared

    monkeypatch.setattr(bootstrap, "prepare_rank", prepare_rank)
    monkeypatch.setattr(rank_group, "prepare_group", lambda *a, **kw: pytest.fail("TP1 started peer orchestration"))
    assert recipe.prepare(snapshot, plan, compile_forwards=True, kernel_cache=tmp_path,
                          progress=None, core_storage=storage) is prepared
    assert len(calls) == 1
    path, config, rank, group, owner = calls[0]
    assert path == snapshot and rank == 0 and group is None and owner is storage
    assert config.sm103 == options and config.devices == (0,) and config.dspark
    assert config.parallax_routing is tracing
    assert usage(prepared) == (RankUsage(64, 1 << 30),)
    peers = SimpleNamespace(memory_report={"ranks": [prepared.memory_report] * 3})
    assert usage(peers) == (RankUsage(64, 1 << 30),) * 3


def test_routing_capture_rejects_multiple_ranks_and_non_boolean_selection(options, tmp_path):
    with pytest.raises(ValueError, match="one GPU"):
        PreparationConfig(parallax_routing=True)
    with pytest.raises(TypeError, match="boolean"):
        PreparationConfig(parallax_routing=1)
    config = PreparationConfig(devices=(0,), sm103=options, kernel_cache=tmp_path, parallax_routing=True)
    assert config.dspark and config.packed_speculative_shapes and config.parallax_routing


@pytest.mark.parametrize("change", ["version", "unknown", "bool_cluster", "duplicate", "incomplete"])
def test_invalid_json_configuration_is_rejected(options, tmp_path, change):
    value = options.to_dict()
    if change == "version":
        value["version"] = True
    elif change == "unknown":
        value["ignored"] = 1
    elif change == "bool_cluster":
        value["host_lane_clusters"] = True
    elif change == "incomplete":
        value["placements"]["layers.0.ffn"]["host"] = []
    text = json.dumps(value)
    if change == "duplicate":
        text = '{"version": 1, ' + text[1:]
    path = tmp_path / "sm103.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        SM103Config.load(path)


@pytest.mark.parametrize("drafting", [False, True])
def test_planning_counts_native_experts_and_physical_cache_layout(options, snapshot, drafting):
    from ignition.recipes.deepseek41.config import ModelConfig
    from ignition.recipes.deepseek41.pool import continuation_layout, page_bytes
    from ignition.recipes.deepseek41.recipe import RECIPE

    if not drafting:
        options = replace(options, placements=options.placements[:40])
    recipe, checkpoint = options.planning_inputs(RECIPE, RECIPE.checkpoints[0], snapshot, dspark=drafting)
    model = ModelConfig.load(snapshot)
    tier = options.expert_bytes(model, dspark=drafting)
    # FP8 projections reserve expanded scales and round up to allocation alignment.
    assert checkpoint.device_bytes == tier["hbm"] + 1536 + 512 + (1536 + 1056 if drafting else 0)
    assert checkpoint.mapped_host_bytes == tier["host"]
    assert checkpoint.expansion_bytes == 0
    assert not checkpoint.scale_compressed_device_bytes
    costs = recipe.ranks[1] if drafting else recipe.plain_ranks[1]
    assert costs.state_row_bytes == continuation_layout(model, dspark=drafting, alignment=512).row_bytes
    assert costs.page_bytes == page_bytes(model, alignment=512)
    assert 1 not in RECIPE.ranks  # Explicit placement must not change ordinary TP admission.


def test_cli_uses_gb300_and_records_placement_without_importing_torch(options, snapshot, tmp_path):
    path = tmp_path / "sm103.json"
    path.write_text(json.dumps(options.to_dict()))
    # A separate process checks the actual planning boundary before CUDA setup.
    script = '''
import json, sys
from pathlib import Path
from google.protobuf.json_format import ParseDict
from ignition import planning
from ignition.cli.config import Configuration, Setting
from ignition.cli.preparation import Host, derive
from ignition.errors import UsageError
from ignition.recipes import ResolvedModel
from ignition.recipes.deepseek41.recipe import RECIPE
from ignition.preparation.rebuild_manifest import from_plan
from ignition.preparation.runtime_core_pb2 import RebuildManifest
from ignition.preparation import host_memory
host_memory.available_bytes = lambda: 512 << 30
root, snapshot, placement = map(Path, sys.argv[1:])
configuration = Configuration(root / "config.toml", {}, exists=True)
gpus = (planning.Gpu(0, "RTX", "GPU-rtx", 96 << 30, 95 << 30, (12, 0)),
        planning.Gpu(1, "GB300", "GPU-gb300", 256 << 30, 250 << 30, (10, 3)))
model = ResolvedModel(RECIPE, RECIPE.checkpoints[0], snapshot)
arguments = dict(concurrency=Setting(1, "test"), speculation=RECIPE.speculation,
                 prefill_chunk=256, sm103_preparation=placement)
for selected in (None, "1"):
    plan = derive(configuration, model, Host(gpus, None), gpus=selected, **arguments)
    assert tuple(g.index for g in plan.gpus) == (1,)
    assert plan.speculation == RECIPE.speculation
    assert plan.expert_scale_compression is None
    assert plan.to_dict()["sm103_preparation"] == plan.sm103.to_dict()
    manifest = ParseDict(from_plan(model, plan, compiled_forwards=True), RebuildManifest())
    restored = RebuildManifest.FromString(manifest.SerializeToString())
    assert json.loads(restored.sm103_preparation_json) == plan.sm103.to_dict()
for selected in ("0", "0,1", "2"):
    try:
        derive(configuration, model, Host(gpus, None), gpus=selected, **arguments)
    except UsageError as error:
        assert "one SM103 GPU" in str(error), str(error)
    else:
        raise AssertionError("invalid GPU accepted")
host_memory.available_bytes = lambda: 1 << 30
try:
    derive(configuration, model, Host(gpus, None), gpus="1", **arguments)
except UsageError as error:
    assert "insufficient host memory" in str(error), str(error)
else:
    raise AssertionError("host budget ignored")
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(snapshot), str(path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
