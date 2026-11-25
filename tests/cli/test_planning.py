# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe-free planning, world-size validation, and PreparationManifest serialization."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
from google.protobuf import json_format

from ignition import planning
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, Speculation
from ignition.errors import CapacityError, UsageError
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto

GIB = 1 << 30
MIB = 1 << 20
DRAFTING = Speculation(draft_tokens=3)


def _recipe(name="fixture", *, device_gib=60, ranks=(1, 2, 4), speculation=DRAFTING,
            row_shard_prefill_world_sizes=(), joint_tuning=False, target_head_quantizations=()):
    checkpoint = Checkpoint(
        alias=f"{name}-alias",
        repository=f"example/{name}",
        revision="ab12cd34" + "0" * 32,
        architecture="FixtureForCausalLM",
        stored_bytes=device_gib * GIB,
        device_bytes=device_gib * GIB,
        draft_bytes=2 * GIB if speculation is not None else 0,
    )
    return Recipe(
        name=name,
        served_model_name=name,
        checkpoints=(checkpoint,),
        context=262144,
        page_tokens=64,
        concurrency=4,
        max_concurrency=64,
        max_draft_tokens=7 if speculation is not None else 0,
        prefill_chunk=4096,
        ranks=MappingProxyType({size: RankCosts(100 * MIB // size, 2 * MIB // size, 4 * GIB) for size in ranks}),
        plain_ranks=MappingProxyType(
            {size: RankCosts(80 * MIB // size, 3 * MIB // (2 * size), 3 * GIB) for size in ranks}
            if speculation is not None else {}
        ),
        prepare=lambda *args, **kwargs: None,
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=-1),
        speculation=speculation,
        joint_tuning=joint_tuning,
        row_shard_prefill_world_sizes=row_shard_prefill_world_sizes,
        target_head_quantizations=target_head_quantizations,
    )


def _gpu(index, free_gib=95, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}",
                        96 * GIB, int(free_gib * GIB), capability)


def test_recipe_validates_row_shard_prefill_world_sizes():
    valid = _recipe(ranks=(1, 2, 4), row_shard_prefill_world_sizes=(2, 4))
    assert valid.row_shard_prefill_world_sizes == (2, 4)

    for bad_sizes in ((1,), (3,), (0,), (-2,), (True,)):
        with pytest.raises(ValueError, match="row_shard_prefill_world_sizes"):
            _recipe(ranks=(1, 2, 4), row_shard_prefill_world_sizes=bad_sizes)


def test_row_shard_prefill_validation_automatic_sizing_and_rebuild_manifest(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    unsupported = _recipe("plain", ranks=(1, 2, 4), row_shard_prefill_world_sizes=())
    supported = _recipe("sharded", ranks=(1, 2, 4), row_shard_prefill_world_sizes=(2, 4), joint_tuning=True)
    tp34 = _recipe("tp34", ranks=(3, 4), row_shard_prefill_world_sizes=(4,))

    pair_gpus = (_gpu(0), _gpu(1))
    settings = dict(selected=(0, 1), concurrency=4, reserve_bytes=0, speculation=DRAFTING,
                    online_mxfp8_scope="all")
    ordinary = planning.plan(supported, supported.checkpoints[0], pair_gpus, **settings)
    sharded = planning.plan(supported, supported.checkpoints[0], pair_gpus, row_shard_prefill=True, **settings)
    assert not ordinary.row_shard_prefill and ordinary.online_mxfp8_scope == "all"
    assert sharded.row_shard_prefill and sharded.online_mxfp8_scope == "all"
    assert sharded.to_dict()["row_shard_prefill"] is True
    assert replace(sharded, row_shard_prefill=False) == ordinary

    fitted = planning.fit(supported, supported.checkpoints[0], pair_gpus, concurrency=4,
                          reserve_bytes=0, speculation=DRAFTING, online_mxfp8_scope="all",
                          row_shard_prefill=True)
    assert fitted.row_shard_prefill

    for candidate in (ordinary, sharded, fitted):
        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), candidate, compiled_forwards=True)
        assert ("--row-shard-prefill" in manifest["reproduction_args"]) == candidate.row_shard_prefill
        encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString())
        keys = [parameter.key for parameter in decoded.parameters]
        assert keys == sorted(keys) and len(keys) == len(set(keys))
        by_key = {parameter.key: parameter for parameter in decoded.parameters}
        assert by_key["row_shard_prefill"].scope == proto.PREPARATION_PARAMETER_SCOPE_DEPLOYMENT
        assert by_key["row_shard_prefill"].value.bool_value == candidate.row_shard_prefill
        assert by_key["online_mxfp8"].scope == proto.PREPARATION_PARAMETER_SCOPE_TARGET_WEIGHT
        assert by_key["online_mxfp8_scope"].scope == proto.PREPARATION_PARAMETER_SCOPE_TARGET_WEIGHT
        assert by_key["online_mxfp8_scope"].value.string_value == "all"
        assert by_key["speculation.has_draft"].scope == proto.PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT
        assert by_key["speculation.has_draft"].value.bool_value is True

    # Unsupported recipe is rejected explicitly by alias.
    with pytest.raises(UsageError, match="--row-shard-prefill is not supported by plain-alias"):
        planning.plan(unsupported, unsupported.checkpoints[0], pair_gpus, row_shard_prefill=True, **settings)

    # TP1 is rejected on both TP1-capable and multi-GPU-only recipes.
    with pytest.raises(UsageError, match="--row-shard-prefill requires tensor parallelism"):
        planning.plan(supported, supported.checkpoints[0], (_gpu(0),), selected=(0,),
                      concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)
    with pytest.raises(UsageError, match="--row-shard-prefill requires tensor parallelism"):
        planning.plan(tp34, tp34.checkpoints[0], (_gpu(0), _gpu(1), _gpu(2), _gpu(3)), selected=(0,),
                      concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)

    # Unsupported TP3 is rejected explicitly while automatic planning skips TP3 and selects TP4.
    four_gpus = tuple(_gpu(i) for i in range(4))
    with pytest.raises(UsageError, match="--row-shard-prefill is not supported at TP3 for tp34-alias; supported tensor-parallel sizes: 4"):
        planning.plan(tp34, tp34.checkpoints[0], four_gpus[:3], selected=(0, 1, 2),
                      concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)
    auto_tp4 = planning.plan(tp34, tp34.checkpoints[0], four_gpus, selected=None,
                             concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)
    assert auto_tp4.world_size == 4 and auto_tp4.row_shard_prefill
    with pytest.raises(CapacityError, match="tp34-alias needs 4"):
        planning.plan(tp34, tp34.checkpoints[0], four_gpus[:3], selected=None,
                      concurrency=4, reserve_bytes=0, speculation=DRAFTING, row_shard_prefill=True)

    # SM103 and --joint-tuning are rejected.
    with pytest.raises(UsageError, match="SM103"):
        planning.plan(supported, supported.checkpoints[0], (_gpu(0, 95, (10, 3)), _gpu(1, 95, (10, 3))),
                      selected=(0, 1), concurrency=4, reserve_bytes=0, speculation=DRAFTING,
                      row_shard_prefill=True)
    with pytest.raises(UsageError, match="--row-shard-prefill cannot be combined with --joint-tuning"):
        replace(sharded, joint_tuning=True)


def test_preparation_manifest_resolves_scopes_and_dflash2_inheritance(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    dflash = Speculation(draft_tokens=3, method="dflash2")
    recipe = replace(
        _recipe("dflash", ranks=(1,), speculation=dflash),
        online_mxfp8=True,
        online_mxfp8_scopes=("all", "kda-input"),
    )
    gpu = (_gpu(0),)

    def decode_params(plan):
        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
        encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString(deterministic=True))
        keys = [parameter.key for parameter in decoded.parameters]
        assert keys == sorted(keys) and len(keys) == len(set(keys))
        return {parameter.key: parameter for parameter in decoded.parameters}

    # Inherited DFlash2 MXFP8 body quantization resolves explicitly when online_mxfp8 + all scope.
    inherited = planning.plan(recipe, recipe.checkpoints[0], gpu, selected=(0,), concurrency=4,
                              reserve_bytes=0, speculation=dflash, online_mxfp8=True, online_mxfp8_scope="all")
    params = decode_params(inherited)
    assert params["speculation.draft_body_quantization"].scope == proto.PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT
    assert params["speculation.draft_body_quantization"].value.string_value == "mxfp8"
    assert params["speculation.has_draft"].scope == proto.PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT
    assert params["speculation.has_draft"].value.bool_value is True

    # Scoped online_mxfp8 (kda-input) does not inherit MXFP8 for the draft body.
    scoped = planning.plan(recipe, recipe.checkpoints[0], gpu, selected=(0,), concurrency=4,
                           reserve_bytes=0, speculation=dflash, online_mxfp8=True, online_mxfp8_scope="kda-input")
    scoped_params = decode_params(scoped)
    assert "speculation.draft_body_quantization" not in scoped_params
    assert scoped_params["online_mxfp8_scope"].value.string_value == "kda-input"

    # Target-only export (--no-mtp) emits no draft-weight parameters.
    target_only = planning.plan(recipe, recipe.checkpoints[0], gpu, selected=(0,), concurrency=4,
                                reserve_bytes=0, speculation=None, online_mxfp8=True, online_mxfp8_scope="all")
    target_params = decode_params(target_only)
    assert not any(
        parameter.scope == proto.PREPARATION_PARAMETER_SCOPE_DRAFT_WEIGHT
        for parameter in target_params.values()
    )

    # End-to-end handoff through Heliopause's native serving-metadata encoder.
    import json
    import flare_heliopause

    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), inherited, compiled_forwards=True)
    server = {
        "model_id": "example/dflash",
        "frontend": {"eos_token_ids": [1]},
        "sampling_defaults": {
            "seed": 0, "temperature": 1.0, "top_k": -1, "top_p": 1.0,
            "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        },
        "assets": {"chat_template": "t", "tokenizer": "k"},
        "preparation_manifest": manifest,
    }
    encoded_index = flare_heliopause.export_serving(
        proto.ModelIndex(version=4).SerializeToString(),
        json.dumps(server),
    )
    decoded_index = proto.ModelIndex.FromString(encoded_index)
    assert decoded_index.HasField("preparation")
    assert not decoded_index.HasField("rebuild")
    expected = json_format.ParseDict(manifest, proto.PreparationManifest())
    assert decoded_index.preparation.SerializeToString(deterministic=True) == expected.SerializeToString(deterministic=True)


def test_target_head_quantization_validation_and_preparation_manifest(monkeypatch):
    monkeypatch.setattr("ignition.preparation.source_identity.source_identity", lambda: "test-source")
    unsupported = _recipe("plain", ranks=(1, 2), target_head_quantizations=())
    supported = _recipe("quanthead", ranks=(1, 2), target_head_quantizations=("mxfp8",))

    with pytest.raises(ValueError, match="target_head_quantizations"):
        _recipe("bad", ranks=(1,), target_head_quantizations=("nvfp4",))

    gpu0 = (_gpu(0),)
    base = planning.plan(supported, supported.checkpoints[0], gpu0, selected=(0,),
                         concurrency=4, reserve_bytes=0, speculation=DRAFTING)
    assert base.target_head_quantization is None
    assert "target_head_quantization" not in base.to_dict()

    quantized = replace(base, target_head_quantization="mxfp8")
    assert quantized.target_head_quantization == "mxfp8"
    assert quantized.to_dict()["target_head_quantization"] == "mxfp8"

    for candidate in (base, quantized):
        manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), candidate, compiled_forwards=True)
        has_flag = "--target-head-quantization" in manifest["reproduction_args"]
        assert has_flag == (candidate.target_head_quantization is not None)
        if candidate.target_head_quantization is not None:
            idx = manifest["reproduction_args"].index("--target-head-quantization")
            assert manifest["reproduction_args"][idx + 1] == candidate.target_head_quantization
        encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
        decoded = proto.PreparationManifest.FromString(encoded.SerializeToString(deterministic=True))
        assert list(decoded.reproduction_args) == manifest["reproduction_args"]
        keys = [parameter.key for parameter in decoded.parameters]
        assert keys == sorted(keys) and len(keys) == len(set(keys))
        by_key = {parameter.key: parameter for parameter in decoded.parameters}
        if candidate.target_head_quantization is None:
            assert "target_head_quantization" not in by_key
        else:
            param = by_key["target_head_quantization"]
            assert param.scope == proto.PREPARATION_PARAMETER_SCOPE_TARGET_WEIGHT
            assert param.value.string_value == "mxfp8"

    plain_plan = planning.plan(unsupported, unsupported.checkpoints[0], gpu0, selected=(0,),
                               concurrency=4, reserve_bytes=0, speculation=DRAFTING)
    with pytest.raises(UsageError, match="--target-head-quantization mxfp8: plain-alias"):
        replace(plain_plan, target_head_quantization="mxfp8")
