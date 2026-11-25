# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared recipe-free planning and PreparationManifest test fixtures."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

from google.protobuf import json_format

from ignition import planning
from ignition.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults, Speculation
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto

GIB = 1 << 30
MIB = 1 << 20
DRAFTING = Speculation(draft_tokens=3)


def make_recipe(name="fixture", *, device_gib=60, ranks=(1, 2, 4), speculation=DRAFTING,
                row_shard_prefill_world_sizes=(), joint_tuning=False):
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
    )


def make_gpu(index, free_gib=95, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}",
                        96 * GIB, int(free_gib * GIB), capability)


def decode_manifest_params(plan, *, compiled_forwards=True):
    """Build, round-trip through protobuf, verify key ordering, and index PreparationManifest parameters."""
    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=compiled_forwards)
    encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
    decoded = proto.PreparationManifest.FromString(encoded.SerializeToString(deterministic=True))
    keys = [parameter.key for parameter in decoded.parameters]
    assert keys == sorted(keys) and len(keys) == len(set(keys))
    return manifest, {parameter.key: parameter for parameter in decoded.parameters}
