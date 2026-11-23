# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Checkpoint.tensor_parallel and CheckpointOverlay contracts."""

from dataclasses import replace
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from ignition import planning
from ignition.cli import main
from ignition.cli.config import Configuration
from ignition.cli.serve import _publisher
from ignition.declaration import Checkpoint, CheckpointOverlay, RankCosts, Recipe, SamplingDefaults
from ignition.errors import CapacityError, UsageError
from ignition.preparation.checkpoint import (
    CheckpointError,
    checkpoint_assets_root,
    overlay_files_present,
    overlay_manifest_files,
    overlay_stored_bytes,
    read_overlay_manifest,
)
from ignition.preparation.protection import License
from ignition.registry import _snapshot_cached, resolve


def _validate_sample_inventory(manifest):
    expected = manifest.get("expected_paths")
    if expected is None:
        return
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest must declare files for expected_paths")
    listed = [entry.get("path") for entry in files if isinstance(entry, dict)]
    if len(listed) != len(set(listed)) or set(listed) != set(expected):
        raise ValueError("manifest files must match expected_paths exactly")


SAMPLE_OVERLAY = CheckpointOverlay(
    manifest="overlay-manifest.json",
    schema="example.overlay.v1",
    assets_subdir="carrier",
    required_files=("meta/transform.json",),
    validate_manifest=_validate_sample_inventory,
)


def _sample_recipe(*, overlay_first=False, overlay=SAMPLE_OVERLAY):
    plain = Checkpoint(
        alias="synth-model-plain",
        repository="example/Synth-Model-Plain",
        revision="1111111111111111111111111111111111111111",
        architecture="SynthModelForCausalLM",
        stored_bytes=10 << 30,
        device_bytes=10 << 30,
        draft_bytes=1 << 30,
    )
    overlaid = Checkpoint(
        alias="synth-model-overlay",
        repository="example/Synth-Model-Overlay",
        revision="2222222222222222222222222222222222222222",
        architecture="SynthModelForCausalLM",
        stored_bytes=12 << 30,
        device_bytes=12 << 30,
        expansion_bytes=1 << 30,
        draft_bytes=2 << 30,
        tensor_parallel=(4,),
        overlay=overlay,
    )
    checkpoints = (overlaid, plain) if overlay_first else (plain, overlaid)
    return Recipe(
        name="synth_model",
        served_model_name="Synth-Model",
        checkpoints=checkpoints,
        context=32768,
        page_tokens=128,
        concurrency=8,
        max_concurrency=16,
        max_draft_tokens=0,
        prefill_chunk=2048,
        ranks=MappingProxyType({
            1: RankCosts(1 << 20, 1 << 20, 1 << 28),
            2: RankCosts(1 << 20, 1 << 20, 1 << 28),
            4: RankCosts(1 << 20, 1 << 20, 1 << 28),
        }),
        prepare=lambda *args, **kwargs: None,
        usage=lambda prepared: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=0.95, top_k=20),
    )


def _gpus(count):
    return tuple(
        planning.Gpu(index, "Synthetic SM120 GPU", f"GPU-{index:04d}", 512 << 30, 512 << 30, (12, 0))
        for index in range(count)
    )


def _write_safetensors(path: Path, payload: bytes = b"\x01" * 16) -> int:
    header = json.dumps({
        "weight": {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]},
    }).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return len(payload)


def _build_overlay_dir(
    root: Path,
    *,
    manifest_doc=None,
    root_config=True,
    carrier_config=True,
    single_file_carrier=False,
    carrier_readme=None,
    sidecars=(("parts/part-0.bin", b"abcd"), ("parts/part-1.bin", b"efghij")),
    include_required=True,
):
    carrier = root / "carrier"
    carrier.mkdir(parents=True, exist_ok=True)
    config_text = json.dumps({"architectures": ["SynthModelForCausalLM"], "model_type": "synth"})
    if root_config:
        (root / "config.json").write_text(config_text)
    if carrier_config:
        (carrier / "config.json").write_text(config_text)
    if single_file_carrier:
        _write_safetensors(carrier / "model.safetensors", b"\x02" * 24)
        carrier_bytes = (carrier / "model.safetensors").stat().st_size
    else:
        payload_bytes = _write_safetensors(carrier / "shard-00001.safetensors", b"\x02" * 24)
        (carrier / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": payload_bytes},
            "weight_map": {"weight": "shard-00001.safetensors"},
        }))
        carrier_bytes = payload_bytes
    if carrier_readme is not None:
        (carrier / "README.md").write_text(carrier_readme)
    if include_required:
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "transform.json").write_text("{}")
    files_list = []
    for rel, content in sidecars:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        files_list.append({"path": rel, "bytes": len(content)})
    if manifest_doc is None:
        manifest_doc = {
            "schema": SAMPLE_OVERLAY.schema,
            "expected_paths": [rel for rel, _ in sidecars],
            "files": files_list,
        }
    (root / SAMPLE_OVERLAY.manifest).write_text(json.dumps(manifest_doc))
    return root, carrier_bytes, sum(len(content) for _, content in sidecars)


def test_checkpoint_tensor_parallel_validation_and_resolution():
    recipe = _sample_recipe()
    plain, overlaid = recipe.checkpoints
    assert recipe.checkpoint_tensor_parallel(plain) == (1, 2, 4)
    assert recipe.checkpoint_tensor_parallel(overlaid) == (4,)

    widened = replace(recipe, checkpoints=(plain, replace(overlaid, tensor_parallel=(2, 4))))
    assert widened.checkpoint_tensor_parallel(widened.checkpoints[1]) == (2, 4)

    # Checkpoint.tensor_parallel composes with Checkpoint.ranks by restricting within checkpoint.ranks.
    subset_costs = MappingProxyType({2: RankCosts(1 << 20, 1 << 20, 1 << 28), 4: RankCosts(1 << 20, 1 << 20, 1 << 28)})
    with_ranks = replace(
        recipe,
        checkpoints=(
            plain,
            replace(overlaid, ranks=subset_costs, plain_ranks=subset_costs, tensor_parallel=()),
        ),
    )
    assert with_ranks.checkpoint_tensor_parallel(with_ranks.checkpoints[1]) == (2, 4)
    restricted_within_ranks = replace(
        recipe,
        checkpoints=(
            plain,
            replace(overlaid, ranks=subset_costs, plain_ranks=subset_costs, tensor_parallel=(4,)),
        ),
    )
    assert restricted_within_ranks.checkpoint_tensor_parallel(restricted_within_ranks.checkpoints[1]) == (4,)
    with pytest.raises(ValueError, match="subset of the checkpoint's or recipe's declared sizes"):
        replace(
            recipe,
            checkpoints=(
                plain,
                replace(overlaid, ranks=subset_costs, plain_ranks=subset_costs, tensor_parallel=(1,)),
            ),
        )


@pytest.mark.parametrize("sizes", [(8,), (0,), (-4,), (4, 4), (4, 2), [4], (4.0,), (True,)])
def test_checkpoint_tensor_parallel_rejects_invalid_sizes(sizes):
    recipe = _sample_recipe()
    plain, overlaid = recipe.checkpoints
    invalid = replace(overlaid, tensor_parallel=sizes)
    with pytest.raises(ValueError, match="subset of the checkpoint's or recipe's declared sizes"):
        replace(recipe, checkpoints=(plain, invalid))


def test_planning_and_help_respect_checkpoint_tensor_parallel(capsys):
    recipe = _sample_recipe()
    plain, overlaid = recipe.checkpoints
    options = dict(concurrency=1, reserve_bytes=0, speculation=None)

    assert planning.plan(recipe, overlaid, _gpus(4), selected=None, **options).world_size == 4
    for selected in ((0,), (0, 1)):
        with pytest.raises(CapacityError, match="needs 4 GPUs"):
            planning.plan(recipe, overlaid, _gpus(4), selected=selected, **options)
    with pytest.raises(CapacityError, match="needs 4"):
        planning.plan(recipe, overlaid, _gpus(2), selected=None, **options)
    assert planning.plan(recipe, plain, _gpus(4), selected=None, **options).world_size == 1

    with pytest.raises(SystemExit) as exc:
        main(["export", "synth-model-overlay", "--help"], recipes=(recipe,))
    assert exc.value.code == 0
    assert "--gpus: select 4 GPUs" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["--help"], recipes=(recipe,))
    assert exc.value.code == 0
    listing = {
        line.split()[0]: line.split()[1]
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("synth-model-")
    }
    assert listing == {"synth-model-plain": "1,2,4", "synth-model-overlay": "4"}


@pytest.mark.parametrize("kwargs", [
    dict(manifest="", schema="v1"),
    dict(manifest="/abs/manifest.json", schema="v1"),
    dict(manifest="../manifest.json", schema="v1"),
    dict(manifest="sub/./manifest.json", schema="v1"),
    dict(manifest="sub\\manifest.json", schema="v1"),
    dict(manifest="manifest.json", schema=""),
    dict(manifest="manifest.json", schema="v1", assets_subdir="/carrier"),
    dict(manifest="manifest.json", schema="v1", assets_subdir="carrier/"),
    dict(manifest="manifest.json", schema="v1", assets_subdir="../carrier"),
    dict(manifest="manifest.json", schema="v1", required_files=["meta/transform.json"]),
    dict(manifest="manifest.json", schema="v1", required_files=("../secret.json",)),
    dict(manifest="manifest.json", schema="v1", validate_manifest="not-callable"),
])
def test_checkpoint_overlay_validation_rejects_malformed_specs(kwargs):
    with pytest.raises(ValueError):
        CheckpointOverlay(**kwargs)


def test_checkpoint_rejects_non_overlay_object():
    with pytest.raises(ValueError, match="CheckpointOverlay"):
        Checkpoint(
            alias="bad",
            repository="example/bad",
            revision="a" * 40,
            architecture="SynthModelForCausalLM",
            stored_bytes=1,
            device_bytes=1,
            overlay="example.overlay.v1",
        )


def test_overlay_assets_root_detection_and_pass_through(tmp_path):
    root, _, _ = _build_overlay_dir(tmp_path / "release")
    assert checkpoint_assets_root(root, SAMPLE_OVERLAY) == root / "carrier"
    assert checkpoint_assets_root(root, None) == root

    single_root, _, _ = _build_overlay_dir(tmp_path / "single-release", single_file_carrier=True)
    assert checkpoint_assets_root(single_root, SAMPLE_OVERLAY) == single_root / "carrier"

    plain = tmp_path / "plain"
    plain.mkdir()
    assert checkpoint_assets_root(plain, SAMPLE_OVERLAY) == plain
    assert read_overlay_manifest(plain, SAMPLE_OVERLAY) is None

    other = tmp_path / "other-schema"
    other.mkdir()
    (other / SAMPLE_OVERLAY.manifest).write_text(json.dumps({"schema": "other.overlay.v1"}))
    assert checkpoint_assets_root(other, SAMPLE_OVERLAY) == other
    assert read_overlay_manifest(other, SAMPLE_OVERLAY) is None

    no_carrier_index = tmp_path / "no-carrier-index"
    no_carrier_index.mkdir()
    (no_carrier_index / SAMPLE_OVERLAY.manifest).write_text(json.dumps({"schema": SAMPLE_OVERLAY.schema}))
    with pytest.raises(CheckpointError, match="lacks safetensors assets"):
        checkpoint_assets_root(no_carrier_index, SAMPLE_OVERLAY)

    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / SAMPLE_OVERLAY.manifest).write_text("{not json")
    with pytest.raises(CheckpointError):
        checkpoint_assets_root(corrupt, SAMPLE_OVERLAY)

    non_object = tmp_path / "non-object"
    non_object.mkdir()
    (non_object / SAMPLE_OVERLAY.manifest).write_text("[]")
    with pytest.raises(CheckpointError, match="JSON object"):
        read_overlay_manifest(non_object, SAMPLE_OVERLAY)


def test_overlay_directory_resolution_caching_and_verification(tmp_path):
    recipe = _sample_recipe(overlay_first=True)
    overlaid, plain = recipe.checkpoints

    # Alias and repository resolution preserve declared checkpoint identity.
    for name in (
        "synth-model-overlay",
        "example/Synth-Model-Overlay",
        "example/Synth-Model-Overlay@" + overlaid.revision,
    ):
        resolved = resolve(name, recipes=(recipe,))
        assert resolved.checkpoint is overlaid
        assert resolved.checkpoint.overlay is SAMPLE_OVERLAY

    # Local overlay directory matches the overlay checkpoint and computes carrier + sidecar bytes.
    release, carrier_bytes, sidecar_bytes = _build_overlay_dir(tmp_path / "local-release")
    model = resolve(str(release), recipes=(recipe,))
    assert model.directory == release.resolve()
    assert model.checkpoint.alias == "synth-model-overlay"
    assert model.checkpoint.stored_bytes == carrier_bytes + sidecar_bytes
    assert model.checkpoint.draft_bytes == overlaid.draft_bytes
    assert model.checkpoint.overlay is SAMPLE_OVERLAY
    assert model.recipe.checkpoint_tensor_parallel(model.checkpoint) == (4,)
    assert model.cached(tmp_path / "hub") is True
    model.verify(tmp_path / "hub")

    # Local overlay directory with config.json only inside assets_subdir also resolves.
    carrier_only_cfg, _, _ = _build_overlay_dir(tmp_path / "carrier-cfg-only", root_config=False, carrier_config=True)
    model_carrier_cfg = resolve(str(carrier_only_cfg), recipes=(recipe,))
    assert model_carrier_cfg.checkpoint.alias == "synth-model-overlay"

    # Plain directory matches the plain checkpoint even when the overlay checkpoint is listed first.
    plain_dir = tmp_path / "plain-dir"
    plain_dir.mkdir()
    (plain_dir / "config.json").write_text(json.dumps({"architectures": ["SynthModelForCausalLM"]}))
    plain_model = resolve(str(plain_dir), recipes=(recipe,))
    assert plain_model.checkpoint.alias == "synth-model-plain"
    assert plain_model.checkpoint.overlay is None
    assert plain_model.checkpoint.stored_bytes == plain.stored_bytes

    # Plain directory carrying an unrelated overlay schema is not mistaken for the declared overlay.
    (plain_dir / SAMPLE_OVERLAY.manifest).write_text(json.dumps({"schema": "unrelated.overlay.v9"}))
    assert resolve(str(plain_dir), recipes=(recipe,)).checkpoint.alias == "synth-model-plain"

    # Hub-cache snapshot lookup works through symlinked overlay release.
    hub_cache = tmp_path / "hub-cache"
    by_alias = resolve("synth-model-overlay", hub_cache, recipes=(recipe,))
    assert by_alias.cached(hub_cache) is False
    # Pre-download verify succeeds when the snapshot path does not exist yet.
    by_alias.verify(hub_cache)
    snapshot = by_alias.snapshot(hub_cache)
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(release)
    assert by_alias.cached(hub_cache) is True
    by_alias.verify(hub_cache)


def test_overlay_rejects_unreadable_incomplete_or_invalid_manifests(tmp_path):
    recipe = _sample_recipe()

    # Unreadable manifest never falls back to a plain checkpoint.
    corrupt, _, _ = _build_overlay_dir(tmp_path / "corrupt-release")
    (corrupt / SAMPLE_OVERLAY.manifest).write_text("{broken")
    with pytest.raises(UsageError, match="unreadable overlay manifest"):
        resolve(str(corrupt), recipes=(recipe,))
    assert _snapshot_cached(corrupt, SAMPLE_OVERLAY) is False

    # Missing required file, missing sidecar, truncated sidecar, or missing carrier shard makes cached False and verify fail.
    for damage in ("missing-required", "missing-sidecar", "short-sidecar", "missing-carrier-shard"):
        root, _, _ = _build_overlay_dir(tmp_path / damage)
        model = resolve(str(root), recipes=(recipe,))
        if damage == "missing-required":
            (root / "meta" / "transform.json").unlink()
        elif damage == "missing-sidecar":
            (root / "parts" / "part-1.bin").unlink()
        elif damage == "short-sidecar":
            (root / "parts" / "part-1.bin").write_bytes(b"short")
        else:
            (root / "carrier" / "shard-00001.safetensors").unlink()
        assert model.cached(tmp_path) is False
        expected_error = (
            "lacks carrier tensor files"
            if damage == "missing-carrier-shard"
            else "lacks files declared by its overlay manifest"
        )
        with pytest.raises(UsageError, match=expected_error):
            model.verify(tmp_path)

    # Generic manifest file-list checks reject empty lists, duplicates, respelled/escaping paths, and non-positive bytes.
    plain_overlay = CheckpointOverlay(manifest="overlay-manifest.json", schema="example.overlay.v1", assets_subdir="carrier")
    for bad_files in (
        [],
        [{"path": "parts/part-0.bin", "bytes": 4}, {"path": "parts/part-0.bin", "bytes": 4}],
        [{"path": "parts/./part-0.bin", "bytes": 4}],
        [{"path": "../escape.bin", "bytes": 4}],
        [{"path": "parts/part-0.bin", "bytes": 0}],
        [{"path": "parts/part-0.bin", "bytes": True}],
    ):
        with pytest.raises(CheckpointError):
            overlay_manifest_files({"schema": plain_overlay.schema, "files": bad_files}, plain_overlay)

    # Custom validate_manifest hook rejects mismatched inventory across resolve, cached, and verify.
    bad_inventory, _, _ = _build_overlay_dir(
        tmp_path / "bad-inventory",
        manifest_doc={
            "schema": SAMPLE_OVERLAY.schema,
            "expected_paths": ["parts/part-0.bin", "parts/part-1.bin"],
            "files": [{"path": "parts/part-0.bin", "bytes": 4}],
        },
    )
    with pytest.raises(UsageError, match="expected_paths"):
        resolve(str(bad_inventory), recipes=(recipe,))
    assert _snapshot_cached(bad_inventory, SAMPLE_OVERLAY) is False

    # Existing snapshot missing its manifest or carrying a wrong schema fails verify for an overlay alias.
    hub_cache = tmp_path / "bad-hub"
    by_alias = resolve("synth-model-overlay", hub_cache, recipes=(recipe,))
    snap = by_alias.snapshot(hub_cache)
    snap.mkdir(parents=True)
    with pytest.raises(UsageError, match="missing overlay manifest"):
        by_alias.verify(hub_cache)
    (snap / SAMPLE_OVERLAY.manifest).write_text(json.dumps({"schema": "wrong.schema.v1"}))
    with pytest.raises(UsageError, match="does not declare schema"):
        by_alias.verify(hub_cache)

    # Overlay directory whose architecture has no overlay checkpoint fails with the overlay schema named.
    unmatched, _, _ = _build_overlay_dir(tmp_path / "unmatched")
    (unmatched / "config.json").write_text(json.dumps({"architectures": ["UnmatchedArch"]}))
    with pytest.raises(UsageError, match=SAMPLE_OVERLAY.schema):
        resolve(str(unmatched), recipes=(recipe,))

    # Without any declared overlay in active recipes, an overlay directory falls back to plain resolution.
    plain_only_recipe = replace(recipe, checkpoints=(recipe.checkpoints[0],))
    valid_overlay, _, _ = _build_overlay_dir(tmp_path / "valid-for-plain-only")
    plain_fallback = resolve(str(valid_overlay), recipes=(plain_only_recipe,))
    assert plain_fallback.checkpoint.alias == "synth-model-plain"
    assert plain_fallback.checkpoint.overlay is None

    # Multiple active recipes matching the same directory with conflicting assets_subdir raise UsageError.
    alt_overlay = CheckpointOverlay(
        manifest=SAMPLE_OVERLAY.manifest,
        schema=SAMPLE_OVERLAY.schema,
        assets_subdir="alt_carrier",
    )
    alt_recipe = replace(_sample_recipe(overlay=alt_overlay), name="alt_synth_model")
    with pytest.raises(UsageError, match="conflicting assets_subdir"):
        resolve(str(valid_overlay), recipes=(recipe, alt_recipe))


def test_overlay_helpers_support_manifest_without_files_key(tmp_path):
    minimal = CheckpointOverlay(
        manifest="overlay.json",
        schema="example.minimal.v1",
        assets_subdir="carrier",
        required_files=("meta/info.json",),
    )
    root = tmp_path / "minimal"
    carrier = root / "carrier"
    carrier.mkdir(parents=True)
    payload_bytes = _write_safetensors(carrier / "weights.safetensors", b"\x03" * 32)
    (carrier / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": payload_bytes},
        "weight_map": {"weight": "weights.safetensors"},
    }))
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text("{}")
    (root / "overlay.json").write_text(json.dumps({"schema": minimal.schema}))

    assert overlay_files_present(root, minimal) is True
    assert overlay_stored_bytes(root, minimal) == payload_bytes


def test_publisher_reads_model_license_through_overlay_assets_root(tmp_path):
    recipe = _sample_recipe()
    card = "---\nlicense: apache-2.0\n---\n# Synthetic Carrier\n"
    release, _, _ = _build_overlay_dir(tmp_path / "licensed-release", carrier_readme=card)
    (release / "carrier" / "LICENSE").write_text("Apache-2.0 text")

    config = Configuration(tmp_path / "cfg" / "config.toml", {}, False)
    args = SimpleNamespace(official_source=None)

    # Local directory overlay resolves license from carrier/README.md.
    local_model = resolve(str(release), recipes=(recipe,))
    signer = _publisher(config, args, local_model, tmp_path / "hub")
    assert signer.license == License("apache-2.0", "")

    # Repository-resolved overlay links the license file under the checkpoint repository and revision.
    hub_cache = tmp_path / "hub"
    repo_model = resolve("synth-model-overlay", hub_cache, recipes=(recipe,))
    snapshot = repo_model.snapshot(hub_cache)
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(release)
    repo_signer = _publisher(config, args, repo_model, hub_cache)
    assert repo_signer.license == License(
        "apache-2.0",
        f"https://huggingface.co/{repo_model.checkpoint.repository}/blob/{repo_model.checkpoint.revision}/LICENSE",
    )

    # Plain checkpoint continues to read README.md from the snapshot root.
    plain_dir = tmp_path / "plain-licensed"
    plain_dir.mkdir()
    (plain_dir / "config.json").write_text(json.dumps({"architectures": ["SynthModelForCausalLM"]}))
    payload_bytes = _write_safetensors(plain_dir / "weights.safetensors", b"\x04" * 16)
    (plain_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": payload_bytes},
        "weight_map": {"weight": "weights.safetensors"},
    }))
    (plain_dir / "README.md").write_text("---\nlicense: mit\n---\n# Plain\n")
    plain_model = resolve(str(plain_dir), recipes=(recipe,))
    assert _publisher(config, args, plain_model, hub_cache).license == License("mit", "")

    # Missing carrier assets raises UsageError instead of silently reading a root README.md license.
    missing_carrier, _, _ = _build_overlay_dir(tmp_path / "missing-carrier-pub", carrier_readme=card)
    missing_carrier_model = resolve(str(missing_carrier), recipes=(recipe,))
    (missing_carrier / "README.md").write_text("---\nlicense: mit\n---\n# Wrong Root License\n")
    (missing_carrier / "carrier" / "model.safetensors.index.json").unlink()
    with pytest.raises(UsageError, match="lacks safetensors assets"):
        _publisher(config, args, missing_carrier_model, hub_cache)

    # Unreadable overlay manifest raises UsageError in _publisher.
    (release / SAMPLE_OVERLAY.manifest).write_text("{corrupt")
    with pytest.raises(UsageError):
        _publisher(config, args, local_model, hub_cache)


def test_core_declaration_and_overlay_imports_stay_model_dependency_free():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ignition.declaration, ignition.registry, ignition.planning, "
            "ignition.cli.help, ignition.preparation.checkpoint; "
            "assert not {'torch', 'transformers', 'flare_heliopause'} & set(sys.modules)",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
