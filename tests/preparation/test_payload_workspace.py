# SPDX-License-Identifier: Apache-2.0
"""Check native payload assembly, standalone commit placement and export stage lifetime."""

from contextlib import nullcontext
from pathlib import Path

import pytest

from helios.tests.signing import publisher
from ignition.preparation.core_file import combine
from ignition.preparation.core_snapshot import export_options
from tests.preparation.payload_fixture import rank_source, schema, verify_signatures


def contents(path, public_key):
    from safetensors import safe_open

    primary = path if path.is_file() else path / "model.safetensors"
    files = [primary] if path.is_file() else [entry for entry in path.rglob("*") if entry.is_file()]
    data = {entry.relative_to(path): entry.read_bytes() for entry in files}
    envelope = verify_signatures(path, public_key)
    with safe_open(primary, framework="numpy") as container:
        original = container.get_tensor("flare.manifest").tobytes()
    # Each publication assigns a random core ID and signs each body under that ID.
    core = schema()("CoreBody").FromString(envelope.core_body)
    core.core_id = envelope.core_id = bytes(32)
    envelope.core_body = core.SerializeToString(deterministic=True)
    envelope.core_signature = bytes(64)
    for body in (*envelope.deployments, *envelope.ranks):
        body.signature = bytes(64)
    key = primary.relative_to(path)
    assert data[key].count(original) == 1
    data[key] = data[key].replace(original, envelope.SerializeToString(deterministic=True))
    return data


@pytest.mark.parametrize("form,cached", [
    ("cache", True), ("directory", True), ("packed", True), ("directory", False), ("packed", False),
])
def test_rank_workspace_preserves_signed_publication(tmp_path, form, cached):
    from flare_heliopause import NativeExportStage

    source = rank_source(tmp_path / "source")
    signer = publisher()
    stage = NativeExportStage(None, tmp_path / "assembly", None)
    workspace = Path(stage.path)
    outputs = []
    try:
        for name, options in (("legacy", {}), ("staged", {"stage": stage})):
            destination = None if form == "cache" else tmp_path / name
            context = export_options(tmp_path / (name + "-cache")) if cached else nullcontext()
            with context:
                result = combine(destination, [source], [], publisher=signer, cache_only=destination is None,
                                 form="directory" if destination is None else form, **options)
            output = destination if destination is not None else Path(result.entry_path)
            outputs.append(contents(output, signer.key.public_key()))
        assert outputs[0] == outputs[1]
        assert list(workspace.iterdir()) == []
    finally:
        stage.close()
    assert not workspace.exists()


def test_cache_only_export_requires_its_supplied_workspace(tmp_path):
    from flare_heliopause import NativeExportStage

    source = rank_source(tmp_path / "source")
    cache = tmp_path / "cache"
    stage = NativeExportStage(None, tmp_path / "assembly", cache)
    workspace = Path(stage.path)
    moved = tmp_path / "moved"
    workspace.rename(moved)
    try:
        with export_options(cache), pytest.raises(ValueError):
            combine(None, [source], [], cache_only=True, publisher=publisher(), stage=stage)
        assert not workspace.exists()
        assert not (cache / "models").exists()
    finally:
        moved.rename(workspace)
        stage.close()
    with export_options(cache), pytest.raises(ValueError, match="export stage is closed"):
        combine(None, [source], [], cache_only=True, publisher=publisher(), stage=stage)


def test_failed_cache_commit_releases_only_its_assembly(tmp_path):
    from flare_heliopause import NativeExportStage

    source = rank_source(tmp_path / "source")
    cache = tmp_path / "cache"
    (cache / "models").mkdir(parents=True)
    occupied = cache / "models" / "model"
    occupied.write_bytes(b"occupied")
    stage = NativeExportStage(None, tmp_path / "scratch", cache)
    retained = Path(stage.path) / "retained-rank-source"
    retained.write_bytes(b"rank")
    try:
        with export_options(cache), pytest.raises(ValueError):
            combine(None, [source], [], cache_only=True, publisher=publisher(), stage=stage)
        assert list(Path(stage.path).iterdir()) == [retained]
        assert occupied.read_bytes() == b"occupied"
    finally:
        stage.close()
