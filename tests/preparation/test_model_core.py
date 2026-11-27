# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Atomic publication and rank ownership in native model cores."""

from contextlib import nullcontext
import multiprocessing
import sys
import time
from types import SimpleNamespace

import pytest

from helios.tests.signing import publisher
from ignition.preparation import model_core
from ignition.preparation.core_file import combine

from ignition.preparation.core_snapshot import export_options
from tests.preparation.payload_fixture import rank_source, manifest


@pytest.fixture(autouse=True)
def configured_cache(tmp_path):
    with export_options(tmp_path / '.flare-cache'):
        yield


@pytest.mark.parametrize('world', [1, 2])
def test_publication_contains_one_bundle_per_rank(monkeypatch, tmp_path, world):
    import torch
    import torch.distributed as dist

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(dist, 'barrier', lambda: None)
    exported = []
    closed = []
    printed = []
    def export(prepared, images, storage, path, **kwargs):
        source = rank_source(path.parent / path.stem)
        path.write_bytes(source.read_bytes())
        exported.append(path.name)
        return SimpleNamespace(memory_inventory={'rank': 0})
    monkeypatch.setattr(model_core, 'export_model', export)
    class Control:
        def send(self, command):
            assert command[0] == 'export'
            export(None, None, None, command[1])
        def recv(self):
            return {'peer_memory': [(0x210000000000, 65536)], 'memory_inventory': {'rank': 1}}
    def close(*, after_export=False):
        assert after_export == (world > 1)
        closed.append('model')
    prepared = SimpleNamespace(model=object(), controls=[Control()] if world == 2 else [], close=close)
    storage = SimpleNamespace(world=world, peer_storage=SimpleNamespace(
        owned={0x200000000000: 65536}, reset=lambda: None),
        host_regions=[SimpleNamespace(close=lambda: closed.append('host'))])
    def assemble(*args, **kwargs):
        assert closed == ['model', 'host']
        assert len(exported) == world
        return combine(*args, **kwargs)
    monkeypatch.setattr(model_core, 'combine', assemble)
    output = tmp_path / 'model'
    def inventory(reports, **kwargs):
        assert (output / "model.safetensors").is_file() and closed == ['model', 'host']
        printed.extend(reports)
    monkeypatch.setattr(model_core, 'print_inventory', inventory)
    model_core.export_core(prepared, None, storage, output, options={}, publisher=publisher())
    envelope = manifest(output)
    assert len(envelope.ranks) == world
    assert sorted(exported) == [f'rank-{rank}.json' for rank in range(world)]
    assert (output / "model.safetensors").is_file()
    assert printed == [{'rank': rank} for rank in range(world)]
    assert {p.name for p in tmp_path.iterdir()} == {output.name, '.flare-cache'}


def test_failed_rank_export_leaves_no_published_model(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError('missing module image')
    monkeypatch.setattr(model_core, 'export_model', fail)
    with pytest.raises(RuntimeError, match='missing module image'):
        model_core.export_core(object(), None, SimpleNamespace(world=1), tmp_path / 'model', options={}, publisher=publisher())
    assert not list(tmp_path.iterdir())


def _write_host_rank(model, images, storage, path, **kwargs):
    source = rank_source(path.parent / path.stem)
    path.write_bytes(source.read_bytes())
    return SimpleNamespace(memory_inventory={})


def _host_export_peer(*args):
    import torch
    from ignition.preparation import core_storage, module_images, rank_group, runtime_core

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, 'set_device', lambda _: None)
        patch.setattr(module_images, 'ModuleImages', nullcontext)
        patch.setattr(core_storage, 'CoreStorage', lambda **kwargs: nullcontext())
        patch.setattr(runtime_core, 'export_model', _write_host_rank)
        rank_group._peer(*args)


def _prepare_host_rank(directory, config, *, rank, core_storage, **kwargs):
    def release_region():
        if rank == 1 and config.fail:
            raise RuntimeError('peer host region failed to close')
        (directory / f'released-{rank}').touch()

    def release_model():
        if rank == config.slow_rank:
            time.sleep(config.delay)

    core_storage.world = len(config.devices)
    core_storage.peer_storage = SimpleNamespace(owned={}, reset=lambda: None)
    core_storage.host_regions = [SimpleNamespace(close=release_region)]
    return SimpleNamespace(manifest=lambda: {}, memory_report={}, close=release_model)


def _export_with_slow_release(directory, slow_rank, delay, fail, deadline_ms=None):
    from ignition.preparation import module_images, rank_group

    config = SimpleNamespace(devices=(0, 1), slow_rank=slow_rank, delay=delay, fail=fail)
    root = SimpleNamespace()
    with pytest.MonkeyPatch.context() as patch, export_options(directory / 'cache'):
        patch.setattr(rank_group, '_peer', _host_export_peer)
        patch.setattr(module_images.ModuleImages, 'current', staticmethod(lambda: True))
        patch.setattr(model_core, 'export_model', _write_host_rank)
        patch.setattr(model_core, 'print_inventory', lambda *args, **kwargs: None)
        with rank_group.prepare_group(directory, config, _prepare_host_rank, core_storage=root) as group:
            if deadline_ms is not None:
                supervisor = group.supervisor
                group.supervisor = SimpleNamespace(
                    begin_shutdown=lambda _: supervisor.begin_shutdown(deadline_ms),
                    finish_shutdown=supervisor.finish_shutdown, abort=supervisor.abort)
            model_core.export_core(group, None, root, directory / 'model', options={}, publisher=publisher())
    assert not multiprocessing.active_children()


@pytest.mark.skipif(sys.platform != 'linux', reason='local rank supervision requires Linux pidfds')
@pytest.mark.parametrize('slow_rank,fail', [(0, False), (1, False), (1, True)])
def test_slow_rank_release_preserves_completed_export(tmp_path, capfd, slow_rank, fail):
    # The native supervisor terminates the coordinator on an unconfirmed drain.
    process = multiprocessing.get_context('spawn').Process(
        target=_export_with_slow_release, args=(tmp_path, slow_rank, 2, fail, 1000))
    process.start()
    try:
        process.join(timeout=60)
        assert process.exitcode == (1 if fail else 0)
    finally:
        if process.is_alive():
            process.kill()
        process.join()
    if fail:
        assert 'peer host region failed to close' in capfd.readouterr().err
        assert not (tmp_path / 'model').exists()
    else:
        assert all((tmp_path / f'released-{rank}').exists() for rank in range(2))
        assert len(manifest(tmp_path / 'model').ranks) == 2
