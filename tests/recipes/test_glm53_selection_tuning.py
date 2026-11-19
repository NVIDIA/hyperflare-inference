"""A prefill context gather prepares ring windows only with a records runtime and binds one transport.

The race itself, peer memory, capture and replay run on four GPUs in
tests/recipes/test_glm53_context_transport_tp_cuda.py, which also checks that the winner is bound to the
gather it registered.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def _context(selection_tuning=None, *, records=object()):
    from ignition.recipes.glm53.dcp import ContextParallel, ContextShards

    coordinator = SimpleNamespace(ranks=(0, 1, 2, 3), group=object(), barrier=lambda *args: None)
    return ContextParallel(ContextShards.on(0, 4, "cpu"), heads=None, candidates=None,
        coordinator=coordinator, records=records, selection_tuning=selection_tuning)


def _scratch(positions=4096, window=1024):
    return SimpleNamespace(slab=torch.zeros(positions, 132, dtype=torch.int32),
        records=torch.zeros(4, window * 528, dtype=torch.uint8), record_counts=torch.zeros(4, dtype=torch.int64))


def test_trial_context_must_not_register_the_models_gathers():
    from ignition.recipes.glm53 import selection_tuning as tuning

    with pytest.raises(ValueError):
        tuning.ContextGatherTuning(_context(selection_tuning=object()))


def test_gathers_of_one_preparation_share_one_slab_shape(monkeypatch):
    from ignition.recipes.glm53 import dcp, selection_tuning as tuning

    monkeypatch.setattr(dcp, "PreparedVarlenGather", lambda *args, **kwargs: object())
    registry = tuning.ContextGatherTuning(_context())
    context = replace(_context(), selection_tuning=registry)
    dcp.PreparedContextGather(None, _scratch(), context, session=None, name="a")
    with pytest.raises(ValueError, match="share one slab shape"):
        dcp.PreparedContextGather(None, _scratch(8192, 1024), context, session=None, name="b")


def _prefill_gather(monkeypatch, *, remote, records, window=True):
    from ignition.recipes.glm53 import dcp
    from ignition.recipes.glm53.dcp import ContextParallel, ContextShards
    from ignition.recipes.glm53.mla import MLAScratch

    monkeypatch.setattr(dcp, "PreparedVarlenGather", lambda *args, **kwargs: object())
    scratch = MLAScratch.allocate(1, 16, 1, "prefill", "cpu", context_ranks=4,
                                  remote=remote, gather_context=records and window)
    context = ContextParallel(ContextShards.on(0, 4, "cpu"), heads=None, candidates=None,
        coordinator=None, remote=remote, records=object() if records else None)
    return scratch, dcp.PreparedContextGather(None, scratch, context, session=None, name="fixture")


@pytest.mark.parametrize("remote,records,expected", [(False, False, "pull"), (False, True, "pull"), (True, True, "ring")])
def test_context_gather_prepares_windows_only_with_a_records_runtime(monkeypatch, remote, records, expected):
    scratch, gather = _prefill_gather(monkeypatch, remote=remote, records=records)
    assert gather.implementation == expected
    assert bool(gather.windows) == bool(scratch.records.numel()) == records
    assert gather.uses_records == (expected == "ring")


def test_context_gather_rejects_a_records_runtime_without_scratch_records(monkeypatch):
    with pytest.raises(ValueError, match="require scratch records"):
        _prefill_gather(monkeypatch, remote=False, records=True, window=False)


def test_context_gather_binds_only_prepared_implementations(monkeypatch):
    _, gather = _prefill_gather(monkeypatch, remote=False, records=True)
    gather.bind("ring")
    assert gather.implementation == "ring"
    with pytest.raises(ValueError):
        gather.bind("tree")
    _, without = _prefill_gather(monkeypatch, remote=False, records=False)
    with pytest.raises(ValueError):
        without.bind("ring")


@pytest.mark.parametrize("mode,remote,records,bound,allocated", [
    ("prefill", False, True, "ring", True),
    # Peer pull never packs records, so a host where it wins keeps the memory.
    ("prefill", False, True, "pull", False),
    ("prefill", True, True, None, True),
    ("prefill", False, False, None, False),
    ("decode", False, False, None, False)])
def test_forward_allocates_records_only_for_the_selected_ring(monkeypatch, mode, remote, records, bound, allocated):
    from ignition.recipes.glm53.mla import MLAScratch, PreparedGlmMLA

    prepared = object.__new__(PreparedGlmMLA)
    prepared.heads, prepared.blocks, prepared.mode, prepared.device = 16, 1, mode, "cpu"
    prepared.context_ranks, prepared.remote = 4, remote
    prepared.context_gather = None
    if mode == "prefill":
        scratch, prepared.context_gather = _prefill_gather(monkeypatch, remote=remote, records=records)
        if bound:
            prepared.context_gather.bind(bound)
    class Allocated(Exception):
        pass
    allocate = MLAScratch.allocate
    def capture(cls, *args, **kwargs):
        raise Allocated(allocate(*args, **kwargs))
    monkeypatch.setattr(MLAScratch, "allocate", classmethod(capture))
    with pytest.raises(Allocated) as raised:
        prepared.forward(torch.zeros(1, 4096, dtype=torch.bfloat16), None)
    s = raised.value.args[0]
    assert bool(s.records.numel()) == bool(s.record_counts.numel()) == allocated
    if allocated:
        # The windows were prepared for this row width, and the pack view the ring path takes must accept it.
        assert s.records.shape == scratch.records.shape and s.record_counts.shape == scratch.record_counts.shape
        assert s.records.view(torch.uint8).view(torch.int32).shape[1] // 132 == prepared.context_gather.window


def test_context_across_hosts_requires_its_records_runtime(monkeypatch):
    from ignition.recipes.glm53 import dcp
    from ignition.recipes.glm53.dcp import ContextParallel, ContextShards

    context = ContextParallel(ContextShards.on(0, 4, "cpu"), heads=None, candidates=None,
                              coordinator=None, remote=True, records=None)
    with pytest.raises(ValueError, match="records runtime"):
        dcp.PreparedContextGather(None, _scratch(), context, session=None, name="fixture")


@pytest.mark.parametrize("agreed", [True, False])
def test_every_rank_must_select_the_same_transport(agreed):
    from ignition.recipes.glm53 import selection_tuning as tuning

    context = _context()
    context.coordinator.same_on_every_rank = lambda value: agreed
    registry = tuning.ContextGatherTuning(context)
    if agreed:
        registry._require_agreement("ring")
    else:
        with pytest.raises(RuntimeError, match="different prefill context transports"):
            registry._require_agreement("ring")
