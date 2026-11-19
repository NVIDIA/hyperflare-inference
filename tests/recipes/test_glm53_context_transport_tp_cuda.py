"""Peer pull and the counted ring each reproduce the latent bytes on four GPUs, and the race binds its winner.

The model-path test runs PreparedGlmMLA prefill with each transport on identical inputs.
"""

from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                              reason="requires four explicitly assigned CUDA devices")


def _worker(rank, rendezvous):
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.collectives import CollectivePreparation, create_varlen_gather_runtime
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.dcp import ContextParallel, ContextShards
    # The trial fixture is private to the tuning module; this test drives it directly to replay
    # each transport against the race's oracle.
    from ignition.recipes.glm53.selection_tuning import ContextGatherTuning, _Fixture

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(minutes=5))
    runtime, fixture, graphs = None, None, {}
    try:
        coordinator = CollectivePreparation(dist.group.WORLD)
        runtime = create_varlen_gather_runtime(dist.group.WORLD, device)
        context = ContextParallel(ContextShards.on(rank, 4, device), heads=None, candidates=None,
                                  records=runtime, coordinator=coordinator)
        with RecordedPreparationSession(device=device, checkpoint="context-transport-fixture",
                workload_schema="tp4-dcp4-context", numerical_policy="exact-cache-bytes",
                coordinator=coordinator, compile_workers=4) as session:
            fixture = _Fixture(8192, 512, context, session)
            session.tune(show_progress=False)
            registry = ContextGatherTuning(context)
            registry.register(fixture.operation)
            report = registry.tune(session)
            # The winner is bound to the gather that registered, and every rank bound the same one.
            assert fixture.operation.implementation == report["implementation"]
            bound = [None] * 4
            dist.all_gather_object(bound, fixture.operation.implementation)
            assert len(set(bound)) == 1
            if rank == 0:
                print("context transport race: " + json.dumps(report), flush=True)
            fixture.produce(0)
            signatures = {kind: GraphSignature("prefill", 1, 8192) for kind in fixture.forwards}
            graphs, _ = capture_forwards(fixture.forwards, signatures, session=session, device=device,
                stream=torch.cuda.Stream(device=device), pool=torch.cuda.graph_pool_handle(),
                coordinator=coordinator, report_catalog=False)
            # Both transports must reproduce the exact bytes, whichever one the race selected.
            for kind in fixture.forwards:
                for length in (0, 1, 37, 2049, 8191, 8192):
                    fixture.lengths = (length,)
                    fixture.produce(0)
                    torch.cuda.synchronize(device)
                    coordinator.barrier(f"{kind} context replay", coordinator.ranks)
                    fixture.reset()
                    graphs[kind].replay()
                    torch.cuda.synchronize(device)
                    fixture.validate()
    finally:
        for graph in graphs.values():
            graph.reset()
        if fixture is not None:
            fixture.shared.close()
        if runtime is not None:
            runtime.close()
        dist.destroy_process_group()


def test_prepared_context_transport_tp4(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_worker, args=((tmp_path / "gloo").as_uri(),), nprocs=4, join=True)


# The model-path test below attends over a context of four ranks' interleaved records. The local cache holds
# more records than one gather window, and the context ends inside the second window.
PAGES = 260
WINDOW = 65536
ROWS = 128
START = 4 * WINDOW + 200
SELECTED = 2051


def _selection(start, rows, seed):
    """Selected global positions of every row, as the prefill indexer produces them: 512 pools and the tail.

    Choosing the pools here keeps attention independent of the indexer's tie-breaking, and covers the
    records of every owner rank across every window.
    """
    generator = torch.Generator().manual_seed(seed)
    physical = torch.full((rows, SELECTED), -1, dtype=torch.int32)
    lengths = torch.zeros(rows, dtype=torch.int32)
    for row in range(rows):
        length = start + row + 1
        complete = length // 4
        # The last pools lie in the final window of every rank.
        pools = torch.cat([torch.arange(complete - 16, complete),
                           torch.randperm(complete - 16, generator=generator)[:496]])
        chosen = torch.cat([(pools[:, None] * 4 + torch.arange(4)).reshape(-1),
                            torch.arange(length // 4 * 4, length)]).int()
        physical[row, :len(chosen)] = chosen
        lengths[row] = len(chosen)
    return physical, lengths


class _FixedSelection:
    """Attend over a fixed selection in place of the indexer's, which reaches every owner rank and window.

    A forward is serialized to compile, so its parts are module-level classes rather than closures.
    """

    def __init__(self, attention, physical, lengths):
        self.attention, self.physical, self.lengths = attention, physical, lengths

    def __call__(self, s, m, query, selected, selected_lengths):
        from ignition.recipes.glm53.mla import PreparedGlmMLA

        selected.copy_(self.physical)
        selected_lengths.copy_(self.lengths)
        return PreparedGlmMLA.attend_gathered_context(self.attention, s, m, query, selected, selected_lengths)


class _ModelForward:
    """One prefill forward of PreparedGlmMLA whose results are copied into persistent storage."""

    def __init__(self, attention, source, metadata, scratch):
        self.attention, self.source, self.metadata = attention, source, metadata
        self.expanded = torch.zeros_like(scratch.expanded)
        self.output = torch.zeros_like(scratch.output)
        self.slab = torch.zeros_like(scratch.slab)
        # A forward allocates the records of a gather that packs them; the stored shape records whether it did.
        self.records = torch.zeros_like(scratch.records)
        # What the indexer selected for this forward.
        self.selected = torch.zeros_like(scratch.index.physical)
        self.selected_lengths = torch.zeros_like(scratch.index.selected_lengths)
        self.pools = torch.zeros_like(scratch.index.global_pools)

    def __call__(self):
        _, local = self.attention.forward(self.source, self.metadata)
        self.expanded.copy_(local.expanded)
        self.output.copy_(local.output)
        self.slab.copy_(local.slab)
        self.records.copy_(local.records)
        self.selected.copy_(local.index.physical)
        self.selected_lengths.copy_(local.index.selected_lengths)
        self.pools.copy_(local.index.global_pools)


def _oracle_slab(caches, table, end):
    """The slab of the first `end` positions, read from every owner's cache bytes."""
    token = torch.arange(end)
    local = (token // 16) * 4 + token % 4
    record = table[(local // 256).long()].long() * 256 + local % 256
    slab = torch.zeros((PAGES * 256 * 4, 528), dtype=torch.uint8)
    for owner, cache in enumerate(caches):
        mine = (token // 4) % 4 == owner
        slab[token[mine]] = cache.view(-1, 528)[record[mine]]
    return slab


def _model_worker(rank, rendezvous, snapshot, indexer_selects=False):
    from dataclasses import replace

    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference

    from ignition import runtime_patches
    runtime_patches.apply()
    import torch.distributed as dist
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.shared_pages import SharedPages
    from ignition.preparation.capture import capture_forwards
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
        create_varlen_gather_runtime)
    from ignition.preparation.execution import GraphSignature
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53.dcp import ContextParallel, ContextShards
    from ignition.recipes.glm53.indexer import IndexMetadata
    from ignition.recipes.glm53.mla import MLAScratch, MLAState, PreparedGlmMLA
    from ignition.recipes.glm53.model import create_model_tree
    from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(minutes=10))
    shared = records = candidates = None
    graphs = {}
    try:
        coordinator = CollectivePreparation(dist.group.WORLD)
        records = create_varlen_gather_runtime(dist.group.WORLD, device)
        candidates = create_all_gather_runtime(dist.group.WORLD, device, max_rows=ROWS, columns=2048 * 4)
        tree = create_model_tree(snapshot)
        with GlmWeightLoader(inspect_indexed_checkpoint(snapshot), device, shard=TensorParallelShard(rank, 4),
                prefixes=(f"{TEXT_PREFIX}.layers.3.self_attn.",)) as loader:
            weights = loader.load_attention(3)
        module = tree.model.language_model.layers[3].self_attn

        # Every rank's latent cache is readable by its peers; the table and the input are the same on every rank.
        shared = SharedPages(dist.group.WORLD, device, PAGES * 256 * 528)
        generator = torch.Generator(device).manual_seed(53 + rank)
        shared.tensor.view(-1, 64, 528).copy_(pack_mla_kv_cache_reference(
            torch.randn(PAGES * 256, 512, device=device, generator=generator)).view(-1, 64, 528))
        state = replace(MLAState.allocate(PAGES, 4, device), cache=shared.tensor.view(PAGES * 4, 64, 528),
                        peers=shared.addresses)
        state.index.cache.zero_()
        state.index.tail.normal_(std=.1)
        if indexer_selects:
            # Index keys repeat: a few valid byte patterns, so that equal scores occur at the selection cutoff.
            rows = state.index.cache.view(-1, state.index.cache[0].numel())
            templates = torch.randint(0x28, 0x48, (16, rows.shape[1]), dtype=torch.uint8, generator=generator, device=device)
            rows.copy_(templates[torch.randint(0, 16, (rows.shape[0],), generator=generator, device=device)])
        saved = (state.index.cache.clone(), state.index.tail.clone())
        shards = ContextShards.on(rank, 4, device)
        common = torch.Generator().manual_seed(7)
        source = (torch.randn((ROWS, 4096), generator=common) * .1).to(torch.bfloat16).to(device)
        metadata = IndexMetadata.allocate(ROWS, 1, PAGES, device)
        table = torch.randperm(PAGES, generator=common).int()
        metadata.block_table[0].copy_(table)
        metadata.positions.copy_(torch.arange(ROWS, device=device) + START)
        metadata.state_slots.fill_(2)
        metadata.query_start_loc.copy_(torch.tensor([0, ROWS], dtype=torch.int32))
        metadata.token_slots.copy_(shards.token_slots(metadata.positions, metadata.block_table[0].expand(ROWS, -1)))
        physical, lengths = (value.to(device) for value in _selection(START, ROWS, 11))

        arena = ScratchArena(device)
        forwards = {}
        with RecordedPreparationSession(device=device, checkpoint="glm-context-gather-model",
                workload_schema="tp4-dcp4-glm-mla-prefill", numerical_policy="exact-cache-bytes",
                coordinator=coordinator, compile_workers=4) as session:
            session.register_scratch(arena)
            for kind in ("pull", "ring"):
                # As in preparation: peer pull has no records runtime and no record scratch; the ring has both.
                context = ContextParallel(shards, heads=None, candidates=candidates, coordinator=coordinator,
                                          records=records if kind == "ring" else None)
                scratch = MLAScratch.allocate(ROWS, 16, PAGES, "prefill", device, context_ranks=4,
                                              gather_context=kind == "ring")
                attention = PreparedGlmMLA(module, weights, source, scratch, state, metadata, mode="prefill",
                    session=session, name=f"mla.{kind}", workspace=arena, context=context)
                if kind == "ring":
                    attention.context_gather.bind("ring")
                assert attention.context_gather.implementation == kind
                assert attention.context_gather.uses_records == (kind == "ring")
                if not indexer_selects:
                    attention.attend_gathered_context = _FixedSelection(attention, physical, lengths)
                attention.functional(session=session)
                forwards[kind] = _ModelForward(attention, source, metadata, scratch)
            session.tune(show_progress=False)
            arena.release_preparation()
            graphs, _ = capture_forwards(forwards, {kind: GraphSignature("prefill", 1, ROWS) for kind in forwards},
                session=session, device=device, stream=torch.cuda.Stream(device=device),
                pool=torch.cuda.graph_pool_handle(), coordinator=coordinator, report_catalog=False)
            results = {}
            for kind in ("pull", "ring"):
                for repeat in range(2):
                    for tensor in (forwards[kind].expanded, forwards[kind].output, forwards[kind].slab,
                                   forwards[kind].records):
                        tensor.zero_()
                    # Every replay starts from the same index state.
                    state.index.cache.copy_(saved[0])
                    state.index.tail.copy_(saved[1])
                    torch.cuda.synchronize(device)
                    coordinator.barrier(f"{kind} replay {repeat}", coordinator.ranks)
                    graphs[kind].replay()
                    torch.cuda.synchronize(device)
                    results[kind, repeat] = {name: getattr(forwards[kind], name).cpu()
                                             for name in ("expanded", "output", "slab", "records", "selected",
                                                          "selected_lengths", "pools")}
            caches = [torch.empty(PAGES * 256 * 528, dtype=torch.uint8) for _ in range(4)]
            dist.all_gather(caches, shared.tensor.cpu())

        end = START + ROWS
        expected = _oracle_slab(caches, table.cpu(), end)
        # The selection reaches every owner and both windows of the local records.
        chosen = results["pull", 0]["selected"] if indexer_selects else physical.cpu()
        covered = {(int(position) // 4 % 4, ((int(position) // 16) * 4 + int(position) % 4) // WINDOW)
                   for position in chosen.flatten().tolist() if position >= 0}
        if indexer_selects:
            print(f"rank {rank}: indexer selection covers {len(covered)} of 8 (owner, window) cells", flush=True)
        else:
            assert covered == {(owner, window) for owner in range(4) for window in range(2)}, covered
        for kind in ("pull", "ring"):
            first, again = results[kind, 0], results[kind, 1]
            # The gathered slab is the owners' bytes at every position of the context, and zero beyond it.
            assert torch.equal(first["slab"][:end], expected[:end]), f"{kind} slab differs from the owners' records"
            assert not first["slab"][end:].any(), f"{kind} slab is not zero beyond the context"
            # With the fixed selection the indexer's own output is unused and may differ between replays.
            for name in (first if indexer_selects else ("expanded", "output", "slab", "records")):
                assert torch.equal(first[name], again[name]), f"{kind} {name} differs between replays"
            assert torch.isfinite(first["expanded"].float()).all() and first["expanded"].any()
        assert not results["pull", 0]["records"].numel() and results["ring", 0]["records"].any()
        names = ("slab", "expanded", "output") + (("selected", "selected_lengths", "pools") if indexer_selects else ())
        differing = {}
        for name in names:
            a, b = results["pull", 0][name], results["ring", 0][name]
            if not torch.equal(a, b):
                differing[name] = int((a != b).sum())
        if indexer_selects:
            print(f"rank {rank}: ring differs from pull in {differing or 'nothing'}", flush=True)
        assert not differing, f"ring differs from pull in {differing}"
    finally:
        for graph in graphs.values():
            graph.reset()
        if shared is not None:
            shared.close()
        for runtime in (records, candidates):
            if runtime is not None:
                runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("indexer_selects", [False, True], ids=["fixed-selection", "indexer-selection"])
def test_glm_mla_prefill_ring_matches_peer_pull_tp4(tmp_path, indexer_selects):
    """The ring and peer pull gather the same slab and give bit-identical attention output through PreparedGlmMLA."""
    import torch.multiprocessing as mp

    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    mp.spawn(_model_worker, args=((tmp_path / "gloo").as_uri(), snapshot, indexer_selects), nprocs=4, join=True)
