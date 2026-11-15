# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-sharded Inkling 2 prefill layers equal the replicated layers bit for bit, without a checkpoint.

One GPU runs every rank of a tensor-parallel group in turn, twice: with the row-local stages
replicated on every rank, and with them sharded by rows. The RoCEnante transport is replaced by a
group whose collectives exchange the ranks' actual operands: reductions sum them in FP32 in rank
order, gathers copy them. Attention and the routed experts, which are sharded by column and not by
row, are replaced by rank-dependent functions that read every row of their inputs, so a row that a
gather misplaced changes the following reduction. The layers, the stack, the row-shard collectives,
the mixtures, convolutions, router projection and router are the production ones with synthetic
weights. One case also compiles one rank's sharded forward, captures it in a CUDA graph and replays
it against the operands its peers exchanged.
"""

import os
import threading
from types import SimpleNamespace

import pytest
import torch

from ignition.components import collectives, row_shards
from ignition.components.paged_batch import Metadata

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")

LAYERS, RESIDUAL_BLOCK, GLOBAL_LAYERS, TAPS, HALO, TOP_K = 5, 2, (1, 4), (2, 4, 5), 3, 8


class Group:
    """Ranks of one group taking turns on one GPU; a collective completes when every rank has entered it.

    Every exchange is logged. `replay` then runs one rank alone: each of its collectives combines
    its live operand with the operands its peers contributed to the logged exchange of that kind
    whose operand from this rank has the same bits. A compiled forward may order independent
    collectives differently from the eager forward, identically on every rank, so exchanges are
    matched by content and not by ordinal. `replay_order` repeats the matched order without reading
    operands, as CUDA graph capture requires.
    """

    def __init__(self, world):
        self.world = world
        self.turn = threading.Lock()
        self.barrier = threading.Barrier(world)
        self.operands = [None] * world
        self.kinds = [[] for _ in range(world)]
        self.log = [[] for _ in range(world)]
        self.replayed = self.ordered = None
        self.order = []

    def replay(self, rank):
        self.replayed, self.ordered, self.order = list(self.log[rank]), None, []

    def replay_order(self):
        self.replayed, self.ordered = None, iter(self.order)

    def exchange(self, rank, kind, operand):
        if self.ordered is not None:
            logged_kind, operands = next(self.ordered)
            assert logged_kind == kind
            return [operand if peer == rank else value for peer, value in enumerate(operands)]
        if self.replayed is not None:
            for index, (logged_kind, operands) in enumerate(self.replayed):
                if (logged_kind == kind and operands[rank].shape == operand.shape
                        and torch.equal(operands[rank].view(torch.int16), operand.view(torch.int16))):
                    self.order.append(self.replayed.pop(index))
                    return [operand if peer == rank else value for peer, value in enumerate(operands)]
            raise AssertionError(f"rank {rank} launched a {kind} whose operand no logged exchange holds")
        self.kinds[rank].append(kind)
        self.operands[rank] = operand
        self.turn.release()
        try:
            self.barrier.wait()
            operands = list(self.operands)
            self.barrier.wait()
        finally:
            self.turn.acquire()
        self.log[rank].append((kind, operands))
        return operands

    def reduce(self, rank, kind, source):
        total = None
        for operand in self.exchange(rank, kind, source):
            total = operand.float() if total is None else total + operand.float()
        return total.to(source.dtype)

    def run(self, forwards):
        results, errors = [None] * self.world, []

        def rank_main(rank):
            with self.turn:
                try:
                    with torch.no_grad():
                        results[rank] = forwards[rank]()
                    torch.cuda.synchronize()
                except BaseException as error:
                    errors.append(error)
                    self.barrier.abort()

        threads = [threading.Thread(target=rank_main, args=(rank,)) for rank in range(self.world)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        assert all(kinds == self.kinds[0] for kinds in self.kinds)
        return results


class RankRuntime:
    """One rank's transport: the operations the prepared collectives launch, over the group's exchange."""

    def __init__(self, group, rank):
        self.group, self.rank, self.world_size = group, rank, group.world

    def reduce_scatter(self, source, *, out, plan):
        rows = out.shape[0]
        out.copy_(self.group.reduce(self.rank, "reduce_scatter", source)[self.rank * rows:(self.rank + 1) * rows])

    def gather_shards(self, source, *, out, plan):
        out.copy_(torch.cat(self.group.exchange(self.rank, f"gather_shards{source.shape[1]}", source)))

    def all_gather(self, source, *, dim, out, plan):
        out.copy_(torch.cat(self.group.exchange(self.rank, "all_gather", source), dim=dim))

    def all_reduce(self, source):
        return self.group.reduce(self.rank, "all_reduce", source)


def _bind(component, runtime, output):
    component.runtime, component.output, component.single, component.plan = runtime, output, False, None
    component.handle = next(collectives._HANDLES)
    collectives._OWNERS[component.handle] = component


class ReduceScatter(collectives.PreparedReduceScatter):
    def __init__(self, source, output, *, runtime, session, coordinator, name):
        assert source.dtype == torch.bfloat16 and output.is_contiguous()
        assert output.shape == (source.shape[0] // runtime.world_size, source.shape[1])
        _bind(self, runtime, output)


class ShardGather(collectives.PreparedShardGather):
    def __init__(self, source, output, *, runtime, session, coordinator, name):
        assert source.dtype == torch.bfloat16 and source.shape[1] * 2 % 16 == 0 and output.is_contiguous()
        assert output.shape == (source.shape[0] * runtime.world_size, source.shape[1])
        _bind(self, runtime, output)


class EdgeGather(collectives.PreparedAllGather):
    def __init__(self, source, output, *, runtime, session, coordinator, name, dependencies=()):
        assert source.dtype == torch.bfloat16 and source.shape[1] * 2 % 16 == 0
        assert output.shape == (source.shape[0], source.shape[1] * runtime.world_size)
        _bind(self, runtime, output)
        self.roce, self.dependencies = True, []


class Attention:
    """A rank's attention partial sums as a function of every row of every input stream."""

    def __init__(self, layer, weights, text, cache, metadata, **kwargs):
        self.layer, self.rank, self.marker = layer, weights.rank, weights

    def attend(self, streams, metadata):
        if self.marker.seen is not None:
            self.marker.seen[self.layer, "streams"] = streams
        mixed = sum(stream.float() for stream in streams)
        partial = mixed * (.25 + .125 * self.rank) + mixed.roll(1 + self.rank, 0) * .0625
        return None, partial.to(torch.bfloat16), None


class Experts:
    """A rank's expert partial sums as a function of every row's input, expert IDs and weights."""

    def __init__(self, source, weights, ids, route_weights, partial, *, name, **kwargs):
        self.name, self.rank, self.marker = name.split(".", 2)[2], weights.rank, weights

    def functional(self, *, session):
        return self

    def __call__(self, source, ids, weights):
        if self.marker.seen is not None:
            self.marker.seen[self.name] = (source, ids, weights)
        value = source.float()
        partial = (value * weights.sum(1, keepdim=True) * (.5 + .25 * self.rank)
                   + ids.float().sum(1, keepdim=True) * 2 ** -12 + value.roll(2, 0) * .03125)
        return partial.to(torch.bfloat16)


def _stack_weights(hidden, experts, device, generator):
    def random(*shape, scale=1., dtype=torch.float32):
        return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)

    layers = []
    for layer in range(LAYERS):
        streams = 3 if layer in GLOBAL_LAYERS else 1
        layers.append(dict(router=random(experts, hidden, scale=hidden ** -.5, dtype=torch.bfloat16),
            correction=random(experts, scale=.1),
            attention_convolution=random(hidden, 1, HALO + 1, scale=.3),
            mlp_convolution=random(hidden, 1, HALO + 1, scale=.3),
            input_norms=tuple(1 + random(hidden, scale=.1, dtype=torch.bfloat16) for _ in range(streams)),
            mlp_norm=1 + random(hidden, scale=.1, dtype=torch.bfloat16),
            attention_query=None if layer == 0 else random(streams, hidden, scale=hidden ** -.5).contiguous(),
            mlp_query=random(1, hidden, scale=hidden ** -.5)))
    return dict(layers=layers, output_query=random(1, hidden, scale=hidden ** -.5),
                final_norm=1 + random(hidden, scale=.1, dtype=torch.bfloat16))


def _rank_stack(shared, text, history, metadata, rank, group, *, sharded, tokens, session, arena, examples, device):
    from ignition.recipes.inkling2.decoder import PreparedStack

    seen, runtime = {}, RankRuntime(group, rank)
    marker = SimpleNamespace(rank=rank, seen=seen)
    weights = SimpleNamespace(config=SimpleNamespace(text=text, draft=SimpleNamespace(taps=TAPS)), draft=marker,
        layers=[SimpleNamespace(attention=marker, experts=marker, **layer) for layer in shared["layers"]],
        output_query=shared["output_query"], final_norm=shared["final_norm"])
    histories = {}
    pool = SimpleNamespace(memory_capacity=torch.tensor([2, 8]), ring_row_pages=1, geometry=SimpleNamespace(ring=128),
        history=lambda layer, name: histories.setdefault((layer, name), history[layer, name].clone()))
    rows = route_gather = None
    if sharded:
        shards = row_shards.RowShards.of(rank, group.world, tokens, device, halo=HALO)
        collective = dict(runtime=runtime, session=session, coordinator=None, examples=examples)
        rows = row_shards.RowShardCollectives(shards, text.hidden, requests=1, name=f"rank{rank}.rows", **collective)
        route_gather = row_shards.PreparedRouteGather(shards, TOP_K, name=f"rank{rank}.routes", **collective)
    stack = PreparedStack(weights, pool, metadata, runtime.all_reduce, torch.bfloat16, mode="prefill", tokens=tokens,
        session=session, workspace=arena, name=f"rank{rank}.stack{int(sharded)}", examples=examples, rows=rows,
        route_gather=route_gather)
    stack.marker = marker
    return stack, seen, histories


def _compiled_replay(stack, histories, history, expected, expected_histories, source, m, group, rank, session):
    """One rank's sharded forward, compiled and captured, equals its eager result on every replay."""
    from ignition.preparation.compilation import compile_forward

    def restore():
        for key, value in histories.items():
            value.copy_(history[key])

    def check(result):
        normalized, taps, _ = result
        assert torch.equal(normalized, expected[0])
        assert all(torch.equal(got, want) for got, want in zip(taps, expected[1], strict=True))
        assert all(torch.equal(value, expected_histories[key]) for key, value in histories.items())

    stack.marker.seen = None
    restore()
    group.replay(rank)
    compiled = compile_forward(lambda: stack(source, m))
    restore()
    group.replay(rank)
    check(compiled())
    assert not group.replayed
    restore()
    group.replay_order()
    graph = torch.cuda.CUDAGraph()
    with session.capture(), torch.cuda.graph(graph):
        result = compiled()
    for _ in range(3):
        for value in (result[0], *result[1]):
            value.fill_(float("nan"))
        restore()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        check(result)
    graph.reset()


# World size, chunk capacity, hidden width, experts, live rows and the chunk's first position: the
# model's geometry with a full and a padded chunk at a nonzero start, then smaller chunks whose
# live rows end one row past a shard boundary, inside the first shard's first rows, and nowhere.
# The last argument selects a rank whose sharded forward is also compiled and replayed.
@pytest.mark.parametrize("world,tokens,hidden,experts,live,start,compiled_rank", [
    (4, 2048, 4608, 384, 2048, 0, None), (4, 2048, 4608, 384, 1500, 8192, None), (4, 256, 512, 64, 65, 3, 1),
    (2, 256, 512, 64, 2, 0, None), (4, 256, 512, 64, 130, 0, None), (4, 256, 512, 64, 0, 0, None)])
@torch.no_grad()
def test_row_sharded_layers_match_replicated_layers(world, tokens, hidden, experts, live, start, compiled_rank,
                                                    monkeypatch):
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.inkling2 import decoder

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    monkeypatch.setattr(decoder, "PreparedAttention", Attention)
    monkeypatch.setattr(decoder, "PreparedRoutedExperts", Experts)
    monkeypatch.setattr(decoder, "LayerCache", SimpleNamespace(of=lambda pool, layer: None))
    monkeypatch.setattr(row_shards, "PreparedReduceScatter", ReduceScatter)
    monkeypatch.setattr(row_shards, "PreparedShardGather", ShardGather)
    monkeypatch.setattr(row_shards, "PreparedAllGather", EdgeGather)
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(1115)
    text = SimpleNamespace(hidden=hidden, experts=experts, experts_per_token=TOP_K, route_scale=2.5,
        residual_block=RESIDUAL_BLOCK, eps=1e-6, local=lambda layer: layer not in GLOBAL_LAYERS)
    shared = _stack_weights(hidden, experts, device, generator)
    history = {(layer, name): torch.randn((3, HALO, hidden), generator=generator).to(device, torch.bfloat16)
               for layer in range(LAYERS) for name in ("attention_convolution", "mlp_convolution")}
    m = Metadata.allocate(tokens, 1, 1, device)
    hidden_rows = torch.randn((tokens, hidden), generator=generator).to(device, torch.bfloat16)
    group = Group(world)
    with RecordedPreparationSession(device=device, checkpoint="inkling2-row-shards",
            workload_schema="inkling2-row-shards", numerical_policy="default", autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        examples = {}
        variants = {sharded: [_rank_stack(shared, text, history, m, rank, group, sharded=sharded, tokens=tokens,
                                          session=session, arena=arena, examples=examples, device=device)
                              for rank in range(world)] for sharded in (False, True)}
        session.tune()
        m.set(torch.tensor([start], device=device), torch.tensor([live], dtype=torch.int32, device=device),
              torch.tensor([1], device=device), torch.zeros((1, 1), dtype=torch.int32, device=device),
              torch.zeros((1, tokens), dtype=torch.int64, device=device), torch.tensor([1], device=device))
        source = torch.where((m.positions >= 0)[:, None], hidden_rows, 0)
        results = {}
        for sharded, ranks in variants.items():
            group.kinds, group.log = [[] for _ in range(world)], [[] for _ in range(world)]
            results[sharded] = group.run([lambda stack=stack: stack(source, m) for stack, _, _ in ranks])
            kinds = group.kinds[0]
            if sharded:
                # Per layer: one row gather per attention stream after the first layer, two
                # reduce-scatters each followed by an edge exchange, the route gather and the expert
                # input gather; then one gather per tap before a layer and the final gather.
                assert kinds.count("reduce_scatter") == kinds.count("all_gather") == 2 * LAYERS
                assert kinds.count(f"gather_shards{4 * TOP_K}") == LAYERS
                streams = sum(3 if layer in GLOBAL_LAYERS else 1 for layer in range(1, LAYERS))
                assert kinds.count(f"gather_shards{hidden}") == streams + LAYERS + len(TAPS)
                assert "all_reduce" not in kinds
            else:
                assert kinds == ["all_reduce"] * (2 * LAYERS)
        if compiled_rank is not None:
            stack, _, histories = variants[True][compiled_rank]
            expected_histories = {key: value.clone() for key, value in histories.items()}
            _compiled_replay(stack, histories, history, results[True][compiled_rank], expected_histories, source, m,
                             group, compiled_rank, session)

    reference, reference_seen, reference_histories = results[False][0], variants[False][0][1], variants[False][0][2]
    normalized, taps, _ = reference
    assert len(taps) == len(TAPS) and torch.isfinite(normalized).all()
    assert bool(normalized[:live].any()) == (live > 0) and not normalized[live:].any()
    for sharded, ranks in variants.items():
        for rank, (_, seen, histories) in enumerate(ranks):
            actual, actual_taps, _ = results[sharded][rank]
            assert torch.equal(actual, normalized), (sharded, rank)
            for index, (got, want) in enumerate(zip(actual_taps, taps, strict=True)):
                assert torch.equal(got, want), (sharded, rank, index)
            assert seen.keys() == reference_seen.keys()
            for key, values in seen.items():
                for got, want in zip(values, reference_seen[key], strict=True):
                    # NaN-free integers and floats: IDs, route weights, attention streams and expert inputs.
                    assert got.dtype == want.dtype and torch.equal(got, want), (sharded, rank, key)
            for key, value in histories.items():
                assert torch.equal(value, reference_histories[key]), (sharded, rank, key)
                assert torch.equal(value[0], history[key][0]) and torch.equal(value[2], history[key][2])
                assert (live > 0) or torch.equal(value, history[key])
    routes = reference_seen["layer0.experts"]
    assert bool((routes[1][:live] >= 0).all()) and bool((routes[1][live:] == -1).all())
