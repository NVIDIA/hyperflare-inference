# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Replay the mixed GLM router/gate projection with serial and concurrent expert contributions."""

from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.weight_first_projection import PreparedWeightFirstProjection
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53_full.attention import ExampleBuffers
from ignition.recipes.glm53_full.decoder import PreparedMoE
from ignition.recipes.glm53_full.weights import GlmWeightLoader
from . import glm53_full_support as support


@support.requires_reference
@torch.inference_mode()
def test_projected_shared_expert_matches_serial_routing_and_join():
    device = torch.device("cuda:0")
    manifest = inspect_indexed_checkpoint(Path(support.SNAPSHOT))
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(0, 8),
                         prefixes=("model.layers.8.mlp.",)) as loader:
        weights, config = loader.load_mlp(8), loader.config
    fixture = support.layer_input(support.fixture("short"), 8, device).bfloat16()
    source = torch.empty_like(fixture[:4])
    source.copy_(fixture[:4])
    seen_shared, seen_routed, seen_output = (torch.empty_like(source) for _ in range(3))
    arena = ScratchArena(device, overlap=True)
    with RecordedPreparationSession(device=device, checkpoint="glm53-full-test",
            workload_schema="glm53-projected-shared-expert", numerical_policy="default") as session:
        session.register_scratch(arena)
        moe = PreparedMoE(config, source, weights, session=session, name="moe", workspace=arena,
                          buffers=ExampleBuffers(device), overlap_gate_up=True)
        mixed = PreparedWeightFirstProjection(source, weights.router, weights.shared.gate_up,
                                              session=session, name="input_projections")
        moe.functional(session=session)
        session.tune()
        arena.release_preparation()

        def serial():
            logits, gate_up = mixed(source)
            route = moe.routing(logits)
            routed = moe.experts(source, route.ids, route.weights)
            shared = moe.shared.from_gate_up(gate_up)
            seen_shared.copy_(shared)
            seen_routed.copy_(routed)
            seen_output.copy_(routed + shared)

        join = moe.concurrent_shared.finish

        def observe_join(pending, routed):
            shared = join(pending, routed)
            seen_routed.copy_(routed)
            seen_shared.copy_(shared)
            # Both branches have finished reading these private operands when the main stream joins them.
            pending.workspace.fill_(173)
            pending.source.zero_()
            for tensor in pending.encoded:
                tensor.zero_()
            return shared

        moe.concurrent_shared.finish = observe_join

        def overlap():
            seen_output.copy_(moe(source, mixed(source)))

        graphs = []
        for run in (serial, overlap):
            compiled = compile_forward(run)
            compiled()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs.append((graph, compiled))
        session.freeze()

        def replay(graph):
            graph.replay()
            torch.cuda.synchronize()
            return [t.cpu().clone() for t in (seen_shared, seen_routed, seen_output)]

        before = torch.cuda.memory_allocated(device)
        for start in (0, 8, 16, 4, 0):
            source.copy_(fixture[start:start+4])
            reference = replay(graphs[0][0])
            actual = replay(graphs[1][0])
            assert all(torch.equal(a, b) for a, b in zip(reference, actual, strict=True))
            corrupted = actual[0].clone()
            corrupted[0, 0] = float("nan")
            with pytest.raises(AssertionError):
                torch.testing.assert_close(reference[0], corrupted, rtol=0, atol=0)
        assert torch.cuda.memory_allocated(device) == before
