# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared GLM index head projections preserve FP32 output from BF16 checkpoint operands."""

from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.rotary import PreparedRotaryAngles
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.reference.glm_moe_dsa import CheckpointReader
from ignition.reference.glm_moe_dsa_attention import fp32_dot_allowance

from . import glm53_full_support as support

pytestmark = support.requires_reference


@pytest.mark.parametrize("capacity", [1, 8])
@torch.no_grad()
def test_index_head_projection_retains_fp32_in_graph_replay(capacity):
    from ignition.recipes.glm53_full.attention import AttentionMetadata, PreparedAttention, rotary_frequencies
    from ignition.recipes.glm53_full.weights import GlmWeightLoader

    device = torch.device("cuda:0")
    manifest = inspect_indexed_checkpoint(Path(support.SNAPSHOT))
    fixture = support.fixture("short")
    config, canonical = support.canonical_layer(manifest, 6, torch.device("cpu"))
    source = canonical.input_layernorm(fixture["layers"][5][-capacity:].float()).bfloat16().to(device)
    reader = CheckpointReader(manifest, torch.device("cpu"), torch.float32)
    weight = reader.floating("model.layers.6.self_attn.indexer.weights_proj.weight").double()
    reader.close()
    expected = source.cpu().double() @ weight.T
    magnitude = source.cpu().double().abs() @ weight.abs().T
    allowance = fp32_dot_allowance(expected, magnitude, 6144, bf16_output=False)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(0, 8),
                         prefixes=("model.layers.6.self_attn.",)) as loader:
        weights = loader.load_attention(6)
    metadata = AttentionMetadata.allocate(capacity, 1, 1, device)
    cache = torch.zeros((4, 64, 656), dtype=torch.uint8, device=device)
    arena = ScratchArena(device, overlap=False)
    angles = PreparedRotaryAngles(metadata.positions, rotary_frequencies(config.rope_parameters["rope_theta"], device))
    with RecordedPreparationSession(device=device, checkpoint="glm53-full-test",
            workload_schema="glm53-index-head-precision", numerical_policy="default") as session:
        session.register_scratch(arena)
        attention = PreparedAttention(weights, source, metadata, cache, angles, heads=8, selects=True,
            eps=1e-6, softmax_scale=1/16, mode="decode", session=session,
            name="index.head_precision", workspace=arena)
        attention.functional(session=session)
        angles.functional()
        session.tune()
        arena.release_preparation()
        metadata.positions.copy_(torch.arange(capacity, device=device))
        metadata.token_slots.copy_(metadata.positions)
        metadata.block_table.zero_()
        rotary = angles(metadata.positions)
        for _ in range(2):
            attention.project(source, metadata, rotary)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projected = attention.project(source, metadata, rotary)
        actual = projected.index_weights
        assert actual.dtype == torch.float32, "index head weights must retain the canonical FP32 projection result"
        initial = None
        for _ in range(3):
            graph.replay()
            value = actual.cpu().double()
            assert torch.isfinite(value).all() and ((value - expected).abs() <= allowance).all()
            if initial is None:
                initial = value
            else:
                assert torch.equal(value, initial)
        session.freeze()
