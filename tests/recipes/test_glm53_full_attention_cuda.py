# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 prepared attention equals the Transformers attention of one layer on reference inputs."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.rotary import PreparedRotaryAngles
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

from . import glm53_full_support as support

pytestmark = support.requires_reference


@pytest.fixture(scope="module")
def manifest():
    return inspect_indexed_checkpoint(Path(support.SNAPSHOT))


@torch.no_grad()
def prepared_attention_output(manifest, layer, world, source, records, lengths, *, rotate=True, decode_rows=0):
    """Sum over ranks of prepared attention outputs for one prefill of every row.

    With `decode_rows`, also return the sum of decode outputs that recompute the
    trailing rows one at a time against the records the prefill wrote.
    """
    from ignition.recipes.glm53_full.attention import (
        PAGE_TOKENS, AttentionMetadata, PreparedAttention, Selection, rotary_frequencies)
    from ignition.recipes.glm53_full.config import ModelConfig
    from ignition.recipes.glm53_full.weights import GlmWeightLoader

    device, rows = source.device, source.shape[0]
    config = ModelConfig.load(manifest.snapshot)
    pages = -(-rows // PAGE_TOKENS)
    total = torch.zeros((rows, 6144), dtype=torch.float32, device=device)
    decoded = torch.zeros((decode_rows, 6144), dtype=torch.float32, device=device)
    for rank in range(world):
        shard = TensorParallelShard(rank, world)
        with GlmWeightLoader(manifest, device, shard=shard, prefixes=(f"model.layers.{layer}.self_attn.",)) as loader:
            weights = loader.load_attention(layer)
        cache = torch.zeros((pages * 4, 64, 656), dtype=torch.uint8, device=device)
        frequencies = rotary_frequencies(config.rope_theta, device)
        arena = ScratchArena(device, overlap=False)
        with RecordedPreparationSession(device=device, checkpoint="glm53-full-test",
                workload_schema="glm53-full-attention-v1", numerical_policy="default") as session:
            session.register_scratch(arena)
            prepared = {}
            for mode, capacity in (("prefill", rows), *((("decode", 1),) if decode_rows else ())):
                metadata = AttentionMetadata.allocate(capacity, 1, pages, device)
                angles = PreparedRotaryAngles(metadata.positions, frequencies)
                attention = PreparedAttention(weights, source[:capacity].clone(), metadata, cache, angles,
                    heads=64 // world, selects=config.selects(layer), eps=1e-6,
                    softmax_scale=256 ** -0.5, mode=mode, session=session, name=f"attention.{mode}",
                    workspace=arena)
                attention.functional(session=session)
                angles.functional()
                prepared[mode] = (attention, angles, metadata)
            session.tune()
            arena.release_preparation()
            cache.zero_()

            def run(mode, first, last):
                attention, angles, metadata = prepared[mode]
                positions = torch.arange(first, last, device=device)
                metadata.positions.copy_(positions)
                metadata.token_slots.copy_(positions)
                metadata.block_table.copy_(torch.arange(pages, dtype=torch.int32, device=device))
                values = angles(metadata.positions if rotate else torch.zeros_like(metadata.positions))
                projected = attention.project(source[first:last], metadata, values)
                return attention.attend(projected.absorbed, Selection(records[first:last], lengths[first:last]),
                                        metadata).float()

            total += run("prefill", 0, rows)
            for index in range(decode_rows):
                row = rows - decode_rows + index
                decoded[index] += run("decode", row, row + 1)[0]
            session.freeze()
        del prepared, weights, cache
        torch.cuda.empty_cache()
    return total, decoded


@pytest.mark.parametrize("sequence,layer,world", [
    ("short", 1, 1), ("short", 5, 1), ("long", 1, 1), ("long", 5, 1), ("long", 5, 8), ("long", 1, 8)])
@torch.no_grad()
def test_attention_matches_transformers(manifest, sequence, layer, world):
    """Layer 1 selects its own positions; layer 5 reuses layer 2's selection."""
    device = torch.device("cuda:0")
    reference = support.fixture(sequence)
    config, module = support.canonical_layer(manifest, layer, device)
    shared = None if module.self_attn.indexer is not None else support.prior_selection(reference, layer)
    hidden = support.layer_input(reference, layer, device)
    normalized, expected, selected = support.canonical_attention(config, module, hidden, shared)
    records, lengths = support.causal_selection(selected, device)
    source = normalized.to(torch.bfloat16)
    decode_rows = 3 if world == 1 else 0
    actual, decoded = prepared_attention_output(manifest, layer, world, source, records, lengths,
                                                decode_rows=decode_rows)
    errors = support.row_errors(actual, expected)
    print(f"{sequence} layer {layer} world {world}: {errors}")
    # Proposed bound: twice the measured error of BF16 arithmetic with FP8 latent records on these inputs.
    assert errors["median"] < 0.03 and errors["relative_l2"] < 0.03, errors
    if decode_rows:
        decode = support.row_errors(decoded, expected[-decode_rows:])
        print(f"{sequence} layer {layer} decode: {decode}")
        assert decode["maximum"] < 0.05, decode
    # Without rotation the same comparison must fail.
    unrotated, _ = prepared_attention_output(manifest, layer, world, source, records, lengths, rotate=False)
    control = support.row_errors(unrotated, expected)
    print(f"{sequence} layer {layer} world {world} without rotation: {control}")
    assert control["median"] > 0.1, control
