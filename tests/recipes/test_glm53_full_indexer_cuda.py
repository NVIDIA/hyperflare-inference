# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 prepared top-2048 selection against the Transformers indexer's ranking of every position."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.rotary import PreparedRotaryAngles
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.reference.glm_moe_dsa_cache import quantize_index
from ignition.reference.glm_moe_dsa_index import index_score_interval, valid_index_cutoff

from . import glm53_full_support as support

pytestmark = support.requires_reference
LAYER = 6


@pytest.fixture(scope="module")
def manifest():
    return inspect_indexed_checkpoint(Path(support.SNAPSHOT))


@pytest.fixture(scope="module")
def canonical(manifest):
    """Normalized layer input, attention output, selection, and full score ranking of the long sequence."""
    device = torch.device("cuda:0")
    reference = support.fixture("long")
    config, module = support.canonical_layer(manifest, LAYER, device)
    hidden = support.layer_input(reference, LAYER, device)
    normalized, expected, selected = support.canonical_attention(config, module, hidden)
    # Ranking every position: the indexer returns positions by descending score.
    module.self_attn.indexer.index_topk = hidden.shape[0]
    _, _, ranking = support.canonical_attention(config, module, hidden)
    rank = torch.empty_like(ranking, dtype=torch.long)
    rank.scatter_(1, ranking.long(), torch.arange(hidden.shape[0], device=device).expand_as(ranking))
    # The fixture's selection, computed on another device, agrees with this ranking up to near ties.
    stored = reference["selections"][LAYER].to(device).long()
    causal = stored <= torch.arange(hidden.shape[0], device=device)[:, None]
    stored_rank = rank.gather(1, stored)[causal]
    assert float((stored_rank < 2048).double().mean()) > 0.9999 and int(stored_rank.max()) < 2048 + 8
    return normalized.to(torch.bfloat16), expected, selected, rank


@torch.no_grad()
def prepared_selection(manifest, source, *, shard=TensorParallelShard(0, 1), mode="prefill", rotate=True, verify_fp8=False):
    """Selections (as positions) and attention output of one rank; decode replays the trailing rows."""
    from ignition.recipes.glm53_full.attention import PAGE_TOKENS, AttentionMetadata, PreparedAttention, rotary_frequencies
    from ignition.recipes.glm53_full.config import ModelConfig
    from ignition.recipes.glm53_full.indexer import PreparedIndexer
    from ignition.recipes.glm53_full.weights import GlmWeightLoader

    device, rows = source.device, source.shape[0]
    config = ModelConfig.load(manifest.snapshot)
    pages = -(-rows // PAGE_TOKENS)
    with GlmWeightLoader(manifest, device, shard=shard, prefixes=(f"model.layers.{LAYER}.self_attn.",)) as loader:
        weights = loader.load_attention(LAYER)
    cache = torch.zeros((pages * 4, 64, 656), dtype=torch.uint8, device=device)
    index_cache = torch.zeros((pages * 4, 64 * 132), dtype=torch.uint8, device=device)
    frequencies = rotary_frequencies(config.rope_theta, device)
    arena = ScratchArena(device, overlap=False)
    decode_rows = 4
    with RecordedPreparationSession(device=device, checkpoint="glm53-full-test",
            workload_schema="glm53-full-indexer-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        prepared = {}
        for name, capacity in (("prefill", rows), ("decode", 1)):
            metadata = AttentionMetadata.allocate(capacity, 1, pages, device)
            angles = PreparedRotaryAngles(metadata.positions, frequencies)
            attention = PreparedAttention(weights, source[:capacity].clone(), metadata, cache, angles,
                heads=64 // shard.world_size, selects=True, eps=1e-6, softmax_scale=256 ** -0.5, mode=name,
                session=session, name=f"attention.{name}", workspace=arena)
            indexer = PreparedIndexer(weights, torch.zeros((capacity, 2048), dtype=torch.bfloat16, device=device),
                torch.zeros((capacity, 128), dtype=torch.bfloat16, device=device),
                torch.zeros((capacity, 32), dtype=torch.float32, device=device), metadata, index_cache, angles,
                mode=name, session=session, name=f"indexer.{name}", workspace=arena)
            attention.functional(session=session)
            indexer.functional(session=session)
            angles.functional()
            prepared[name] = (attention, indexer, angles, metadata)
        session.tune()
        arena.release_preparation()
        cache.zero_()
        index_cache.zero_()

        if verify_fp8:
            reference_keys = torch.empty((rows, 128), dtype=torch.uint8, device=device)
            reference_scales = torch.empty(rows, dtype=torch.float32, device=device)

        def run(name, first, last):
            attention, indexer, angles, metadata = prepared[name]
            positions = torch.arange(first, last, device=device)
            metadata.positions.copy_(positions)
            metadata.token_slots.copy_(positions)
            metadata.block_table.copy_(torch.arange(pages, dtype=torch.int32, device=device))
            values = angles(metadata.positions if rotate else torch.zeros_like(metadata.positions))
            projected = attention.project(source[first:last], metadata, values)
            arguments = (projected.query_residual, projected.index_key, projected.index_weights, metadata, values)
            if verify_fp8:
                rotated = {}
                apply = angles.apply

                def capture(tensor, *args):
                    result = apply(tensor, *args)
                    rotated[tensor.ndim] = tensor.clone()
                    return result

                with patch.object(angles, "apply", capture):
                    selection = indexer(*arguments)
                keys, scales = quantize_index(rotated[2], query=False)
                reference_keys[first:last].copy_(keys)
                reference_scales[first:last].copy_(scales)
                actual_keys = index_cache[:, :8192].contiguous().view(-1, 128)
                actual_scales = index_cache[:, 8192:].contiguous().view(torch.float32).flatten()
                assert torch.equal(actual_keys[first:last], keys) and torch.equal(actual_scales[first:last], scales)
                for start in range(0, last - first, 16):
                    subset = slice(start, min(start + 16, last - first))
                    scores, allowance = index_score_interval(rotated[3][subset], projected.index_weights[subset],
                                                             reference_keys[:last], reference_scales[:last])
                    records = selection.records[subset]
                    lengths = selection.lengths[subset]
                    pos = metadata.positions[subset]
                    valid = valid_index_cutoff(records, lengths, pos, scores, allowance)
                    assert valid.all(), pos[~valid].tolist()
                    if first + start <= rows - 1 < first + start + len(pos):
                        row = records[-1:].clone()
                        worst, best = scores[-1].argmin(), scores[-1].argmax()
                        assert (row == best).any() and not (row == worst).any()
                        row.masked_fill_(row == best, worst)
                        assert not valid_index_cutoff(row, lengths[-1:], pos[-1:], scores[-1:], allowance[-1:]).item()
                        future = records[-1:].clone()
                        future[0, 0] = pos[-1] + 1
                        assert not valid_index_cutoff(future, lengths[-1:], pos[-1:], scores[-1:], allowance[-1:]).item()
            else:
                selection = indexer(*arguments)
            return selection, attention.attend(projected.absorbed, selection, metadata).float()

        selection, output = run("prefill", 0, rows)
        decoded = [run("decode", row, row + 1) for row in range(rows - decode_rows, rows)] if mode == "decode" else []
        session.freeze()
    return selection, output, decoded


def overlap(selection, selected, rank):
    """Per row: whether valid records lead, the kept count, and how the kept positions rank canonically."""
    records, lengths = selection.records.long(), selection.lengths.long()
    rows = records.shape[0]
    column = torch.arange(records.shape[1], device=records.device)
    valid = column[None] < lengths[:, None]
    assert bool(((records >= 0) == valid).all()), "valid records must form a prefix of every row"
    expected = torch.minimum(torch.arange(1, rows + 1, device=records.device), torch.tensor(2048, device=records.device))
    ranks = torch.where(valid, rank.gather(1, records.clamp(min=0)), -1)
    inside = ((ranks >= 0) & (ranks < 2048)).sum(1)
    return lengths, expected, inside, ranks.max(1).values


@torch.no_grad()
def test_selection_matches_transformers_ranking(manifest, canonical):
    source, expected_output, selected, rank = canonical
    selection, output, decoded = prepared_selection(manifest, source, mode="decode", verify_fp8=True)
    lengths, expected, inside, worst = overlap(selection, selected, rank)
    rows = source.shape[0]
    assert torch.equal(lengths, expected)
    # Identity page table: a physical record index is its position. No row selects a later position.
    assert bool((selection.records.long().max(1).values <= torch.arange(rows, device=source.device)).all())
    assert all(len(set(row[:count].tolist())) == count
               for row, count in zip(selection.records[2040:2060].cpu(), lengths[2040:2060].cpu()))
    # Rows with at most 2048 visible positions select all of them.
    assert bool((inside[:2048] == expected[:2048]).all())
    tail = slice(2048, rows)
    fraction = inside[tail].double() / 2048
    excess = (worst[tail] - 2047).clamp(min=0)
    print(f"rows beyond 2048 visible positions: {rows - 2048}; canonical top-2048 kept: mean {fraction.mean():.5f}, "
          f"minimum {fraction.min():.5f}; rows with identical sets: {int((inside[tail] == 2048).sum())}; "
          f"deepest canonical rank selected: {int(worst[tail].max())} (excess {int(excess.max())})")
    errors = support.row_errors(output, expected_output)
    print(f"layer {LAYER} attention with prepared selection: {errors}")
    assert errors["median"] < 0.03 and errors["relative_l2"] < 0.03, errors
    for index, (single, _) in enumerate(decoded):
        row = rows - len(decoded) + index
        same = len(set(single.records[0, :2048].tolist()) & set(selection.records[row, :2048].tolist()))
        print(f"decode row {row}: {same} of 2048 records shared with prefill")
        assert int(single.lengths[0]) == 2048


@torch.no_grad()
def test_wrong_selections_are_rejected(manifest, canonical):
    """The most recent 2048 positions, and unrotated index vectors, fail the ranking comparison."""
    from ignition.recipes.glm53_full.attention import Selection

    source, _, selected, rank = canonical
    rows, device = source.shape[0], source.device
    start = (torch.arange(rows, device=device) - 2047).clamp(min=0)
    recent = start[:, None] + torch.arange(2048, device=device)[None]
    lengths = torch.minimum(torch.arange(1, rows + 1, device=device), torch.tensor(2048, device=device))
    recent = torch.where(torch.arange(2048, device=device)[None] < lengths[:, None], recent, -1)
    _, _, inside, worst = overlap(Selection(recent.to(torch.int32), lengths.to(torch.int32)), selected, rank)
    fraction = inside[2048:].double() / 2048
    print(f"most recent 2048: canonical top-2048 kept minimum {fraction.min():.5f}, deepest rank {int(worst.max())}")
    assert fraction.min() < 0.97 and int(worst[2048:].max()) - 2047 >= 256
    selection, _, _ = prepared_selection(manifest, source, rotate=False)
    _, _, inside, worst = overlap(selection, selected, rank)
    fraction = inside[2048:].double() / 2048
    print(f"unrotated: canonical top-2048 kept minimum {fraction.min():.5f}, deepest rank {int(worst.max())}")
    assert fraction.min() < 0.97 and int(worst[2048:].max()) - 2047 >= 256


@torch.no_grad()
def test_every_rank_selects_the_same_records(manifest, canonical):
    """Replicated index weights and inputs select the same ordered logical positions."""
    source = canonical[0]
    first, _, _ = prepared_selection(manifest, source)
    other, _, _ = prepared_selection(manifest, source, shard=TensorParallelShard(5, 8))
    assert torch.equal(first.lengths, other.lengths)
    assert torch.equal(first.records, other.records)
