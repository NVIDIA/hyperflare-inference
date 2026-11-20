# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare GLM 5.3 tensor-parallel decoder layers and full-stack outputs with Transformers.

Each rank is one process on one GPU; ranks sum partial updates through b12x
all-reduces and gather vocabulary shards. Two comparisons use the fixtures of
`glm53_full_support`:

- Layer-isolated: every layer receives the reference output of the preceding
  layer, so its residual update is compared without the error that earlier
  layers accumulate. The test bounds relative errors on the two fixtures.
- Whole stack: the prepared forward runs from token IDs to logits. BF16
  arithmetic alone moves late hidden states far from an FP32 reference: the
  Transformers forward in BF16 differs from its FP32 run by 0.21 in the final
  hidden states of the `long` fixture and changes 299 of its 2722 greedy
  tokens, 29 of them at a top-two logit margin of at least 1.0. Token equality
  with the FP32 reference is checked on the final position. Across the sequence,
  the test bounds the hidden-state error and the number of changed tokens by
  twice those measurements. It does not bound the largest margin at which a
  token changes, because that value is not repeatable: the prepared stack is
  not bitwise deterministic, and five runs of this test on the same weights
  disagreed with each other on 298 to 331 tokens of the `long` fixture, 27 to
  42 of them at a margin of at least 1.0 and one at a margin of 8.2.
"""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from . import glm53_full_support as support

# Error of the layer-isolated residual update when the Transformers forward itself runs in BF16 with FP8
# latent records, against its FP32 run on the same fixture: relative L2 over all rows, and the median
# per-row relative error. Bounds are twice these. Layer 0 has no recorded input in that measurement and
# uses the values of layer 1, which has the same composition.
UPDATE_FLOOR = {
    "short": {0: (0.0070, 0.0081), 2: (0.0070, 0.0101), 3: (0.0059, 0.0171), 6: (0.0048, 0.0164),
              40: (0.0887, 0.0224), 77: (0.0033, 0.0116)},
    "long": {0: (0.0076, 0.0075), 2: (0.0088, 0.0090), 3: (0.0151, 0.0148), 6: (0.0148, 0.0154),
             40: (0.0368, 0.0202), 77: (0.0088, 0.0093)}}
# The same BF16 run's whole-forward differences from the FP32 run on the `long` fixture.
HIDDEN_FLOOR, CHANGED_TOKENS, CHANGED_DECISIVE_TOKENS = 0.21, 299 / 2722, 29 / 1516
# Rows of the decode forward that verifies consecutive positions; the trailing rows hold no token.
VERIFY_ROWS = 8
# Prefill rows of each fixture's forward. `short` (24 tokens) leaves rows without a token; `long`
# (2722 tokens) takes two chunks, the second continuing the first's context with rows left over.
CAPACITY = {"short": 32, "long": 1408}


def _check_causal_suffix(decoder, place_rows, pool, runtime, session):
    """One pending token has the same logits with zero or two later verification tokens."""
    metadata = place_rows(3)
    pages = [*pool.pages.values(), pool.backing]
    snapshot = [page.clone() for page in pages]
    tokens = metadata.token_ids.clone()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=tokens.device)
    stream.wait_stream(torch.cuda.current_stream(tokens.device))
    with runtime.capture(stream=stream), session.capture(), torch.cuda.graph(graph, stream=stream):
        _, logits, _, _ = decoder(metadata)
    torch.cuda.current_stream(tokens.device).wait_stream(stream)

    def replay(inputs):
        for page, saved in zip(pages, snapshot, strict=True):
            page.copy_(saved)
        metadata.token_ids.copy_(inputs)
        graph.replay()
        return logits[0].clone()

    expected = replay(tokens)
    assert bool(torch.isfinite(expected).all())
    assert expected.numel() == 154880
    expected_bits = expected.view(torch.uint8)
    assert torch.equal(replay(tokens).view(torch.uint8), expected_bits)
    changed = tokens.clone()
    changed[1:3] = tokens.new_tensor([7856, 1234])
    assert torch.equal(replay(changed).view(torch.uint8), expected_bits)
    place_rows(1)
    assert torch.equal(replay(metadata.token_ids.clone()).view(torch.uint8), expected_bits)
    place_rows(3)
    changed[0] = (tokens[0] + 1) % 154880
    assert not torch.equal(replay(changed), expected)
    for page, saved in zip(pages, snapshot, strict=True):
        page.copy_(saved)
    metadata.token_ids.copy_(tokens)


def _rank(rank, world, rendezvous, snapshot, reference, layers, sequences, output):
    from ignition.components.paged_batch import Metadata
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation, create_all_gather_runtime, create_all_reduce_runtime)
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53_full.attention import PAGE_TOKENS
    from ignition.recipes.glm53_full.decoder import PreparedDecoder
    from ignition.recipes.glm53_full.pool import ContinuationPool
    from ignition.recipes.glm53_full.weights import ModelWeights

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    def progress(message):
        if rank == 0:
            print(f"TP{world}: {message}", flush=True)

    fixtures = {name: torch.load(Path(reference) / f"{name}.pt", mmap=True) for name in sequences}
    lengths = {name: fixture["token_ids"].numel() for name, fixture in fixtures.items()}
    pages = -(-max(lengths.values()) // PAGE_TOKENS)
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    weights = ModelWeights.load(manifest, device, shard=TensorParallelShard(rank, world), progress=progress,
                                layers=layers)
    config, count = weights.config, len(weights.layers)
    complete = count == config.layers
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world,
                            timeout=timedelta(minutes=30))
    group = dist.group.WORLD
    coordinator = CollectivePreparation(group, timeout=timedelta(minutes=30))
    runtime = create_all_reduce_runtime(group, device, max_tokens=max(CAPACITY.values()), hidden_size=6144)
    vocabulary_runtime = create_all_gather_runtime(group, device, max_rows=max(CAPACITY.values()),
                                                   columns=weights.head.shape[0] * world)
    pool = ContinuationPool.for_model(config, 2, pages, device)
    arena = ScratchArena(device, overlap=False)
    results = {}
    with torch.no_grad(), RecordedPreparationSession(device=device, checkpoint="glm53-full-test",
            workload_schema="glm53-full-stack-tp-v1", numerical_policy="default", coordinator=coordinator) as session:
        session.register_scratch(arena)
        decoders, metadata = {}, {}
        for name, rows in (*((name, CAPACITY[name]) for name in sequences), ("decode", 1), ("verify", VERIFY_ROWS)):
            mode = "decode" if name in ("decode", "verify") else "prefill"
            # Every row is sampled, so greedy tokens compare over the whole sequence.
            metadata[name] = Metadata.allocate(rows, 1, pages, device, samples=rows)
            decoders[name] = PreparedDecoder(weights, metadata[name], pool, mode=mode, runtime=runtime,
                vocabulary_runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name,
                workspace=arena)
        progress("tuning")
        session.tune()
        arena.release_preparation()

        def place(name, first, last, token_ids):
            """Pack positions [first, last) of the sequence; rows beyond them hold no token."""
            target, count = metadata[name], last - first
            samples = target.sample_indices.numel()
            target.token_ids.zero_()
            target.token_ids[:count].copy_(token_ids[first:last].to(device))
            target.positions.fill_(-1)
            target.positions[:count].copy_(torch.arange(first, last, device=device))
            target.requests.zero_()
            target.page_table[0].copy_(torch.arange(pages, dtype=torch.int32, device=device))
            target.live_counts.copy_(torch.tensor([count, 1], dtype=torch.int32, device=device))
            target.sample_indices.copy_(torch.arange(samples, device=device).clamp(max=count - 1))
            target.starts.fill_(first)
            target.lengths.fill_(count)
            target.query_start_loc.copy_(torch.tensor([0, count], dtype=torch.int32, device=device))
            target.state_rows.fill_(1)
            return target

        def padded(rows, capacity):
            result = torch.zeros((capacity, rows.shape[1]), dtype=torch.bfloat16, device=device)
            result[:rows.shape[0]].copy_(rows)
            return result

        def isolated(name, layer, hidden, selection):
            from ignition.recipes.glm53_full.decoder import attention_metadata

            decoder, target = decoders[name], metadata[name]
            angles = decoder.angles(target.positions)
            return decoder.layers[layer](hidden, attention_metadata(target), angles, selection)

        for name, fixture in fixtures.items():
            rows, token_ids, capacity = lengths[name], fixture["token_ids"], CAPACITY[name]
            # Chunks continue one context; the last chunk leaves rows without a token.
            chunks = [(first, min(first + capacity, rows)) for first in range(0, rows, capacity)]
            reference_layers = fixture["layers"]
            record = results[name] = {"rows": rows, "chunks": chunks, "isolated": {}, "stack": {}}
            progress(f"{name}: layer-isolated updates")
            pool.clear_for_preparation()
            embedded = weights.embedding[token_ids.to(device)]
            selections = {}
            control_layer = min(40, count - 1)
            for layer in range(count):
                exact = embedded.float() if layer == 0 else reference_layers[layer - 1].to(device)
                expected = reference_layers[layer].to(device) - exact
                updates = []
                for first, last in chunks:
                    place(name, first, last, token_ids)
                    source = padded(exact[first:last], capacity)
                    hidden, selections[first] = isolated(name, layer, source, selections.get(first))
                    updates.append((hidden.float() - source.float())[:last - first])
                record["isolated"][layer] = support.row_errors(torch.cat(updates), expected)
                if layer == control_layer:
                    # Control: a wrong output projection must move this layer's update past its bound.
                    projection = weights.layers[layer].attention["o_proj"]
                    projection = projection if isinstance(projection, torch.Tensor) else projection.weight.values
                    saved = projection.clone()
                    generator = torch.Generator(device=device).manual_seed(rank)
                    projection.mul_(1 + 0.5 * torch.randn(projection.shape, device=device, generator=generator))
                    perturbed, _ = isolated(name, layer, source, selections[first])
                    projection.copy_(saved)
                    record["control"] = {"layer": layer, **support.row_errors(
                        (perturbed.float() - source.float())[:last - first], expected[first:last])}
            progress(f"{name}: whole stack")
            pool.clear_for_preparation()
            observed = {}

            def observe(layer, hidden, selection):
                if layer in UPDATE_FLOOR[name] or layer == count - 1:
                    observed.setdefault(layer, []).append(hidden[:last - first].float())

            tokens, hidden = [], []
            for first, last in chunks:
                normalized, logits, _, _ = decoders[name](place(name, first, last, token_ids), observe=observe)
                tokens.append(logits[:last - first].float().argmax(1).cpu())
                hidden.append(normalized[:last - first].float())
            for layer, parts in observed.items():
                record["stack"][layer] = support.row_errors(torch.cat(parts), reference_layers[layer].to(device))
            tokens, expected_tokens, margins = torch.cat(tokens), fixture["next_tokens"], fixture["margins"]
            changed = tokens != expected_tokens
            record.update(final_token=int(tokens[-1]), expected_final_token=int(expected_tokens[-1]),
                changed=int(changed.sum()), decisive_rows=int((margins >= 1.0).sum()),
                changed_decisive=int((changed & (margins >= 1.0)).sum()),
                largest_changed_margin=float(margins[changed].max()) if changed.any() else 0.0)
            if complete:
                record["hidden"] = support.row_errors(torch.cat(hidden), fixture["hidden"].to(device))
            # Decode recomputes the trailing rows one at a time against the records the prefill wrote.
            decoded, alone = [], []
            for row in range(rows - 3, rows):
                normalized, logits, _, _ = decoders["decode"](place("decode", row, row + 1, token_ids))
                alone.append(normalized[:1].float().clone())
                decoded.append(dict(row=row, token=int(logits.float().argmax(1)), prefill_token=int(tokens[row]),
                    expected_token=int(expected_tokens[row]), margin=float(margins[row]),
                    hidden=support.row_errors(normalized, fixture["hidden"][row:row + 1].to(device))["relative_l2"]))
            record["decoded"] = decoded
            # One decode forward recomputes the same rows as consecutive positions of one request, the
            # layout of speculative verification, beside rows that hold no token.
            normalized, _, _, _ = decoders["verify"](place("verify", rows - 3, rows, token_ids))
            # Each row is compared with its own single-row decode and, as the control, with its neighbor's.
            record["verified"] = [dict(
                row=rows - 3 + index,
                own=support.row_errors(normalized[index:index + 1], alone[index])["relative_l2"],
                neighbor=support.row_errors(normalized[index:index + 1], alone[(index + 1) % 3])["relative_l2"])
                for index in range(3)]
            if complete:
                start = rows - 3
                pool.clear_for_preparation()
                for first in range(0, start, capacity):
                    decoders[name](place(name, first, min(first + capacity, start), token_ids))
                _check_causal_suffix(decoders["verify"],
                    lambda count: place("verify", start, start + count, token_ids), pool, runtime, session)
        session.freeze()
    dist.barrier(group=group)
    if rank == 0:
        Path(output).write_text(json.dumps(results))
    dist.destroy_process_group()


def _run(world, layers, sequences, tmp_path):
    assert torch.cuda.device_count() >= world, "assign one CUDA device per rank"
    output = tmp_path / "results.json"
    mp.spawn(_rank, args=(world, (tmp_path / "rendezvous").as_uri(), support.SNAPSHOT, support.REFERENCE, layers,
                          sequences, str(output)), nprocs=world, join=True)
    results = json.loads(output.read_text())
    for name, record in results.items():
        print(f"{name}: rows {record['rows']} chunks {record['chunks']}")
        for layer, errors in record["isolated"].items():
            print(f"  isolated layer {layer}: {errors}")
        for layer, errors in record["stack"].items():
            print(f"  stack layer {layer}: {errors}")
        print(f"  control: {record['control']}")
        print(f"  tokens: {({key: record[key] for key in ('final_token', 'expected_final_token', 'changed', 'decisive_rows', 'changed_decisive', 'largest_changed_margin')})}")
        print(f"  hidden: {record.get('hidden')}")
        for row in record["decoded"]:
            print(f"  decode: {row}")
    return results


def _check_verified(record):
    """Compare each verified row with its single-row decode and a whole-row substitution."""
    for row in record["verified"]:
        assert row["own"] < 2 * HIDDEN_FLOOR < row["neighbor"], row


def _check_isolated(name, record):
    floors = UPDATE_FLOOR[name]
    for layer, (relative, median) in floors.items():
        errors = record["isolated"].get(str(layer))
        if errors is not None:
            assert errors["relative_l2"] < 2 * relative and errors["median"] < 2 * median, (layer, errors)
    # The perturbed output projection changes every row's attention update; rows with the largest
    # updates dominate the relative L2 error, so the median decides.
    control = record["control"]
    assert control["median"] > 2 * floors[control["layer"]][1], control


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1" or not support.SNAPSHOT or not support.REFERENCE,
                    reason="requires two GPUs, FLARE_RUN_TP_CUDA_TESTS=1, and the GLM 5.3 snapshot and reference")
def test_leading_layers_match_reference_at_two_ranks(tmp_path):
    """Layers 0 to 3 cover an indexer layer, a dense MLP, selection reuse, and routed experts."""
    results = _run(2, 4, ("short",), tmp_path)
    _check_isolated("short", results["short"])
    _check_verified(results["short"])


@pytest.mark.skipif(os.environ.get("FLARE_RUN_FULL_MODEL_TP_CUDA_TESTS") != "1" or not support.SNAPSHOT
                    or not support.REFERENCE,
                    reason="requires eight 96 GB GPUs, FLARE_RUN_FULL_MODEL_TP_CUDA_TESTS=1, and the GLM 5.3 "
                           "snapshot and reference")
def test_text_stack_matches_reference_at_eight_ranks(tmp_path):
    results = _run(8, None, ("short", "long"), tmp_path)
    for name, record in results.items():
        _check_isolated(name, record)
        _check_verified(record)
        rows = record["rows"]
        assert record["final_token"] == record["expected_final_token"], record
        assert record["hidden"]["relative_l2"] < 2 * HIDDEN_FLOOR, record["hidden"]
        # Counts on the 24-token fixture are too small to bound by a rate.
        if rows >= 1024:
            assert record["changed"] < 2 * CHANGED_TOKENS * rows, record
            assert record["changed_decisive"] < 2 * CHANGED_DECISIVE_TOKENS * record["decisive_rows"], record
        for row in record["decoded"]:
            assert row["hidden"] < 2 * HIDDEN_FLOOR, row
            assert row["token"] in (row["prefill_token"], row["expected_token"]) or row["margin"] < 1.0, row
