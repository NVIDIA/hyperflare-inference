# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint attention preserves selected sets and output/state at matched ordering."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.fused_mla import FusedMLAPreparation, PreparedFusedMLA
from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.attention import AttentionScratch, PreparedAttention, Rotary
from ignition.recipes.deepseek41.config import ModelConfig
from ignition.recipes.deepseek41.pool import ContinuationPool
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


class ReferenceSelectionOrder:
    """Check the real reference selection, then match its unordered result to replay."""

    def __init__(self, selector, recorded):
        self.selector, self.recorded = selector, recorded

    def __call__(self, *args, **kwargs):
        result = self.selector(*args, **kwargs)
        ids = result[0] if isinstance(result, tuple) else result
        torch.testing.assert_close(ids.sort(-1).values, self.recorded.sort(-1).values, rtol=0, atol=0)
        # Atomic top-K scatter has no order contract. Couple only that ordering;
        # the reference must still compute exactly the same token set itself.
        permutation = ids.argsort(-1).gather(1, self.recorded.argsort(-1).argsort(-1))
        ordered = ids.gather(1, permutation)
        return (ordered, *result[1:]) if isinstance(result, tuple) else ordered


@pytest.fixture(scope="module")
def preparation(tmp_path_factory):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    snapshot, library = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT"), os.environ.get("FLARE_FLASHMLA_LIBRARY")
    if not snapshot or not library:
        pytest.skip("requires the pinned checkpoint and FlashMLA library")
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    config = ModelConfig.load(manifest.metadata_path)
    rotary = Rotary(config, torch.device("cuda:0"))
    layout = FusedMLAPreparation.create((rotary.frequencies[False], rotary.frequencies[True]), 12288,
        kernel_cache=tmp_path_factory.mktemp("fused-mla"), library=library,
        deepselect_library=os.environ.get("FLARE_DEEPSELECT_LIBRARY"))
    return manifest, config, rotary, layout


def test_prefill_declarations_share_full_context_operands(preparation):
    _, _, _, layout = preparation
    metadata = Metadata.allocate(4, 1, 512, "cuda:0")
    query = torch.zeros((4, 64, 512), device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros(4, device="cuda", dtype=torch.int32)
    sink = torch.zeros(64, device="cuda", dtype=torch.float32)
    window = torch.zeros((1, 128, 528), device="cuda", dtype=torch.uint8)
    compressed = torch.zeros((1, 128, 288), device="cuda", dtype=torch.uint8)
    indices = torch.full((4, 128), -1, device="cuda", dtype=torch.int32)
    selected = torch.full((4, 512), -1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(4, device="cuda", dtype=torch.int32)
    kv_bytes = (512 * 128 + 128) * 512 * 2
    measured = []
    for shared in (False, True):
        examples = {} if shared else None
        with RecordedPreparationSession(device="cuda:0", checkpoint="declaration-storage",
                workload_schema="four-prefill-layers", numerical_policy="storage-only",
                compile_workers=1, autotune_effort="off") as session:
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            operators = [PreparedFusedMLA(query, positions, sink, window, indices, lengths, metadata,
                preparation=layout, rotary=1, mode="prefill", compressed=compressed,
                selected=selected, selected_lengths=lengths, session=session,
                name=f"attention.{index}", examples=examples) for index in range(4)]
            torch.cuda.synchronize()
            measured.append(torch.cuda.memory_allocated() - before)
            assert len(operators) == 4
    assert measured[0] >= 4 * kv_bytes
    assert kv_bytes <= measured[1] < kv_bytes + 1024 * 1024
    print(f"prefill declaration bytes: independent={measured[0]}, shared={measured[1]}")


@pytest.mark.parametrize("layer,draft,consumer", [(0, False, None), (2, False, None),
                         (20, False, None), (20, False, 24), (0, True, None)],
                         ids=["window", "ratio-two", "ratio-one", "hierarchical", "draft"])
@pytest.mark.parametrize("overlap", [False, True], ids=["serial", "overlap"])
def test_checkpoint_attention_compiled_state_matches_uncompiled(preparation, layer, draft, consumer, overlap):
    manifest, config, rotary, layout = preparation
    if consumer is not None and layout.deepselect_library is None:
        pytest.skip("hierarchical selection requires the pinned DeepSelect library")
    layers = (layer,) if consumer is None else (layer, consumer)
    prefixes = tuple(f"{'mtp' if draft else 'layers'}.{index}.attn." for index in layers)
    with DeepSeekWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
            prefixes=prefixes, dspark=draft, fused_attention=layout) as loader:
        weights = {index: loader.load_attention(index, draft=draft) for index in layers}
    cases = ((8, 1, 8, "draft"), (64, 8, 8, "draft"), (192, 1, 192, "draft")) if draft else (
        (1, 1, 1, "decode"), (17, 1, 17, "verify"), (64, 8, 8, "verify"), (257, 1, 257, "prefill"))
    if not draft:
        cases += ((8192, 1, 8192, "prefill"),)
    pool = ContinuationPool.allocate(config, 12, 8 * 67 + 1, "cuda:0", dspark=draft,
                                     window_record_planes=(512, 16))
    arena = ScratchArena("cuda:0", overlap=overlap)
    catalog, graphs, functions, examples = [], [], [], {}
    generator = torch.Generator(device="cuda").manual_seed(1029 + layer)
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint="deepseek41-fused-attention",
                workload_schema="decode-verify-prefill-draft", numerical_policy="compiled-state-bitwise",
                compile_workers=2, autotune_effort="off") as session:
            session.register_scratch(arena)
            for tokens, batch, width, mode in cases:
                metadata = Metadata.allocate(tokens, batch, 67, "cuda:0", max_query_tokens=width)
                scratch = AttentionScratch.allocate(tokens, 1, "cuda:0")
                source = torch.randn((tokens, 5120), device="cuda", dtype=torch.bfloat16, generator=generator) * .05
                selected = torch.full_like(scratch.selected, -1)
                candidates = torch.full_like(scratch.candidates, -1)
                candidate_counts = torch.zeros_like(scratch.candidate_counts)
                attentions = tuple(PreparedAttention(index, weights[index], source, scratch, metadata, pool, rotary,
                    mode=mode, world=1, session=session, name=f"attention.{index}.{mode}.{tokens}",
                    workspace=arena, examples=examples)
                    for index in layers)
                output = torch.empty_like(source)
                accepted = torch.zeros(batch, device="cuda", dtype=torch.int64)
                selections = tuple(torch.empty_like(selected) for _ in attentions)

                def forward(attentions=attentions, source=source, metadata=metadata, selected=selected,
                            candidates=candidates, candidate_counts=candidate_counts, output=output,
                            accepted=accepted, mode=mode, selections=selections):
                    result, selection = source, (selected, candidates, candidate_counts)
                    for attention, recorded in zip(attentions, selections):
                        result, records, selection = attention(result, metadata, selection)
                        recorded.copy_(selection[0])
                        if mode == "verify":
                            attention.commit(accepted, records, metadata)
                    output.copy_(result)

                catalog.append((metadata, source, output, accepted, forward, attentions, selections))
            session.tune()
            torch.cuda.synchronize()
            examples.clear()
            arena.release_preparation()
            for (tokens, batch, width, mode), operands in zip(cases, catalog):
                metadata, source, output, accepted, forward, _, _ = operands
                starts = torch.full((batch,), 126, device="cuda", dtype=torch.int64)
                lengths = torch.full((batch,), width, device="cuda", dtype=torch.int32)
                rows = torch.arange(1, batch + 1, device="cuda", dtype=torch.int64)
                pages = torch.arange(1, batch * 67 + 1, device="cuda", dtype=torch.int32).view(batch, 67)
                metadata.set(starts, lengths, rows, pages, torch.zeros((batch, width), device="cuda", dtype=torch.int64),
                             torch.tensor([batch], device="cuda", dtype=torch.int32))
                pool.clear_for_preparation()
                function = compile_forward(forward)
                function()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    function()
                graphs.append(graph)
                functions.append(function)
            session.freeze()
            for step in range(2):
                for case, operands, graph in zip(cases, catalog, graphs):
                    tokens, batch, width, mode = case
                    metadata, source, output, accepted, forward, attentions, selections = operands
                    source.neg_()
                    starts = torch.full((batch,), 126 + 2 * step, device="cuda", dtype=torch.int64)
                    lengths = torch.tensor([width if step == 0 or i == 0 else width // 2 for i in range(batch)],
                                           device="cuda", dtype=torch.int32)
                    rows = torch.arange(1, batch + 1, device="cuda", dtype=torch.int64).flip(0)
                    pages = torch.arange(1, batch * 67 + 1, device="cuda", dtype=torch.int32).view(batch, 67).flip(0)
                    metadata.set(starts, lengths, rows, pages, torch.zeros((batch, width), device="cuda", dtype=torch.int64),
                                 torch.tensor([batch], device="cuda", dtype=torch.int32))
                    accepted.copy_(lengths if step == 0 else lengths // 2)
                    pool.clear_for_preparation()
                    output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                    assert torch.isfinite(output).all()
                    actual = output.clone()
                    actual_state = pool.backing.clone()
                    actual_pages = {name: tensor.clone() for name, tensor in pool.pages.items()}
                    recorded_selections = tuple(ids.clone() for ids in selections)
                    original_selectors = tuple(attention.selectors for attention in attentions)
                    try:
                        for attention, recorded in zip(attentions, recorded_selections):
                            selectors, first = [], 0
                            for selector in attention.selectors:
                                count = selector.scoring.plan.query.rows
                                selectors.append(ReferenceSelectionOrder(selector, recorded[first:first + count]))
                                first += count
                            attention.selectors = selectors
                        pool.clear_for_preparation()
                        forward()
                    finally:
                        for attention, selectors in zip(attentions, original_selectors):
                            attention.selectors = selectors
                    torch.testing.assert_close(actual, output, rtol=0, atol=0)
                    torch.testing.assert_close(actual_state, pool.backing, rtol=0, atol=0)
                    for name, tensor in pool.pages.items():
                        torch.testing.assert_close(actual_pages[name], tensor, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
