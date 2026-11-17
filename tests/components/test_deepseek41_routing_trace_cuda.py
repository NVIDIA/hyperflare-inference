# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical route histograms preserve a checkpoint expert block's captured output."""

import json
import os
from pathlib import Path

import pytest
import torch

from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
from ignition.components.megamoe_tiers import MegaMoeLayerLayout
from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.components.routing_trace import RoutingTrace
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.decoder import PreparedLayer, Scratch
from ignition.recipes.deepseek41.decoder_prefill import PreparedDecoderSuffix
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def test_checkpoint_routing_is_canonical_and_plain_graph_does_not_write_telemetry(tmp_path, megamoe_core_storage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    rowmap = os.environ.get("FLARE_MEGAMOE_REFERENCE_ROWMAP")
    if not snapshot or not rowmap:
        pytest.skip("requires the checkpoint and explicit canonical row map")
    placement = json.loads(Path(rowmap).read_text())["layers"]["0"]
    host, hbm = tuple(placement["host"]), tuple(placement["hot"] + placement["cache"])
    workspace = MegaMoeWorkspace(MegaMoeGeometry(len(host) + len(hbm), 8192, 6, 5120, 2304, 1),
                                 device="cuda:0", kernel_cache=tmp_path)
    layout = MegaMoeLayerLayout(workspace, host, hbm, megamoe_core_storage)
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    with DeepSeekWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
            prefixes=("layers.0.ffn.",), dspark=False, megamoe_layouts={"layers.0.ffn": layout}) as loader:
        weights, text = loader.load_ffn(0), loader.config.text
    output = torch.full((2, 40, 388), -17, device="cuda", dtype=torch.int64)
    frame = torch.zeros(1, device="cuda", dtype=torch.int32)
    sequence = torch.ones(1, device="cuda", dtype=torch.int64)
    trace = RoutingTrace(output, frame, sequence)
    arena = ScratchArena("cuda:0", overlap=True)
    graphs, functions, cases = [], [], []
    generator = torch.Generator(device="cuda").manual_seed(713)
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint="deepseek41-layer0-routing",
                workload_schema="canonical-routing-noninterference", numerical_policy="reference-bitwise",
                compile_workers=2, autotune_effort="off") as session:
            session.register_scratch(arena)
            for rows, mode in ((1, "decode"), (17, "verify"), (64, "verify"), (128, "prefill"), (8192, "prefill")):
                scratch = Scratch.allocate(rows, rows, 1, "cuda:0", experts=384, top_k=6)
                x = scratch.ffn_hc.normalized
                x.copy_(torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator) * .1)
                layer = object.__new__(PreparedLayer)
                layer.has_image_mask = False
                layer.image_mask = None
                layer._prepare_experts(weights, scratch, text, mode=mode, session=session,
                                       name=f"routes-{rows}", workspace=arena)
                traced = layer.with_routing_trace(trace, 0)
                assert layer.routing_counter is None and traced.router is layer.router
                lengths = torch.tensor([rows], device="cuda", dtype=torch.int32)
                plain_output, traced_output = torch.empty_like(x), torch.empty_like(x)
                plain_ids, traced_ids = (torch.empty((rows, 6), device="cuda", dtype=torch.int32) for _ in range(2))
                cases.append((x, lengths, layer, traced, plain_output, traced_output, plain_ids, traced_ids))
            session.tune()
            arena.release_preparation()
            for x, lengths, layer, traced, plain_output, traced_output, plain_ids, traced_ids in cases:
                pair = []
                for operation, destination, ids, enabled in ((layer, plain_output, plain_ids, False),
                                                             (traced, traced_output, traced_ids, True)):
                    def forward(x=x, lengths=lengths, operation=operation, destination=destination, ids=ids, enabled=enabled):
                        metadata = Metadata.pack(torch.zeros(1, device=x.device, dtype=torch.int64), lengths,
                            torch.zeros(1, device=x.device, dtype=torch.int64),
                            torch.zeros((1, 1), device=x.device, dtype=torch.int32),
                            torch.zeros((1, x.shape[0]), device=x.device, dtype=torch.int64),
                            torch.ones(1, device=x.device, dtype=torch.int32), token_capacity=x.shape[0])
                        if enabled:
                            trace.begin()
                        value, route = operation._run_experts(x, metadata)
                        destination.copy_(value)
                        ids.copy_(route.ids)
                    fn = compile_forward(forward)
                    fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with session.capture(), torch.cuda.graph(graph):
                        fn()
                    pair.append(graph)
                    functions.append(fn)
                graphs.append(pair)
            session.freeze()
            for step in range(3):
                frame.fill_(step % 2)
                sequence.fill_(step + 11)
                for (x, lengths, _, _, plain_output, traced_output, plain_ids, traced_ids), pair in zip(cases, graphs):
                    live = x.shape[0] if step == 0 else max(0, x.shape[0] - 3) if step == 1 else 0
                    lengths.fill_(live)
                    x.neg_()
                    before = output.clone()
                    torch.cuda.synchronize()
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    pair[0].replay()
                    torch.cuda.synchronize()
                    assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
                    torch.testing.assert_close(output, before, rtol=0, atol=0)
                    torch.cuda.synchronize()
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    pair[1].replay()
                    torch.cuda.synchronize()
                    assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
                    torch.testing.assert_close(plain_output, traced_output, rtol=0, atol=0)
                    torch.testing.assert_close(plain_ids, traced_ids, rtol=0, atol=0)
                    torch.testing.assert_close(output[1 - step % 2], before[1 - step % 2], rtol=0, atol=0)
                    assert output[step % 2, 0, :4].cpu().tolist() == [step + 11, live, 6, 0]
                    counts = torch.bincount(plain_ids[:live].long().flatten(), minlength=384)
                    torch.testing.assert_close(output[step % 2, 0, 4:], counts, rtol=0, atol=0)
                    assert torch.count_nonzero(output[step % 2, 1:, 1:]).item() == 0
    finally:
        torch.cuda.synchronize()
        for pair in graphs:
            for graph in pair:
                graph.reset()


@pytest.mark.parametrize("phase", ["packed_verification", "prefill_suffix"])
def test_real_metadata_packing_drives_layer_specific_valid_counts(phase):
    capacity, requests, width = (17, 4, 4) if phase == "packed_verification" else (257, 1, 257)
    lengths = torch.full((requests,), width, dtype=torch.int32, device="cuda")
    starts = torch.arange(requests, device="cuda", dtype=torch.int64) * 128
    states = torch.arange(requests, device="cuda", dtype=torch.int64)
    pages = torch.zeros((requests, 1), device="cuda", dtype=torch.int32)
    tokens = torch.zeros((requests, width), device="cuda", dtype=torch.int64)
    live_rows = torch.tensor([requests], device="cuda", dtype=torch.int32)
    metadata = Metadata.pack(starts, lengths, states, pages, tokens, live_rows, token_capacity=capacity)
    suffix = PreparedDecoderSuffix(metadata).functional() if phase == "prefill_suffix" else None
    output = torch.zeros((2, 40, 388), dtype=torch.int64, device="cuda")
    frame = torch.zeros(1, dtype=torch.int32, device="cuda")
    sequence = torch.ones(1, dtype=torch.int64, device="cuda")
    trace = RoutingTrace(output, frame, sequence)
    ids = (torch.arange(capacity * 6, device="cuda") % 384).reshape(capacity, 6).int()
    prefix_counter = trace.functional_counter(0, capacity, 6)
    suffix_counter = trace.functional_counter(20, 128, 6) if suffix is not None else None

    def forward():
        packed = Metadata.pack(starts, lengths, states, pages, tokens, live_rows, token_capacity=capacity)
        trace.begin()
        prefix_counter(ids, packed.live_counts[:1])
        if suffix is not None:
            retained = suffix.begin(packed)
            suffix_counter(ids[:128], retained.metadata.live_counts[:1])

    fn = compile_forward(forward)
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    try:
        counts = ([4, 0, 2, 1], [0, 3, 0, 1], [0, 0, 0, 0]) if requests == 4 else ([241], [7], [0])
        for step, values in enumerate(counts):
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
            frame.fill_(step % 2)
            sequence.fill_(step + 21)
            graph.replay()
            torch.cuda.synchronize()
            expected_layers = [(0, sum(values))]
            if suffix is not None:
                expected_layers.append((20, min(sum(values), 128)))
            for layer, count in expected_layers:
                assert output[step % 2, layer, :4].cpu().tolist() == [step + 21, count, 6, 0]
                expected = torch.bincount(ids[:count].long().flatten(), minlength=384)
                torch.testing.assert_close(output[step % 2, layer, 4:], expected, rtol=0, atol=0)
    finally:
        graph.reset()
