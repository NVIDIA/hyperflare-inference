# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek's recipe router feeds fused target and draft expert execution."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
from ignition.components.megamoe_tiers import MegaMoeLayerLayout
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.decoder import PreparedLayer, Scratch
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.mark.parametrize("draft", [False, True], ids=["target", "draft"])
def test_recipe_fused_experts_match_canonical_hbm(tmp_path, megamoe_core_storage, draft):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    snapshot = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    rowmap = os.environ.get("FLARE_MEGAMOE_REFERENCE_ROWMAP")
    if not snapshot or not rowmap:
        pytest.skip("requires the lossless-CSF checkpoint and DerpSeek exact rowmap")
    if draft:
        canonical, top_k, host, hbm = 128, 3, (), tuple(range(128))
        prefix, cases = "mtp.0.ffn", ((1, "draft"), (8, "draft"), (64, "draft"), (192, "draft"))
    else:
        rows = json.loads(Path(rowmap).read_text())["layers"]["0"]
        canonical, top_k = 384, 6
        host, hbm = tuple(rows["host"]), tuple(rows["hot"] + rows["cache"])
        prefix, cases = "layers.0.ffn", ((1, "decode"), (8, "verify"), (64, "prefill"), (256, "prefill"))
    storage = megamoe_core_storage
    workspace = MegaMoeWorkspace(MegaMoeGeometry(len(host) + len(hbm), 256, top_k, 5120, 2304, 1),
                                 device="cuda:0", kernel_cache=tmp_path)
    reference_workspace = MegaMoeWorkspace(MegaMoeGeometry(canonical, 256, top_k, 5120, 2304, 1),
                                           device="cuda:0", kernel_cache=tmp_path)
    layout = MegaMoeLayerLayout(workspace, host, hbm, storage)
    reference_layout = MegaMoeLayerLayout(reference_workspace, (), tuple(range(canonical)), storage)
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    with DeepSeekWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
            prefixes=(prefix + ".",), dspark=draft, megamoe_layouts={prefix: layout}) as loader:
        weights, text = loader.load_ffn(0, draft=draft), loader.config.text
    with DeepSeekWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 1),
            prefixes=(prefix + ".",), dspark=draft, megamoe_layouts={prefix: reference_layout}) as loader:
        reference = loader.load_ffn(0, draft=draft).experts
    arena = ScratchArena("cuda:0", overlap=True)
    graphs, functions, catalog = [], [], []
    generator = torch.Generator(device="cuda").manual_seed(711)
    try:
        with RecordedPreparationSession(device="cuda:0", checkpoint="deepseek41-recipe-ffn",
                workload_schema="target-draft-routing", numerical_policy="reference-bitwise",
                compile_workers=2, autotune_effort="off") as session:
            session.register_scratch(arena)
            for rows, mode in cases:
                scratch = Scratch.allocate(rows, rows, 1, "cuda:0", experts=canonical, top_k=top_k)
                x = scratch.ffn_hc.normalized
                x.copy_(torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=generator) * .1)
                metadata = SimpleNamespace(positions=torch.arange(rows, device="cuda"),
                                          token_ids=torch.zeros(rows, device="cuda", dtype=torch.int64))
                layer = object.__new__(PreparedLayer)
                layer.has_image_mask = mode == "prefill" and weights.vision_correction is not None
                layer.image_mask = torch.zeros(rows, device="cuda", dtype=torch.bool) if layer.has_image_mask else None
                layer._prepare_experts(weights, scratch, text, mode=mode, session=session,
                                       name=f"ffn-{rows}", workspace=arena)
                assert layer.shared is None and layer.concurrent_shared is None
                actual, expected = torch.empty_like(x), torch.empty_like(x)
                ids = torch.empty((rows, top_k), device="cuda", dtype=torch.int32)
                ref = PreparedMegaMoe(x, scratch.router.ids, scratch.router.weights,
                    metadata.positions < 0, reference.routed, reference.shared,
                    workspace=reference.workspace, clamp=text["swiglu_limit"], session=session,
                    name=f"reference-{rows}")
                catalog.append((x, metadata, layer, actual, expected, ids, ref))
            session.tune()
            arena.release_preparation()
            for x, metadata, layer, actual, expected, ids, ref in catalog:
                def forward(x=x, metadata=metadata, layer=layer, actual=actual, expected=expected, ids=ids, ref=ref):
                    value, route = layer._run_experts(x, metadata)
                    actual.copy_(value)
                    ids.copy_(route.ids)
                    expected.copy_(ref(x, route.ids, route.weights, metadata.positions < 0))
                fn = compile_forward(forward)
                fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    fn()
                graphs.append(graph)
                functions.append(fn)
            session.freeze()
            for step in range(3):
                for (x, metadata, layer, actual, expected, ids, _), graph in zip(catalog, graphs):
                    x.neg_()
                    live = torch.arange(x.shape[0], device="cuda") % 3 != step
                    metadata.positions.copy_(torch.where(live, 17, -1))
                    metadata.token_ids.copy_(torch.where(live, 129280, 5))
                    tier = weights.experts.host_tier
                    before = None if tier is None else tier.control[16:].clone()
                    actual.fill_(float("nan"))
                    torch.cuda.synchronize()
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert allocations == torch.cuda.memory_stats()["allocation.all.allocated"]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    assert bool(((ids >= 0) & (ids < canonical)).all())
                    if tier is not None:
                        counts = torch.bincount(weights.experts.expert_map[ids[live].long()].long().flatten(),
                                                minlength=workspace.geometry.experts)
                        torch.testing.assert_close((tier.control[16:] - before).long(), counts, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        if weights.experts.host_tier is not None:
            for region in weights.experts.host_tier.regions:
                region.close()
