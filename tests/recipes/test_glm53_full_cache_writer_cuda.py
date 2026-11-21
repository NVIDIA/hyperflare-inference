# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare captured GLM query preparation and final cache slots with independent byte encodings."""

import os
from pathlib import Path

import pytest
import torch
from flashinfer.triton.norm import rms_norm

from ignition.components.parallel import TensorParallelShard
from ignition.components.rotary import PreparedRotaryAngles
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53_full.attention import AttentionMetadata, PreparedAttention, rotary_frequencies
from ignition.recipes.glm53_full.config import ModelConfig
from ignition.recipes.glm53_full.indexer import PreparedIndexer
from ignition.recipes.glm53_full.weights import GlmWeightLoader
from ignition.reference.glm_moe_dsa_cache import encode_latent, quantize_index

from . import glm53_full_support as support


def assert_latent_records(actual, latent, rotary):
    expected = encode_latent(latent, rotary)
    assert torch.equal(actual[:, 512:], expected[:, 512:])
    scales = expected[:, 512:528].contiguous().view(torch.float32)
    lower = upper = scales.reciprocal()
    for _ in range(2):
        lower = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
        upper = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    values = latent.float().reshape(-1, 4, 128)
    ends = torch.stack((values * lower[..., None], values * upper[..., None]))
    ends = ends.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    saved = actual[:, :512].contiguous().view(torch.float8_e4m3fn).float().reshape_as(values)
    assert ((saved >= ends.amin(0)) & (saved <= ends.amax(0))).all()


@support.requires_reference
@pytest.mark.parametrize("cache_only", [False, True])
@torch.inference_mode()
def test_captured_projection_writes_only_assigned_records(cache_only):
    device = torch.device("cuda:0")
    manifest = inspect_indexed_checkpoint(Path(support.SNAPSHOT))
    cfg = ModelConfig.load(manifest.snapshot)
    with GlmWeightLoader(manifest, device, shard=TensorParallelShard(0, 8),
                         prefixes=("model.layers.1.self_attn.",)) as loader:
        weights = loader.load_attention(1)
    rows = 4
    source = support.layer_input(support.fixture("short"), 1, device)[:rows].bfloat16().contiguous()
    source[0].zero_()
    meta = AttentionMetadata.allocate(rows, 1, 1, device)
    meta.positions.copy_(torch.tensor([0, 1, 2, 3],device=device))
    rot = PreparedRotaryAngles(meta.positions, rotary_frequencies(cfg.rope_theta, device))
    latent = torch.full((4, 64, 656), 173, dtype=torch.uint8, device=device)
    index = torch.full((4, 64 * 132), 173, dtype=torch.uint8, device=device)
    arena = ScratchArena(device, overlap=False)
    with RecordedPreparationSession(device=device, checkpoint="glm53-full-test",
            workload_schema="glm53-captured-cache", numerical_policy="default") as session:
        session.register_scratch(arena)
        attn = PreparedAttention(weights, source, meta, latent.clone(), rot, heads=8,
                selects=True, eps=1e-6, softmax_scale=256 ** -0.5, mode="decode",
                session=session, name="attention", workspace=arena, cache_only=cache_only)
        indexer = PreparedIndexer(weights, source[:, :2048], source[:, :128], source[:, :32].float(),
                meta, index.clone(), rot, mode="decode", session=session, name="indexer",
                workspace=arena, cache_only=True)
        attn.functional(session=session)
        session.tune()
        arena.release_preparation()
        attn.rebind_pages(latent)
        indexer.rebind_pages(index)
        angles = rot()
        projected = attn.projections["input_proj"](source)
        reference_latent = torch.empty((rows, 512), dtype=torch.bfloat16, device=device)
        reference_query = torch.empty((rows, 2048), dtype=torch.bfloat16, device=device)
        rms_norm(projected[:, :2048], weights["q_a_layernorm.weight"], reference_query, 1e-6,
                 round_before_weight=True)
        rms_norm(projected[:, 2048:2560], weights["kv_a_layernorm.weight"], reference_latent, 1e-6,
                 round_before_weight=True)
        rotary = projected[:, 2560:2624].clone()
        rot.apply(rotary, angles)
        rotated = support.reference_index_key(projected[:, 2624:], weights)
        rot.apply(rotated, angles)
        reference_keys, reference_scales = quantize_index(rotated.cpu(), query=False)
        query_out = torch.empty_like(reference_query)

        def write():
            if cache_only:
                key = attn.store(source, meta, angles)
            else:
                prepared = attn.project(source, meta, angles)
                query_out.copy_(prepared.query_residual)
                key = prepared.index_key
            indexer.store(key, meta, angles)

        write()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write()
        for slots in ([63, 64, -1, 255], [-1] * rows, [128, -1, 0, 1], [63, 64, -1, 255]):
            meta.token_slots.copy_(torch.tensor(slots,device=device))
            latent.fill_(173)
            index.fill_(173)
            graph.replay()
            if not cache_only:
                assert torch.equal(query_out, reference_query)
            actual = latent.cpu().view(-1, 656)
            physical = torch.tensor(slots)
            keep = physical >= 0
            target = physical[keep]
            untouched = torch.ones(len(actual), dtype=torch.bool)
            untouched[target] = False
            assert (actual[untouched] == 173).all()
            expected_index = torch.full_like(index, 173, device="cpu")
            if keep.any():
                assert_latent_records(actual[target], reference_latent.cpu()[keep], rotary.cpu()[keep])
                for row, slot in enumerate(slots):
                    if slot >= 0:
                        page, offset = divmod(slot, 64)
                        expected_index[page, offset * 128:(offset + 1) * 128] = reference_keys[row]
                        expected_index[page, 8192:].view(torch.float32)[offset] = reference_scales[row]
            assert torch.equal(index.cpu(), expected_index)
        session.freeze()


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires FLARE_RUN_CUDA_TESTS=1")
@torch.inference_mode()
def test_captured_writers_preserve_bf16_and_fp8_rounding_boundaries():
    from ignition.components.glm_mla_cache import store_glm_index_key, store_glm_latent

    device = torch.device("cuda:0")
    rows = 4
    projected = torch.ones((rows, 2752), device=device, dtype=torch.bfloat16)
    projected[0].zero_()
    for width, offset in ((2048, 0), (512, 2048)):
        spreading = (torch.arange(width, device=device) % 15 - 7).float() / 8
        projected[2:, offset:offset + width] = spreading.bfloat16()
    latent_weight = torch.full((512,), 1.0625, device=device, dtype=torch.bfloat16)
    latent_weight[::128] = 448
    query_weight = (1.25 + (torch.arange(2048, device=device) % 5) / 16).bfloat16()
    key_weight = torch.ones(128, device=device, dtype=torch.bfloat16)
    bias = torch.zeros_like(key_weight)
    bias[:2] = torch.tensor([2.59375, -0.486328125], device=device, dtype=bias.dtype)
    angles = torch.zeros((rows, 64), device=device, dtype=torch.float32)
    angles[:, ::2] = 1
    angles[2::2, ::2] = 0
    angles[2::2, 1::2] = 1
    slots = torch.tensor([0, 63, 64, -1], device=device)
    cache = torch.full((2, 64, 656), 173, device=device, dtype=torch.uint8)
    index = torch.full((2, 64 * 132), 173, device=device, dtype=torch.uint8)
    query = torch.empty((rows, 2048), device=device, dtype=torch.bfloat16)

    def write():
        store_glm_latent(projected, query_weight, latent_weight, angles, slots, cache, 1e-6, query=query)
        store_glm_index_key(projected[:, 2624:], key_weight, bias, angles, slots, index, 1e-6)

    write()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    cache.fill_(173)
    index.fill_(173)
    graph.replay()

    def reference(values, weight):
        values = values.cpu().double()
        normal = values * torch.rsqrt(values.square().mean(1, keepdim=True) + 1e-6)
        return (normal.bfloat16().double() * weight.cpu().double()).bfloat16(), (normal * weight.cpu().double()).bfloat16()

    expected_query, late = reference(projected[:, :2048], query_weight)
    assert (expected_query[2] != late[2]).any()
    assert torch.equal(query.cpu(), expected_query)
    latent, _ = reference(projected[:, 2048:2560], latent_weight)
    rotary = projected[:, 2560:2624].cpu().clone()
    rotary[2, ::2] = -rotary[2, ::2]
    records = cache.cpu().reshape(-1, 656)
    assert_latent_records(records[[0, 63, 64]], latent[:3], rotary[:3])
    midpoint = encode_latent(latent[1:2], rotary[1:2])[0]
    assert torch.equal(records[63], midpoint) and midpoint[1].view(torch.float8_e4m3fn).float().item() == 1.0
    assert (records[1:63] == 173).all() and (records[65:] == 173).all()
    # With a constant index input its centered part vanishes, leaving only the affine bias.
    oriented = bias.cpu().roll(64).repeat(3, 1)
    a, b = oriented[2, 64:66].clone()
    oriented[2, 64] = -b
    oriented[2, 65] = a
    keys, scales = quantize_index(oriented, query=False)
    assert keys[1, 65].item() == 0xeb and scales[1].view(torch.int32).item() == 1002288859
    expected = torch.full_like(index, 173, device="cpu")
    for row, slot in enumerate((0, 63, 64)):
        page, offset = divmod(slot, 64)
        expected[page, offset * 128:(offset + 1) * 128] = keys[row]
        expected[page, 8192:].view(torch.float32)[offset] = scales[row]
    assert torch.equal(index.cpu(), expected)
