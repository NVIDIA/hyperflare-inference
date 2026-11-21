# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling2 checkpoint vision numerics, padded rows, and CUDA graph replay."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.recipes.inkling2.vision import PreparedVision, VisionMetadata, VisionScratch, VisionWeights

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")


@pytest.mark.parametrize("hidden", [2304, 4608])
def test_vision_geometry(hidden):
    full = VisionScratch.allocate(256, "cpu", hidden=hidden)
    rows = full.rows(64)
    expected = VisionScratch.allocate(64, "cpu", hidden=hidden)
    for field in fields(rows):
        actual = getattr(rows, field.name)
        assert actual.shape == getattr(expected, field.name).shape
        assert actual.untyped_storage()._cdata == getattr(full, field.name).untyped_storage()._cdata
    metadata = VisionMetadata.allocate(64, 1, "cpu")
    assert rows.patches.shape == (64, 1200)
    assert rows.query.shape == (65, 16, 64)
    assert rows.output.shape == (16, hidden)
    assert metadata.position_ids.shape == (65, 2)
    assert metadata.interpolation_indices.shape == (64, 4)
    with pytest.raises(ValueError):
        full.rows(63)
    with pytest.raises(ValueError):
        VisionMetadata.allocate(64, 2, "cpu")


@pytest.mark.parametrize("capacity", [64, 256, 1024, 4096, 16384, 65536])
def test_packed_vision_capacity_layout(capacity):
    scratch = VisionScratch.allocate(capacity, "meta")
    metadata = VisionMetadata.allocate(capacity, 1, "meta")
    assert scratch.qkv.shape == (capacity + 1, 3072)
    for tensor in (scratch.query, scratch.key, scratch.value):
        assert tensor.shape == (capacity + 1, 16, 64)
        assert tensor.stride() == (3072, 64, 1)
    assert scratch.query.storage_offset() == 0
    assert scratch.key.storage_offset() == 1024
    assert scratch.value.storage_offset() == 2048
    assert scratch.output.shape == (capacity // 4, 4608)
    assert metadata.position_ids.shape == (capacity + 1, 2)


@pytest.fixture(scope="module", params=["small", "mini"])
def manifest(request):
    if request.param == "mini":
        path = os.environ.get("FLARE_INKLING2_MINI_SNAPSHOT")
        if path is None:
            pytest.skip("requires FLARE_INKLING2_MINI_SNAPSHOT")
        from ignition.preparation.checkpoint import inspect_indexed_checkpoint
        return inspect_indexed_checkpoint(Path(path))
    snapshot = Path(os.environ.get("FLARE_INKLING2_SNAPSHOT", "/data/models/Inkling-2-Small-NVFP4"))
    if not snapshot.is_dir():
        pytest.skip("Inkling2 checkpoint is unavailable")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    filenames = sorted({shard for name, shard in index.items() if name.startswith("model.vision_tower.")})
    tensors, shards = {}, []
    for filename in filenames:
        record, entries = _read_header(snapshot / filename, filename)
        shards.append(record)
        tensors.update(entries)
    return IndexedCheckpoint(snapshot, tensors, tuple(shards))


def _module(manifest):
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

    raw = json.loads((manifest.snapshot / "config.json").read_text())
    config = Inkling2VisionConfig.from_dict(raw["vision_config"], text_hidden_size=raw["text_config"]["hidden_size"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = Inkling2VisionModel(config)
    return module.eval().requires_grad_(False)


def _reference(manifest, weights, device):
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionRotaryEmbedding

    module = _module(manifest)
    weights.bind(module)
    module.rotary_emb = Inkling2VisionRotaryEmbedding(module.config).to(device)
    return module


def _weights(manifest, module, device):
    if "model.llm.embed.weight" in manifest.tensors:
        from ignition.components.parallel import TensorParallelShard
        from ignition.recipes.inkling2.mini_weights import MiniWeightLoader
        context = MiniWeightLoader(manifest, device, shard=TensorParallelShard(0, 1))
    else:
        context = DirectWeightLoader(manifest, device, prefixes=("model.vision_tower.",))
    with context as loader:
        return VisionWeights.load(module, loader)


def _input(width, height, variant):
    from PIL import Image
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

    pixels = bytes(component for y in range(height) for x in range(width)
                   for component in ((x * 7 + variant * 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
    return Inkling2ImageProcessor()(images=[Image.frombytes("RGB", (width, height), pixels)], return_tensors="pt")


def _upload(inputs, pixels, metadata):
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

    grid = inputs.image_grid_thw * torch.tensor([1, 2, 2])
    patches = inputs.pixel_values.reshape(-1, 2, 20, 2, 20, 3).transpose(2, 3).reshape(-1, 1200)
    count = patches.shape[0]
    pixels.fill_(float("nan"))
    pixels[:count].copy_(patches)
    metadata.position_ids.zero_()
    metadata.position_ids[1:count + 1].copy_(get_vision_position_ids(grid, 2) + 1)
    indices, weights = get_vision_interpolation_indices_and_weights(grid, 32, spatial_merge_size=2)
    metadata.interpolation_indices.zero_()
    metadata.interpolation_weights.zero_()
    metadata.interpolation_indices[:count].copy_(indices)
    metadata.interpolation_weights[:count].copy_(weights)
    metadata.cu_seqlens.copy_(torch.tensor([0, count + 1], dtype=torch.int32))
    metadata.num_patches.fill_(count)
    metadata.num_features.fill_(count // 4)
    return count


def _relative(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


@CUDA
@torch.no_grad()
def test_mini_vllm_image_features(manifest):
    if "model.llm.embed.weight" not in manifest.tensors:
        pytest.skip("requires the Mini checkpoint")
    path = os.environ.get("FLARE_INKLING2_MINI_VISION_REFERENCE")
    if path is None:
        pytest.skip("requires independent vLLM vision fixtures")
    from b12x.preparation import PreparationSession
    from PIL import Image
    from safetensors.torch import load_file
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tensors = load_file(path)
    module = _module(manifest)
    weights = _weights(manifest, module, device)
    reference = _reference(manifest, weights, device)
    scratch = VisionScratch.allocate(1024, device, hidden=2304)
    metadata = VisionMetadata.allocate(1024, 1, device)
    pixels = torch.zeros((1024, 1200), dtype=torch.float32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = PreparedVision(module, scratch, metadata, weights, session=session)
        prepared.functional(session=session)
        session.freeze()
        for key in sorted(name.removesuffix(".image") for name in tensors if name.endswith(".image")):
            picture = Image.fromarray(tensors[f"{key}.image"].numpy())
            inputs = Inkling2ImageProcessor()(images=[picture], return_tensors="pt")
            count = _upload(inputs, pixels, metadata)
            live = torch.arange(1024, device=device) < count
            actual = prepared(torch.where(live[:, None], pixels, 0).bfloat16(), metadata)[:count // 4]
            expected = tensors[f"{key}.features"].to(device)
            canonical = reference(inputs.pixel_values.to(device), inputs.image_grid_thw).pooler_output
            reference.config._attn_implementation = "sdpa"
            alternate = reference(inputs.pixel_values.to(device), inputs.image_grid_thw).pooler_output
            reference.config._attn_implementation = "eager"
            canonical_spread = _relative(alternate, canonical)
            reference_spread = _relative(expected, canonical)
            error = _relative(actual, expected)
            print(dict(image=key, relative_l2=error, canonical_backend_relative_l2=canonical_spread,
                       reference_backend_relative_l2=reference_spread))
            assert torch.isfinite(actual).all()
            assert _relative(actual, canonical) < canonical_spread + .03
            assert error < reference_spread + .03


def _check_packed_checkpoint_weights(manifest, weights):
    from safetensors import safe_open

    for attribute in ("weight", "bias"):
        packed = weights.parameters[f"layers.0.self_attn.qkv.{attribute}"]
        for role, actual in zip(("q", "k", "v"), packed.chunk(3, dim=0), strict=True):
            name = f"model.vision_tower.layers.0.self_attn.{role}_proj.{attribute}"
            if "model.llm.embed.weight" in manifest.tensors:
                name = name.replace("model.vision_tower.", "model.visual.backbone.")
            with safe_open(str(manifest.snapshot / manifest.tensors[name].shard), framework="pt", device="cpu") as reader:
                expected = reader.get_tensor(name)
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _check_stages(prepared, reference, inputs, metadata, pixels):
    count = int(metadata.num_patches.item())
    grid = inputs.image_grid_thw * torch.tensor([1, 2, 2])
    patch_index = torch.arange(1, count + 1, device=pixels.device)
    hidden = reference.embeddings(inputs.pixel_values.to(pixels.device, torch.bfloat16), grid, patch_index)
    position = (reference.embeddings.position_embedding(metadata.interpolation_indices)
                * metadata.interpolation_weights[:, :, None]).sum(1).to(torch.bfloat16)
    assert _relative(prepared.patch_embed(pixels.to(torch.bfloat16)) + position, hidden[1:]) < .03
    hidden = reference.pre_layernorm(hidden)
    rotary = reference.rotary_emb(hidden, metadata.position_ids)
    rotary_cache = torch.cat((rotary[0][:, ::2], rotary[1][:, ::2]), dim=-1)
    stages = []
    for index, (actual, expected) in enumerate(zip(prepared.module.layers, reference.layers, strict=True)):
        output = expected(hidden, cu_seqlens=metadata.cu_seqlens, position_embeddings=rotary)
        error = _relative(actual(hidden, cu_seqlens=metadata.cu_seqlens, position_embeddings=rotary_cache), output)
        stages.append((index, error))
        assert error < .03, stages
        hidden = output
    merged = hidden[1:].reshape(-1, 4096)
    assert _relative(prepared.module.merger(merged), reference.merger(merged)) < .03
    return max(error for _, error in stages)


@CUDA
@torch.no_grad()
@pytest.mark.parametrize("capacity", [64, 256])
@pytest.mark.parametrize("compiler", ["none", "trace", "aot"])
def test_checkpoint_vision_native_replay(manifest, capacity, compiler):
    from b12x.preparation import PreparationSession

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    module = _module(manifest)
    weights = _weights(manifest, module, device)
    assert sum(t.nbytes for t in weights.parameters.values()) == 617637888 + 8192 * module.config.text_hidden_size
    _check_packed_checkpoint_weights(manifest, weights)
    assert not any("_proj." in path and path.rsplit(".", 2)[-2] in ("q_proj", "k_proj", "v_proj")
                   for path in weights.parameters)
    reference = _reference(manifest, weights, device)
    for index in range(23):
        packed = weights.parameters[f"layers.{index}.self_attn.qkv.weight"]
        for role in ("q", "k", "v"):
            original = getattr(reference.layers[index].self_attn, f"{role}_proj").weight
            assert original.untyped_storage()._cdata == packed.untyped_storage()._cdata
    scratch = VisionScratch.allocate(capacity, device, hidden=module.config.text_hidden_size)
    metadata = VisionMetadata.allocate(capacity, 1, device)
    pixels = torch.zeros((capacity, 1200), dtype=torch.float32, device=device)
    with PreparationSession(device=device, autotune=True, compile_workers=0) as session:
        prepared = PreparedVision(module, scratch, metadata, weights, session=session)
        prepared.functional(session=session)
        session.freeze()

        def forward():
            patch_live = torch.arange(capacity, device=device) < metadata.num_patches
            source = torch.where(patch_live[:, None], pixels, 0).to(torch.bfloat16)
            scratch.output.copy_(prepared(source, metadata))
            return scratch.output

        if compiler == "trace":
            from ignition.preparation.compilation import compile_forward
            forward = compile_forward(forward)
        elif compiler == "aot":
            from ignition.preparation.aot import aot_forward
            owner = forward
            artifact = aot_forward(owner)
            forward = lambda: artifact(owner)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph},
            buffers={"pixels": pixels, "embeddings": scratch.output},
            resources=(prepared, weights, session, stream, forward)).start_worker()
        side = int(capacity ** .5) * 20
        measurements = []
        try:
            for variant, (width, height) in enumerate(((side, side), (41, 79), (1, 1), (80, 40))):
                inputs = _input(width, height, variant)
                count = _upload(inputs, pixels, metadata)
                expected = reference(inputs.pixel_values.to(device), inputs.image_grid_thw).pooler_output
                if variant == 0:
                    stage_error = _check_stages(prepared, reference, inputs, metadata, pixels)
                reference.config._attn_implementation = "sdpa"
                alternate = reference(inputs.pixel_values.to(device), inputs.image_grid_thw).pooler_output
                reference.config._attn_implementation = "eager"
                spread = _relative(alternate, expected)
                replay_expected = forward().clone()
                scratch.output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                ticket = worker.submit(0)
                completion = worker.poll(timeout_ms=30000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
                actual = scratch.output[:count // 4]
                error = _relative(actual, expected)
                measurements.append(dict(patches=count, relative_l2=error, canonical_backend_relative_l2=spread,
                                         maximum_absolute=float((actual - expected).abs().max())))
                assert error < spread + .03, measurements
                torch.testing.assert_close(scratch.output, replay_expected, rtol=0, atol=0)
                assert torch.count_nonzero(scratch.output[count // 4:]) == 0
            print(dict(capacity=capacity, compiler=compiler, maximum_stage_relative_l2=stage_error,
                       measurements=measurements))
        finally:
            worker.close()


@CUDA
@torch.no_grad()
@pytest.mark.parametrize("compiler", ["none", "trace", "aot"])
def test_generation_publishes_features_and_rejects_invalid_uploads(manifest, compiler):
    from b12x.preparation import PreparationSession
    from ignition.components.completion import CompletionOutputs
    from ignition.components.sampling import DeviceRequestState
    from ignition.components.vision_state import PrefillVisionInputs, VisionState
    from ignition.recipes.inkling2.vision_state import PreparedVisionGeneration, VisionInputs

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    module = _module(manifest)
    weights = _weights(manifest, module, device)
    scratch = VisionScratch.allocate(64, device, hidden=module.config.text_hidden_size)
    metadata = VisionMetadata.allocate(64, 1, device)
    inputs = VisionInputs.allocate(device)
    requests = DeviceRequestState.allocate(device, vocab_size=201024, capacity=2)
    batch = SimpleNamespace(token_ids=torch.zeros(3, dtype=torch.int32, device=device),
                           row_slot_ids=torch.ones(1, dtype=torch.int32, device=device),
                           row_generations=torch.full((1,), 7, dtype=torch.int64, device=device))
    prefill = PrefillVisionInputs.allocate(3, device)
    state = VisionState(batch, prefill, requests, image_token=200051, features=64,
                        hidden=module.config.text_hidden_size, image_slots=1)
    completion = CompletionOutputs.allocate((2, 2, 32), device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    sequence = torch.ones(1, dtype=torch.int64, device=device)
    requests.generation[1], requests.status[1] = 7, 1
    inputs.vision_slot.fill_(1)
    inputs.vision_generation.fill_(7)
    state.feature_slot.zero_()
    with PreparationSession(device=device, autotune=True, compile_workers=0) as session:
        encoder = PreparedVision(module, scratch, metadata, weights, session=session)
        encoder.functional(session=session)
        session.freeze()
        operation = PreparedVisionGeneration(encoder, inputs, state, completion, frame, sequence)
        errors = torch.zeros(1, dtype=torch.int32, device=device)

        def forward():
            output, status = operation.forward()
            scratch.output.copy_(output)
            errors.copy_(status)

        if compiler == "trace":
            from ignition.preparation.compilation import compile_forward
            forward = compile_forward(forward)
        elif compiler == "aot":
            from ignition.preparation.aot import aot_forward
            owner = forward
            artifact = aot_forward(owner)
            forward = lambda: artifact(owner)

        forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        worker = PreparedGraphBundle(device=0, graphs={"vision": graph}, buffers={},
            resources=(operation, forward, weights, session, scratch)).start_worker()

        def replay():
            torch.cuda.synchronize(device)
            allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            ticket = worker.submit(0)
            result = worker.poll(timeout_ms=30000)
            assert result is not None and result.ticket == ticket and result.error is None
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations

        upload = VisionMetadata(inputs.vision_position_ids[:65], inputs.vision_interpolation_indices[:64],
            inputs.vision_interpolation_weights[:64], inputs.vision_cu_seqlens, inputs.vision_num_patches,
            inputs.vision_num_features)
        try:
            offset = 0
            for variant, size in enumerate(((160, 160), (41, 79), (1, 1), (80, 40))):
                count = _upload(_input(*size, variant), inputs.vision_pixel_values[:64], upload)
                inputs.vision_feature_offset.fill_(offset)
                requests.status[1] = 1
                expected = operation.forward()[0][:count // 4].clone()
                state.count[1] = offset
                state.arena[:, offset:].fill_(float("nan"))
                replay()
                assert errors.item() == 0
                assert state.generation.tolist() == [0, 7]
                assert state.count.tolist() == [0, offset + count // 4]
                torch.testing.assert_close(state.arena[0, offset:offset + count // 4], expected, rtol=0, atol=0)
                assert torch.isnan(state.arena[0, offset + count // 4:]).all()
                offset += count // 4
            batch.token_ids.copy_(torch.tensor([7, 200051, 200051], device=device))
            prefill.vision_token_indices.copy_(torch.tensor([-1, 0, offset - 1], device=device))
            assert state.validate_prefill(batch.token_ids, torch.tensor([3], device=device))
            embedding = torch.ones((3, module.config.text_hidden_size), dtype=torch.bfloat16, device=device)
            state.inject(batch, embedding)
            assert (embedding[0] == 1).all()
            torch.testing.assert_close(embedding[1], state.arena[0, 0], rtol=0, atol=0)
            torch.testing.assert_close(embedding[2], state.arena[0, offset - 1], rtol=0, atol=0)
            inputs.vision_feature_offset.fill_(offset)
            for invalid in ("stale_generation", "class_position", "interpolation_index", "feature_count", "pixel_nan"):
                before = state.arena.clone()
                count = _upload(_input(40, 40, 0), inputs.vision_pixel_values[:64], upload)
                if invalid == "stale_generation":
                    inputs.vision_generation.fill_(6)
                elif invalid == "class_position":
                    inputs.vision_position_ids[0, 0] = 1
                elif invalid == "interpolation_index":
                    inputs.vision_interpolation_indices[0, 0] = 1024
                elif invalid == "feature_count":
                    inputs.vision_num_features.fill_(2)
                else:
                    inputs.vision_pixel_values[0, 0] = float("nan")
                requests.status[1] = 1
                replay()
                assert errors.item() != 0, invalid
                assert state.count.tolist() == [0, offset]
                torch.testing.assert_close(state.arena, before, rtol=0, atol=0, equal_nan=True)
                if invalid == "stale_generation":
                    assert requests.status[1].item() == 1
                else:
                    assert requests.status[1].item() == 3
                inputs.vision_generation.fill_(7)
        finally:
            worker.close()
