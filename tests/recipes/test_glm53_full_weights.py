# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 tensor-parallel partitions cover every checkpoint tensor exactly once."""

from dataclasses import replace
import math
import os
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from safetensors import safe_open

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.glm53_full.config import ModelConfig
from ignition.recipes.glm53_full.weights import GlmWeightLoader, Partition, expected_tensor_names, tensor_partitions

SNAPSHOT = os.environ.get("FLARE_GLM53_FULL_SNAPSHOT")
pytestmark = pytest.mark.skipif(not SNAPSHOT, reason="requires FLARE_GLM53_FULL_SNAPSHOT, a GlmMoeDsaForCausalLM checkpoint")
WORLD = 8
# One tensor of every partitioned kind, in a dense layer, a routed NVFP4 layer, and the BF16 draft layer.
SAMPLES = ("lm_head.weight",
    *(f"model.layers.0.{name}.weight" for name in (
        "self_attn.q_b_proj", "self_attn.kv_b_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")),
    *(f"model.layers.3.mlp.{name}" for name in (
        "shared_experts.gate_proj.weight", "shared_experts.down_proj.weight",
        "experts.7.gate_proj.weight", "experts.7.gate_proj.weight_scale", "experts.7.up_proj.weight",
        "experts.7.down_proj.weight", "experts.7.down_proj.weight_scale")),
    *(f"model.layers.78.mlp.experts.255.{name}.weight" for name in ("up_proj", "down_proj")))


@pytest.fixture(scope="module")
def manifest():
    return inspect_indexed_checkpoint(Path(SNAPSHOT))


@pytest.fixture(scope="module")
def config():
    return ModelConfig.load(SNAPSHOT)


@pytest.fixture(scope="module")
def partitions(manifest, config):
    return [tensor_partitions(config, manifest.tensors, TensorParallelShard(rank, WORLD), draft=True)
            for rank in range(WORLD)]


def read(manifest, name, extent=...):
    with safe_open(str(manifest.snapshot / manifest.tensors[name].shard), framework="pt") as source:
        return source.get_slice(name)[extent].clone() if extent is not ... else source.get_tensor(name)


def reassembled(manifest, name, parts):
    axis = parts[0].axis
    return torch.cat([read(manifest, name, part.extent()) for part in parts], dim=axis)


def same_bytes(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def test_configuration_names_every_checkpoint_tensor(manifest, config):
    assert expected_tensor_names(config) == manifest.tensors.keys()


def test_every_tensor_is_assigned_and_only_the_draft_layer_can_be_unused(manifest, config, partitions):
    for rank in partitions:
        assert rank.keys() == manifest.tensors.keys() and None not in rank.values()
    plain = tensor_partitions(config, manifest.tensors, TensorParallelShard(0, WORLD), draft=False)
    unused = {name for name, part in plain.items() if part is None}
    assert unused == {name for name in manifest.tensors if name.startswith("model.layers.78.")}
    assert len(unused) == 791


def test_partitions_tile_each_stored_axis(manifest, partitions):
    for name, tensor in manifest.tensors.items():
        parts = [rank[name] for rank in partitions]
        if parts[0].axis is None:
            assert all(part == Partition() for part in parts)
            continue
        assert {part.axis for part in parts} == {parts[0].axis}
        assert parts[0].start == 0 and parts[-1].stop == tensor.shape[parts[0].axis]
        assert all(left.stop == right.start and left.stop > left.start for left, right in zip(parts, parts[1:]))
        widths = {part.stop - part.start for part in parts}
        if name == "lm_head.weight":
            # Equal 64-row-aligned widths; the last rank's stored rows end at the vocabulary.
            assert widths == {19392, 19136}
        else:
            assert len(widths) == 1


def test_rank_bytes_sum_to_checkpoint_bytes_plus_replication(manifest, partitions):
    def stored(rank):
        total = 0
        for name, part in rank.items():
            tensor = manifest.tensors[name]
            total += tensor.nbytes if part.axis is None else tensor.nbytes // tensor.shape[part.axis] * (part.stop - part.start)
        return total

    ranks = [stored(rank) for rank in partitions]
    replicated = sum(manifest.tensors[name].nbytes for name, part in partitions[0].items() if part.axis is None)
    checkpoint = sum(tensor.nbytes for tensor in manifest.tensors.values())
    # The last rank stores 256 fewer vocabulary rows; every other tensor partitions equally.
    assert len(set(ranks[:-1])) == 1 and ranks[0] - ranks[-1] == 256 * 6144 * 2
    assert sum(ranks) == checkpoint + (WORLD - 1) * replicated
    print(f"stored bytes per rank: {ranks[0]} ({ranks[0] / 2 ** 30:.2f} GiB), replicated: {replicated}")


@pytest.mark.parametrize("name", SAMPLES)
def test_rank_shards_reassemble_to_checkpoint_bytes(manifest, partitions, name):
    parts = [rank[name] for rank in partitions]
    assert parts[0].axis is not None
    whole = read(manifest, name)
    assert same_bytes(reassembled(manifest, name, parts), whole)
    # A shard shifted by one stored element must not reassemble.
    shifted = [replace(part, start=part.start + 1, stop=part.stop + 1) if index == 2 else part
               for index, part in enumerate(parts)]
    assert not same_bytes(reassembled(manifest, name, shifted), whole)
    # Two ranks reading each other's shard must not reassemble.
    swapped = [parts[1], parts[0], *parts[2:]]
    assert not same_bytes(reassembled(manifest, name, swapped), whole)


def test_loader_rejects_a_dropped_or_unknown_tensor(manifest):
    initialized = torch.cuda.is_initialized()
    shard = TensorParallelShard(0, WORLD)
    GlmWeightLoader(manifest, "cuda:0", shard=shard)
    for dropped in ("model.layers.40.self_attn.q_a_layernorm.weight", "model.layers.3.mlp.experts.200.up_proj.weight_scale",
                    "model.layers.6.self_attn.indexer.k_norm.bias"):
        tensors = dict(manifest.tensors)
        del tensors[dropped]
        with pytest.raises(ValueError, match="differ from the configuration"):
            GlmWeightLoader(replace(manifest, tensors=MappingProxyType(tensors)), "cuda:0", shard=shard)
    tensors = dict(manifest.tensors)
    tensors["model.layers.5.self_attn.indexer.wk.weight"] = tensors["model.layers.6.self_attn.indexer.wk.weight"]
    with pytest.raises(ValueError, match="differ from the configuration"):
        GlmWeightLoader(replace(manifest, tensors=MappingProxyType(tensors)), "cuda:0", shard=shard)
    assert torch.cuda.is_initialized() == initialized


def test_loader_refuses_double_consumption_and_unused_tensors(manifest):
    loader = GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, WORLD))
    loader._use("model.norm.weight")
    with pytest.raises(ValueError, match="consumed twice"):
        loader._use("model.norm.weight")
    with pytest.raises(ValueError, match="unused in this preparation"):
        loader._use("model.layers.78.enorm.weight")


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_supported_rank_counts_partition_every_axis(config, world):
    for rank in range(world):
        config.validate_shard(TensorParallelShard(rank, world))


def test_unsupported_rank_count_is_rejected(config):
    with pytest.raises(ValueError, match="aligned rank partitions"):
        config.validate_shard(TensorParallelShard(0, 3))


def test_precision_follows_the_checkpoint_declaration(manifest, config):
    for name, tensor in manifest.tensors.items():
        if not name.endswith(".weight") or len(tensor.shape) != 2:
            continue
        packed = config.projection_format(name.removesuffix(".weight")) == "nvfp4"
        assert (tensor.dtype == "U8") == packed, name
        assert (name + "_scale" in manifest.tensors) == packed, name
    assert math.prod(manifest.tensors["lm_head.weight"].shape) == config.vocabulary * config.hidden


cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                          reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")


@cuda
@pytest.mark.parametrize("rank,activation", [(0, "a16"), (5, "a4")])
def test_loaded_rank_holds_its_checkpoint_shards(manifest, config, partitions, rank, activation):
    """Layers 0 (indexer, dense MLP) and 3 (no indexer, routed NVFP4 experts) on one of eight ranks."""
    from ignition.recipes.glm53_full.weights import ModelWeights

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    weights = ModelWeights.load(manifest, "cuda:0", shard=TensorParallelShard(rank, WORLD), layers=4,
                                expert_activation=activation)
    parts = partitions[rank]

    def stored(name):
        return read(manifest, name, parts[name].extent())

    assert torch.equal(weights.embedding.cpu(), stored("model.embed_tokens.weight"))
    assert torch.equal(weights.norm.cpu(), stored("model.norm.weight"))
    head, rows = weights.head.cpu(), stored("lm_head.weight")
    assert head.shape[0] == -(-config.vocabulary // (WORLD * 64)) * 64
    assert torch.equal(head[:rows.shape[0]], rows) and not head[rows.shape[0]:].any()
    for layer in (0, 3):
        prefix = f"model.layers.{layer}"
        loaded = weights.layers[layer]
        inputs = ["q_a_proj", "kv_a_proj_with_mqa"] + (["indexer.wk", "indexer.weights_proj"] if layer == 0 else [])
        assert torch.equal(loaded.attention["input_proj"].cpu(),
                           torch.cat([stored(f"{prefix}.self_attn.{name}.weight") for name in inputs]))
        heads = config.raw["num_attention_heads"] // WORLD
        joint = stored(f"{prefix}.self_attn.kv_b_proj.weight").view(heads, 448, 512)
        assert torch.equal(loaded.attention["key"].cpu(), joint[:, :192])
        assert torch.equal(loaded.attention["value"].cpu(), joint[:, 192:].transpose(1, 2))
        names = ["q_b_proj", "o_proj"] + (["indexer.wq_b"] if layer == 0 else [])
        for name in names:
            assert torch.equal(loaded.attention[name].cpu(), stored(f"{prefix}.self_attn.{name}.weight")), name
        names = ["q_a_layernorm.weight", "kv_a_layernorm.weight"] + (
            ["indexer.k_norm.weight", "indexer.k_norm.bias"] if layer == 0 else [])
        for name in names:
            assert torch.equal(loaded.attention[name].cpu(), stored(f"{prefix}.self_attn.{name}")), name
        assert ("indexer.wq_b" in loaded.attention) == (layer == 0)
        for name, value in loaded.norms.items():
            assert torch.equal(value.cpu(), stored(f"{prefix}.{name}.weight"))
        mlp = loaded.mlp if layer == 0 else loaded.mlp.shared
        stem = f"{prefix}.mlp" + ("" if layer == 0 else ".shared_experts")
        assert torch.equal(mlp.gate_up.cpu(), torch.cat([stored(f"{stem}.{name}.weight") for name in ("gate_proj", "up_proj")]))
        assert torch.equal(mlp.down.cpu(), stored(f"{stem}.down_proj.weight"))
    routed = weights.layers[3].mlp
    assert torch.equal(routed.router.cpu(), stored("model.layers.3.mlp.gate.weight"))
    assert torch.equal(routed.correction.cpu(), stored("model.layers.3.mlp.gate.e_score_correction_bias"))
    # A4 experts keep the checkpoint's packing; A16 experts hold the same four-bit codes in
    # b12x's tensor-core order, so their code histogram equals the checkpoint shard's.
    implementation = routed.experts._impl
    for expert in (0, 131, 255):
        stem = f"model.layers.3.mlp.experts.{expert}"
        first = torch.cat([stored(f"{stem}.{name}.weight") for name in ("up_proj", "gate_proj")])
        second = stored(f"{stem}.down_proj.weight")
        for actual, expected in ((implementation.w1_fp4[expert], first), (implementation.w2_fp4[expert], second)):
            actual = actual.cpu().contiguous().view(torch.uint8)
            if activation == "a4":
                assert torch.equal(actual.reshape(expected.shape), expected)
            else:
                codes = lambda value: torch.bincount(torch.stack((value & 15, value >> 4)).flatten().long(), minlength=16)
                padding = actual.numel() - expected.numel()
                histogram = codes(actual)
                histogram[0] -= 2 * padding
                assert padding >= 0 and torch.equal(histogram, codes(expected))
    print(f"rank {rank} {activation}: {weights.allocated_bytes} bytes for four layers, embedding, and head")


@cuda
def test_loaded_rank_holds_its_draft_layer_shards(manifest, partitions):
    """Draft layer 78: its input norms and projection, head norm, and unquantized routed experts on rank 5."""
    from ignition.recipes.glm53_full.weights import ModelWeights

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    rank = 5
    weights = ModelWeights.load(manifest, "cuda:0", shard=TensorParallelShard(rank, WORLD), layers=1, draft=True)
    parts = partitions[rank]

    def stored(name):
        return read(manifest, name, parts[name].extent())

    draft, prefix = weights.draft, "model.layers.78"
    assert torch.equal(draft.embedding_norm.cpu(), stored(f"{prefix}.enorm.weight"))
    assert torch.equal(draft.hidden_norm.cpu(), stored(f"{prefix}.hnorm.weight"))
    assert torch.equal(draft.head_norm.cpu(), stored(f"{prefix}.shared_head.norm.weight"))
    assert draft.projection.shape == (6144, 2 * 6144)
    assert torch.equal(draft.projection.cpu(), stored(f"{prefix}.eh_proj.weight"))
    assert "indexer.wq_b" in draft.layer.attention
    experts = draft.layer.mlp.experts
    assert experts.gate_up.dtype == experts.down.dtype == torch.bfloat16
    for expert in (0, 131, 255):
        stem = f"{prefix}.mlp.experts.{expert}"
        assert torch.equal(experts.gate_up[expert].cpu(),
                           torch.cat([stored(f"{stem}.{name}.weight") for name in ("gate_proj", "up_proj")]))
        assert torch.equal(experts.down[expert].cpu(), stored(f"{stem}.down_proj.weight"))
    assert ModelWeights.load(manifest, "cuda:0", shard=TensorParallelShard(rank, WORLD), layers=1).draft is None
