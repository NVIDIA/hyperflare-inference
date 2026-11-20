# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact checkpoint partitioning for GLM tensor-parallel weights."""

import os
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.recipes.glm53 import weights as glm_weights
from ignition.recipes.glm53.weights import TEXT_PREFIX, GlmWeightLoader, _expert_scales
from ignition.reference.qwen38_quant import decode_csf_scales, decode_csf_slab_scales


@pytest.fixture(scope="module")
def manifest():
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    if not snapshot.is_dir():
        pytest.skip("GLM 5.3 Flash checkpoint is not available")
    return inspect_indexed_checkpoint(snapshot)


def read(manifest, name, extent=None):
    with safe_open(str(manifest.snapshot / manifest.tensors[name].shard), framework="pt") as source:
        if extent is not None:
            return source.get_slice(name)[extent].clone()
        return source.get_tensor(name).clone()


def declaration(manifest, prefix):
    """The checkpoint's quantization declaration of a layer prefix, or None when it declares none."""
    import json

    config = json.loads((manifest.snapshot / "config.json").read_text())
    return config["quantization_config"]["quantized_layers"].get(prefix)


@pytest.mark.parametrize("world", [2, 4])
def test_tp_partitions_cover_checkpoint_exactly(world):
    for extent, alignment in ((64, 1), (8192, 128), (16384, 256), (32768, 512), (2048, 128)):
        partitions = [TensorParallelShard(rank, world).partition(extent, alignment=alignment)
                      for rank in range(world)]
        assert partitions[0][0] == 0 and partitions[-1][1] == extent
        assert all(a[1] == b[0] for a, b in pairwise(partitions))
        assert all(a % alignment == b % alignment == 0 for a, b in partitions)


@pytest.mark.parametrize("rank,world", [(1, 1), (0, 0), (-1, 2), (2, 2), (True, 2)])
def test_invalid_tp(rank, world):
    with pytest.raises(ValueError, match="positive world size and a rank within it"):
        TensorParallelShard(rank, world)


def test_loader_construction_leaves_cuda_uninitialized(manifest):
    initialized = torch.cuda.is_initialized()
    loader = GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 2))
    with pytest.raises(RuntimeError, match="open GlmWeightLoader"):
        loader.load_tensor(f"{TEXT_PREFIX}.embed_tokens.weight")
    assert torch.cuda.is_initialized() == initialized


# Routed-expert declarations of both checkpoints: the NVIDIA checkpoint's layers 3..44 declare compressed
# scales and its draft layer W4A16; the calibrated checkpoint declares plain NVFP4 and an MXFP8 draft layer.
NVIDIA_CSF = {"group_size": 16, "quant_algo": "NVFP4", "weight_scale_encoding": "csf_32x2_v1"}
NVIDIA_DRAFT = {"group_size": 16, "quant_algo": "W4A16_NVFP4"}
PLAIN_NVFP4 = {"group_size": 16, "quant_algo": "NVFP4"}
MXFP8_DRAFT = {"group_size": 32, "quant_algo": "MXFP8"}
# layer, declaration, expert activation, expert loader, whether prefill binds A4 launches
EXPERT_SELECTION = [
    (3, NVIDIA_CSF, "a4", "load_nvfp4_experts", False),
    (3, NVIDIA_CSF, "a16", "load_nvfp4_experts", True),
    (45, NVIDIA_DRAFT, "a16", "load_fp4_experts", False),
    (3, PLAIN_NVFP4, "a4", "load_fp4_experts", False),
    (45, MXFP8_DRAFT, "a16", "load_mxfp8_experts", False),
]


def selection_loader(monkeypatch, layer, declaration, activation):
    loader = object.__new__(GlmWeightLoader)
    DirectWeightLoader.__init__(loader, SimpleNamespace(), "cpu")
    loader.__dict__["quantization"] = {f"{TEXT_PREFIX}.layers.{layer}.mlp.experts": declaration}
    loader.shard, loader.expert_activation = TensorParallelShard(0, 4), activation
    chosen = []
    for name in ("load_mxfp8_experts", "load_fp4_experts", "load_nvfp4_experts"):
        monkeypatch.setattr(loader, name, lambda prefixes, *, _name=name, **kwargs: chosen.append(_name) or _name)
    monkeypatch.setattr(glm_weights, "_expert_scales", lambda sources, prefixes: None)
    return loader, chosen


@pytest.mark.parametrize("layer,declaration,activation,expected,a4_prefill", EXPERT_SELECTION)
def test_load_mlp_classifies_experts_without_rejecting_a_declaration(monkeypatch, layer, declaration, activation,
                                                                     expected, a4_prefill):
    """Classification only: the three expert loaders are replaced, as no one snapshot declares all five shapes.

    The CUDA tests below drive the real loaders on each snapshot and check the plan they produce: compressed
    scales (test_actual_nvfp4_shards_and_shared_scale_storage), plain NVFP4 and W4A16 experts
    (test_plain_scale_nvfp4_experts_preserve_packed_weights) and MXFP8 experts
    (test_mxfp8_draft_experts_preserve_tp_partitions).
    """
    loader, chosen = selection_loader(monkeypatch, layer, declaration, activation)
    for name in ("load_tensor", "load_linear_group", "_projection"):
        monkeypatch.setattr(loader, name, lambda *args, **kwargs: None)
    mlp = loader.load_mlp(layer)
    assert chosen == [expected]
    assert mlp.experts_a4_prefill is a4_prefill


class ProceededPastRefusal(Exception):
    pass


@pytest.mark.parametrize("sm103,declaration,mtp,refused", [
    (True, MXFP8_DRAFT, True, True), (False, MXFP8_DRAFT, True, False),
    (True, NVIDIA_DRAFT, True, False), (True, MXFP8_DRAFT, False, False)])
def test_mxfp8_draft_experts_are_refused_on_sm103_before_any_layer_loads(monkeypatch, sm103, declaration, mtp,
                                                                          refused):
    """Drive ModelWeights.load with a loader that fails on the first tensor request."""
    from ignition.components import sm103 as capability
    from ignition.recipes.glm53 import model

    class Loader:
        quantization = {f"{TEXT_PREFIX}.layers.45.mlp.experts": declaration}

        def __init__(self, *args, **options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            return False

        def __getattr__(self, name):
            raise ProceededPastRefusal(name)

    monkeypatch.setattr(capability, "is_sm103", lambda device: sm103)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: next(iter(glm_weights.SUPPORTED_CAPABILITIES)))
    monkeypatch.setattr(glm_weights, "GlmWeightLoader", Loader)
    monkeypatch.setattr(glm_weights, "allocated_tensor_bytes", lambda device: 0)
    monkeypatch.setattr(model, "create_model_tree",
                        lambda snapshot: SimpleNamespace(get_submodule=lambda path: SimpleNamespace()))
    arguments = (SimpleNamespace(snapshot=Path("unused")), "cuda:0")
    if refused:
        with pytest.raises(ValueError, match="draft layer stores MXFP8 experts.*--no-mtp"):
            glm_weights.ModelWeights.load(*arguments, shard=TensorParallelShard(0, 4), mtp=mtp)
    else:
        with pytest.raises(ProceededPastRefusal):
            glm_weights.ModelWeights.load(*arguments, shard=TensorParallelShard(0, 4), mtp=mtp)


def test_gate_up_activation_scale_uses_larger_calibration():
    values = {f"expert.{name}.{suffix}": torch.tensor(value) for name in ("up_proj", "gate_proj", "down_proj")
              for suffix, value in (("input_scale", .25), ("weight_scale_2", .01))}
    values["expert.gate_proj.input_scale"] = torch.tensor(.5)
    scales = _expert_scales(values, ("expert",))
    assert scales[2].item() == 2
    assert scales[3].item() == 4
    values["expert.gate_proj.weight_scale_2"] *= 2
    with pytest.raises(ValueError, match="global scales differ"):
        _expert_scales(values, ("expert",))


cuda = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                         reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")


@cuda
def test_bf16_text_projections_preserve_tp_partitions(manifest):
    if manifest.tensors["lm_head.weight"].dtype != "BF16":
        pytest.skip("requires BF16 GLM text projections")
    shard = TensorParallelShard(3, 4)
    kda = f"{TEXT_PREFIX}.layers.0.self_attn"
    dense = f"{TEXT_PREFIX}.layers.0.mlp"
    shared = f"{TEXT_PREFIX}.layers.3.mlp.shared_experts"
    with GlmWeightLoader(manifest, "cuda:0", shard=shard,
                         prefixes=tuple(p + "." for p in (kda, dense, f"{TEXT_PREFIX}.layers.3.mlp"))) as loader:
        attention, mlp = loader.load_attention(0), loader.load_mlp(0)
        # The recipe's own partition of the shared experts, loaded with the routed experts of layer 3.
        moe = loader.load_mlp(3)
        extent = shard.partition(2048, alignment=128)
        shared_up, shared_down = moe.shared.gate_up, moe.shared.down
        assert loader._session.torch_copy_bytes == 0
    mla = f"{TEXT_PREFIX}.layers.3.self_attn"
    with GlmWeightLoader(manifest, "cuda:0", shard=shard, prefixes=(mla + ".",)) as loader:
        latent = loader.load_attention(3)
        assert loader._session.torch_copy_bytes == 0
    for name, width in (("q_b_proj", 16384), ("kv_b_proj", 32768)):
        assert latent[name].dtype == torch.bfloat16
        assert torch.equal(latent[name].cpu(), read(manifest, f"{mla}.{name}.weight", slice(*shard.partition(width))))
    assert latent["o_proj"].dtype == torch.bfloat16
    assert torch.equal(latent["o_proj"].cpu(),
                       read(manifest, mla + ".o_proj.weight", (slice(None), slice(*shard.partition(16384)))))
    channels, heads = shard.partition(8192), shard.partition(64)
    expected = torch.cat([read(manifest, kda + "." + name + ".weight", slice(*rows) if rows else None)
        for name, rows in (("q_proj", channels), ("k_proj", channels), ("v_proj", channels),
                           ("b_proj", heads), ("f_a_proj", None))])
    assert attention["input_proj"].dtype == torch.bfloat16
    assert torch.equal(attention["input_proj"].cpu(), expected)
    for prefix, up, down, rows in ((dense, mlp.gate_up, mlp.down, shard.partition(12288)),
                                    (shared, shared_up, shared_down, extent)):
        assert up.dtype == down.dtype == torch.bfloat16
        assert torch.equal(up.cpu(), torch.cat([read(manifest, prefix + "." + name + ".weight", slice(*rows))
                                                for name in ("gate_proj", "up_proj")]))
        assert torch.equal(down.cpu(), read(manifest, prefix + ".down_proj.weight", (slice(None), slice(*rows))))


@cuda
def test_mxfp8_draft_experts_preserve_tp_partitions(manifest):
    import json
    from b12x._lib.intrinsics.reference import swizzle_block_scale

    prefix = f"{TEXT_PREFIX}.layers.45.mlp.experts"
    declarations = json.loads((manifest.snapshot / "config.json").read_text())["quantization_config"]["quantized_layers"]
    if declarations[prefix]["quant_algo"] != "MXFP8":
        pytest.skip("requires MXFP8 GLM draft experts")
    shard = TensorParallelShard(3, 4)
    with GlmWeightLoader(manifest, "cuda:0", shard=shard, prefixes=(prefix + ".",)) as loader:
        experts = loader.load_experts(45)
        assert loader._session.torch_copy_bytes == 0
    assert experts.plan.activation.mode == "a8" and experts.plan.activation.swiglu_limit == 10
    first, last = shard.partition(2048, alignment=128)
    local = last - first
    for expert in (0, 287):
        for projection, values, scales in (
            ("up_proj", experts._impl.w1_fp4[expert, :local], experts._impl.w1_blockscale[expert, :local]),
            ("gate_proj", experts._impl.w1_fp4[expert, local:], experts._impl.w1_blockscale[expert, local:]),
            ("down_proj", experts._impl.w2_fp4[expert], experts._impl.w2_blockscale[expert]),
        ):
            rows = (slice(None), slice(first, last)) if projection == "down_proj" else (slice(first, last), slice(None))
            blocks = (slice(None), slice(first // 32, last // 32)) if projection == "down_proj" else rows
            name = f"{prefix}.{expert}.{projection}"
            assert torch.equal(values.cpu().view(torch.uint8), read(manifest, name + ".weight", rows).view(torch.uint8))
            expected_scales = swizzle_block_scale(read(manifest, name + ".weight_scale", blocks))
            assert torch.equal(scales.cpu().view(torch.uint8), expected_scales.view(torch.uint8))


@cuda
def test_nvfp4_draft_head_bytes_match_the_declared_expansion():
    from ignition.components.vocabulary import nvfp4_draft_head
    from ignition.preparation.persistent import allocated_tensor_bytes
    from ignition.recipes import resolve

    device = torch.device("cuda:0")
    shard = torch.randn(154880 // 4, 4096, device=device).bfloat16()
    before = allocated_tensor_bytes(device)
    head = nvfp4_draft_head(shard)
    torch.cuda.synchronize(device)
    # Four shards, each within the allocator's 512-byte rounding of the declared bytes.
    declared = resolve("glm-5.3-flash-nvfp4").checkpoint.expansion_bytes
    measured = 4 * (allocated_tensor_bytes(device) - before)
    assert abs(measured - declared) <= 4 * 512, (measured, declared, type(head))


def assert_projection(manifest, packed, name, *, rows=None, columns=None, output_rows=None):
    """Compare a loaded projection with its checkpoint slice: BF16 tensors exactly, MXFP8 values and scales."""
    if not isinstance(packed, torch.Tensor):
        return assert_mxfp8(manifest, packed, name, rows=rows, columns=columns, output_rows=output_rows)
    expected = read(manifest, name + ".weight", (slice(None) if rows is None else slice(*rows),
                                                 slice(None) if columns is None else slice(*columns)))
    actual = packed.cpu() if output_rows is None else packed[slice(*output_rows)].cpu()
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


def assert_mxfp8(manifest, packed, name, *, rows=None, columns=None, output_rows=None):
    rows = slice(None) if rows is None else slice(*rows)
    columns = slice(None) if columns is None else slice(*columns)
    expected = read(manifest, name + ".weight", (rows, columns)).view(torch.uint8)
    scale_columns = slice(None) if columns.start is None else slice(columns.start // 32, columns.stop // 32)
    expected_scales = read(manifest, name + ".weight_scale", (rows, scale_columns))
    actual = packed.weight.values.view(torch.uint8).reshape(packed.out_features, -1).cpu()
    scales = packed.weight.scale_rows.view(torch.uint8).reshape(packed.out_features, -1).cpu()
    if output_rows is not None:
        actual, scales = actual[slice(*output_rows)], scales[slice(*output_rows)]
    assert torch.equal(actual[:, :expected.shape[1]], expected)
    assert torch.equal(scales[:, :expected_scales.shape[1]], expected_scales)


@cuda
@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_actual_dense_and_attention_shards(manifest, world, rank):
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    prefixes = tuple(f"{TEXT_PREFIX}.layers.{layer}." for layer in (0, 3))
    with GlmWeightLoader(manifest, "cuda:0", shard=shard, prefixes=prefixes) as loader:
        kda = loader.load_attention(0)
        mla = loader.load_attention(3)
        dense = loader.load_mlp(0)
        residual = loader.load_residual(3)
        assert loader._session.torch_copy_bytes == 0
    channels = shard.partition(8192)
    prefix = f"{TEXT_PREFIX}.layers.0.self_attn"
    width = channels[1] - channels[0]
    for index, name in enumerate(("q_proj", "k_proj", "v_proj")):
        assert_projection(manifest, kda["input_proj"], f"{prefix}.{name}", rows=channels,
                        output_rows=(index * width, (index + 1) * width))
    for name in ("f_b_proj", "g_b_proj"):
        assert_projection(manifest, kda[name], f"{prefix}.{name}", rows=channels)
    heads = 64 // world
    assert_projection(manifest, kda["input_proj"], f"{prefix}.b_proj", rows=shard.partition(64),
                    output_rows=(3 * width, 3 * width + heads))
    assert_projection(manifest, kda["input_proj"], f"{prefix}.f_a_proj",
                    output_rows=(3 * width + heads, 3 * width + heads + 128))
    assert_projection(manifest, kda["o_proj"], f"{prefix}.o_proj", columns=channels)
    assert_projection(manifest, kda["g_a_proj"], f"{prefix}.g_a_proj")
    expected_conv = torch.cat([read(manifest, f"{prefix}.{name}_conv1d.weight", slice(*channels))
                               for name in ("q", "k", "v")]).squeeze(1)
    assert torch.equal(kda["conv1d.weight"].cpu(), expected_conv)
    assert torch.equal(kda["A_log"].cpu(), read(manifest, prefix + ".A_log", slice(*shard.partition(64))))
    assert torch.equal(kda["dt_bias"].cpu(), read(manifest, prefix + ".dt_bias", slice(*channels)))
    prefix = f"{TEXT_PREFIX}.layers.3.self_attn"
    for name, width in (("q_b_proj", 16384), ("kv_b_proj", 32768)):
        assert_projection(manifest, mla[name], f"{prefix}.{name}", rows=shard.partition(width))
    assert_projection(manifest, mla["o_proj"], f"{prefix}.o_proj", columns=shard.partition(16384))
    assert_projection(manifest, mla["qkv_a_proj"], f"{prefix}.q_a_proj", output_rows=(0, 1536))
    assert_projection(manifest, mla["qkv_a_proj"], f"{prefix}.kv_a_proj_with_mqa", output_rows=(1536, 2048))
    assert_projection(manifest, mla["qkv_a_proj"], f"{prefix}.indexer.wk", output_rows=(2048, 2176))
    assert_projection(manifest, mla["qkv_a_proj"], f"{prefix}.indexer.weights_proj", output_rows=(2176, 2208))
    assert_projection(manifest, mla["indexer.wq_b"], f"{prefix}.indexer.wq_b")
    prefix = f"{TEXT_PREFIX}.layers.0.mlp"
    extent = slice(*shard.partition(12288))
    expected = torch.cat([read(manifest, f"{prefix}.{name}.weight", extent) for name in ("gate_proj", "up_proj")])
    assert torch.equal(dense.gate_up.cpu(), expected)
    assert torch.equal(dense.down.cpu(), read(manifest, prefix + ".down_proj.weight", (slice(None), extent)))
    for name, value in residual.items():
        assert torch.equal(value.cpu(), read(manifest, f"{TEXT_PREFIX}.layers.3.{name}"))


@cuda
@pytest.mark.parametrize("world,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
def test_actual_nvfp4_shards_and_shared_scale_storage(manifest, world, rank):
    from b12x.moe import fused_moe as moe

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    ids = (0, 287)
    prefix = f"{TEXT_PREFIX}.layers.3.mlp.experts"
    experts = declaration(manifest, prefix)
    if experts is None or not experts.get("weight_scale_encoding"):
        pytest.skip("requires NVFP4 experts with compressed block scales;"
                    " test_plain_scale_nvfp4_experts_preserve_packed_weights covers plain scales")
    with GlmWeightLoader(manifest, "cuda:0", shard=shard,
                         prefixes=tuple(f"{prefix}.{i}." for i in ids)) as loader:
        scratch = loader.allocate_expert_scale_scratch(len(ids))
        experts = loader.load_experts(3, expert_ids=ids, scale_scratch=scratch)
        assert loader._session.torch_copy_bytes == 0
    assert experts.plan.activation.swiglu_limit == 10
    assert experts._impl.w1_blockscale.data_ptr() == scratch[0].data_ptr()
    assert experts._impl.w2_blockscale.data_ptr() == scratch[1].data_ptr()
    moe.expand_scales(experts)
    local = 2048 // world
    decoded = []
    for storage, rows, columns in ((scratch[0], 2 * local, 256), (scratch[1], 4096, local // 16)):
        logical = storage.view(torch.uint8).reshape(len(ids), rows // 128, columns // 4, 32, 4, 4)
        decoded.append(logical.permute(0, 1, 4, 3, 2, 5).reshape(len(ids), rows, columns).cpu())
    first, last = shard.partition(2048)
    for index, expert in enumerate(ids):
        scales, weights = {}, {}
        for name, rows, columns in (("up_proj", 2048, 256), ("gate_proj", 2048, 256), ("down_proj", 4096, 128)):
            name_prefix = f"{prefix}.{expert}.{name}"
            if f"{name_prefix}.weight_scale.csf32_bases" in manifest.tensors:
                planes = [read(manifest, f"{name_prefix}.weight_scale.csf32_{suffix}")
                          for suffix in ("bases", "codes", "page_bases", "offsets", "exceptions")]
                full = decode_csf_scales(*planes, rows=rows, columns=columns)
            else:
                full = decode_csf_slab_scales(
                    read(manifest, f"{name_prefix}.weight_scale.nvfp4_csf_fixed"),
                    read(manifest, f"{name_prefix}.weight_scale.nvfp4_csf_exceptions"),
                    rows=rows, columns=columns)
            scales[name] = full[:, first // 16:last // 16] if name == "down_proj" else full[first:last]
            extent = (slice(None), slice(first // 2, last // 2)) if name == "down_proj" else slice(first, last)
            weights[name] = read(manifest, f"{name_prefix}.weight", extent)
        assert torch.equal(experts._impl.w1_fp4[index].cpu(), torch.cat((weights["up_proj"], weights["gate_proj"])))
        assert torch.equal(experts._impl.w2_fp4[index].cpu(), weights["down_proj"])
        assert torch.equal(decoded[0][index], torch.cat((scales["up_proj"], scales["gate_proj"])))
        assert torch.equal(decoded[1][index], scales["down_proj"])


@cuda
@pytest.mark.parametrize("layer", [3, 45])
@pytest.mark.parametrize("world,rank", [(2, 1), (4, 3)])
def test_plain_scale_nvfp4_experts_preserve_packed_weights(manifest, layer, world, rank, monkeypatch):
    from b12x._lib.intrinsics.reference import swizzle_block_scale
    from b12x.moe import fused_moe

    prefix = f"{TEXT_PREFIX}.layers.{layer}.mlp.experts"
    experts = declaration(manifest, prefix)
    if experts is None or experts["quant_algo"] not in ("NVFP4", "W4A16_NVFP4") or experts.get("weight_scale_encoding"):
        pytest.skip("requires NVFP4 experts with plain E4M3 block scales")
    packed = {}
    prepare = fused_moe.prepare_weights

    def capture(*, plan, weights, **options):
        for name in ("w13", "w2", "w13_block_scales", "w2_block_scales"):
            packed[name] = getattr(weights, name).clone()
        return prepare(plan=plan, weights=weights, **options)

    monkeypatch.setattr(fused_moe, "prepare_weights", capture)

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign a CUDA device explicitly"
    shard = TensorParallelShard(rank, world)
    ids = (0, 287)
    with GlmWeightLoader(manifest, "cuda:0", shard=shard,
                         prefixes=tuple(f"{prefix}.{i}." for i in ids)) as loader:
        experts_loaded = loader.load_experts(layer, expert_ids=ids)
        assert loader._session.torch_copy_bytes == 0
    # The loader read the declaration itself: W4A16 declarations run 16-bit activations, NVFP4 ones A4.
    assert experts_loaded.plan.activation.mode == ("a16" if experts["quant_algo"] == "W4A16_NVFP4" else "a4")
    assert experts_loaded.plan.activation.swiglu_limit == 10
    first, last = shard.partition(2048)
    weights, scales = {}, {}
    for projection in ("up_proj", "gate_proj", "down_proj"):
        extent = (slice(None), slice(first // 2, last // 2)) if projection == "down_proj" else slice(first, last)
        scale_extent = (slice(None), slice(first // 16, last // 16)) if projection == "down_proj" else slice(first, last)
        weights[projection] = torch.stack([read(manifest, f"{prefix}.{expert}.{projection}.weight", extent) for expert in ids])
        scales[projection] = torch.stack([read(manifest, f"{prefix}.{expert}.{projection}.weight_scale", scale_extent) for expert in ids])
    assert torch.equal(packed["w13"].cpu(), torch.cat((weights["up_proj"], weights["gate_proj"]), dim=1))
    assert torch.equal(packed["w2"].cpu(), weights["down_proj"])
    for actual, expected in ((packed["w13_block_scales"], torch.cat((scales["up_proj"], scales["gate_proj"]), dim=1)),
                             (packed["w2_block_scales"], scales["down_proj"])):
        expected = swizzle_block_scale(expected.cuda())
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
