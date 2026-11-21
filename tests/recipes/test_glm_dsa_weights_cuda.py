# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM MoE DSA weight loading on one CUDA device: one rank's tensors against the checkpoint that stores them."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned CUDA device")

# A geometry that eight ranks partition: 64 heads of 80 query, 32 latent and 16 value channels, MLPs of
# 256 channels, a layer with an indexer, a layer that shares it, and the draft layer after them.
CONFIG = dict(
    architectures=["GlmMoeDsaForCausalLM"], model_type="glm_moe_dsa", attention_bias=False, hidden_act="silu",
    hidden_size=128, intermediate_size=256, moe_intermediate_size=256, num_hidden_layers=2,
    num_attention_heads=64, num_key_value_heads=64, q_lora_rank=32, kv_lora_rank=512, qk_nope_head_dim=16,
    qk_rope_head_dim=64, v_head_dim=16, index_n_heads=32, index_head_dim=128, index_topk=2048,
    rope_parameters=dict(rope_theta=10000.0, rope_type="default"), indexer_types=["full", "shared"],
    mlp_layer_types=["dense", "dense"], first_k_dense_replace=2, n_routed_experts=2, n_shared_experts=1,
    num_experts_per_tok=1, n_group=1, topk_group=1, norm_topk_prob=True, num_nextn_predict_layers=1,
    max_position_embeddings=4096, vocab_size=1000, tie_word_embeddings=False, eos_token_id=2)

_ATTENTION = {
    "self_attn.q_a_proj.weight": (32, 128), "self_attn.q_a_layernorm.weight": (32,),
    "self_attn.q_b_proj.weight": (5120, 32), "self_attn.kv_a_proj_with_mqa.weight": (576, 128),
    "self_attn.kv_a_layernorm.weight": (512,), "self_attn.kv_b_proj.weight": (2048, 512),
    "self_attn.o_proj.weight": (128, 1024), "input_layernorm.weight": (128,),
    "post_attention_layernorm.weight": (128,)}
_INDEXER = {
    "self_attn.indexer.wq_b.weight": (4096, 32), "self_attn.indexer.wk.weight": (128, 128),
    "self_attn.indexer.weights_proj.weight": (32, 128), "self_attn.indexer.k_norm.weight": (128,),
    "self_attn.indexer.k_norm.bias": (128,)}
_MLP = {"gate_proj.weight": (256, 128), "up_proj.weight": (256, 128), "down_proj.weight": (128, 256)}
_DENSE = {f"mlp.{name}": shape for name, shape in _MLP.items()}
_DRAFT = {
    "mlp.gate.weight": (2, 128), "mlp.gate.e_score_correction_bias": (2,), "enorm.weight": (128,),
    "hnorm.weight": (128,), "eh_proj.weight": (128, 256), "shared_head.norm.weight": (128,),
    **{f"mlp.shared_experts.{name}": shape for name, shape in _MLP.items()},
    **{f"mlp.experts.{expert}.{name}": shape for expert in range(2) for name, shape in _MLP.items()}}
_LAYERS = ({**_ATTENTION, **_INDEXER, **_DENSE}, {**_ATTENTION, **_DENSE}, {**_ATTENTION, **_INDEXER, **_DRAFT})


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    """Rank 7 of 8's weights of a checkpoint of random BF16 tensors, with the tensors the checkpoint stores."""
    from safetensors.torch import save_file

    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.glm_dsa.weights import GlmDsaWeightLoader

    path = tmp_path_factory.mktemp("checkpoint")
    shapes = {"lm_head.weight": (1000, 128), "model.embed_tokens.weight": (1000, 128), "model.norm.weight": (128,)}
    for layer, tensors in enumerate(_LAYERS):
        shapes.update({f"model.layers.{layer}.{name}": shape for name, shape in tensors.items()})
    generator = torch.Generator().manual_seed(3)
    stored = {name: torch.randn(shape, generator=generator).to(torch.bfloat16) for name, shape in shapes.items()}
    save_file(stored, str(path / "weights.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(tensor.numel() * tensor.element_size() for tensor in stored.values())},
        "weight_map": {name: "weights.safetensors" for name in stored}}))
    (path / "config.json").write_text(json.dumps(CONFIG))
    with torch.no_grad(), GlmDsaWeightLoader(inspect_indexed_checkpoint(path), "cuda:0",
                                             shard=TensorParallelShard(7, 8), mtp=True) as loader:
        weights = SimpleNamespace(stored=stored, head=loader.load_head(), indexed=loader.load_layer(0),
                                  sharing=loader.load_layer(1), draft=loader.load_mtp(None))
    torch.cuda.synchronize()
    return weights


def _same(loaded, *stored):
    return torch.equal(loaded.cpu(), torch.cat(stored))


def test_vocabulary_rows_past_the_checkpoint_are_zero(loaded):
    # Rank 7 owns rows 896 to 1023, and the checkpoint stores 1000 rows.
    assert loaded.head.shape == (128, 128)
    assert _same(loaded.head[:104], loaded.stored["lm_head.weight"][896:])
    assert not loaded.head[104:].any()


def test_attention_projections_partition_by_head(loaded):
    prefix, attention = "model.layers.0.self_attn.", loaded.indexed.attention
    # Heads 56 to 63 hold 80 query rows, 32 latent rows and 16 value columns each.
    assert _same(attention["q_b_proj"], loaded.stored[prefix + "q_b_proj.weight"][4480:])
    assert _same(attention["kv_b_proj"], loaded.stored[prefix + "kv_b_proj.weight"][1792:])
    assert _same(attention["o_proj"], loaded.stored[prefix + "o_proj.weight"][:, 896:])


def test_input_projection_carries_the_indexer_rows_of_a_layer_that_owns_one(loaded):
    stored = loaded.stored
    assert _same(loaded.indexed.attention["input_proj"], stored["model.layers.0.self_attn.q_a_proj.weight"],
                 stored["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"],
                 stored["model.layers.0.self_attn.indexer.wk.weight"],
                 stored["model.layers.0.self_attn.indexer.weights_proj.weight"])
    assert _same(loaded.sharing.attention["input_proj"], stored["model.layers.1.self_attn.q_a_proj.weight"],
                 stored["model.layers.1.self_attn.kv_a_proj_with_mqa.weight"])
    assert "indexer.wq_b" in loaded.indexed.attention and "indexer.wq_b" not in loaded.sharing.attention


def test_mlp_projections_partition_by_channel(loaded):
    stored, dense, shared = loaded.stored, loaded.indexed.mlp, loaded.draft.layer.mlp.shared
    # Channels 224 to 255. A routed layer partitions its shared expert the same way.
    assert _same(dense.gate_up, stored["model.layers.0.mlp.gate_proj.weight"][224:],
                 stored["model.layers.0.mlp.up_proj.weight"][224:])
    assert _same(dense.down, stored["model.layers.0.mlp.down_proj.weight"][:, 224:])
    assert _same(shared.gate_up, stored["model.layers.2.mlp.shared_experts.gate_proj.weight"][224:],
                 stored["model.layers.2.mlp.shared_experts.up_proj.weight"][224:])
    assert _same(shared.down, stored["model.layers.2.mlp.shared_experts.down_proj.weight"][:, 224:])
