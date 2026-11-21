# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The prepared latent decomposition against canonical Transformers GLM MoE DSA attention."""

import pytest
import torch

from ignition.reference import glm_dsa as reference

TOKENS, TOP_K, BASE = 40, 6, 8e6


@pytest.fixture(scope="module")
def model():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

    torch.manual_seed(7)
    config = GlmMoeDsaConfig(
        vocab_size=128, hidden_size=96, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=8, n_routed_experts=8, num_experts_per_tok=2,
        kv_lora_rank=512, q_lora_rank=48, qk_rope_head_dim=64, qk_nope_head_dim=24, v_head_dim=20,
        index_topk=TOP_K, index_head_dim=128, index_n_heads=4, max_position_embeddings=4096,
        rope_parameters=dict(rope_type="default", rope_theta=BASE), first_k_dense_replace=1,
        mlp_layer_types=["dense", "sparse", "sparse"], indexer_types=["full", "shared", "full"],
        pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = "eager"
    model = GlmMoeDsaForCausalLM(config).eval().requires_grad_(False)
    # Default initialization leaves normalization at identity and biases at zero.
    for layer in model.model.layers:
        attention = layer.self_attn
        for norm in (attention.q_a_layernorm, attention.kv_a_layernorm):
            norm.weight.copy_(torch.rand_like(norm.weight) + 0.5)
        if attention.indexer is not None:
            attention.indexer.k_norm.weight.copy_(torch.rand(128) + 0.5)
            attention.indexer.k_norm.bias.copy_(torch.randn(128) * 0.1)
            # Larger index projections separate the selection scores of nearby positions.
            for linear in (attention.indexer.wq_b, attention.indexer.wk, attention.indexer.weights_proj):
                linear.weight.mul_(20)
    return model


@pytest.fixture(scope="module")
def hidden():
    torch.manual_seed(11)
    return torch.randn(TOKENS, 96)


def canonical(model, layer, hidden, previous=None):
    tokens = hidden.shape[0]
    positions = torch.arange(tokens)[None]
    embeddings = model.model.rotary_emb(hidden[None], position_ids=positions)
    # Eager attention adds its mask to the scores, so the causal mask is additive.
    causal = torch.full((tokens, tokens), torch.finfo(torch.float32).min).triu(1)[None, None]
    output, _, selected = model.model.layers[layer].self_attn(
        hidden_states=hidden[None], position_embeddings=embeddings, attention_mask=causal,
        position_ids=positions, prev_topk_indices=previous)
    return output[0], selected


def decomposed(model, layer, hidden, *, selected=None):
    module = model.model.layers[layer].self_attn
    positions = torch.arange(hidden.shape[0])
    projection = reference.AttentionProjection(module)
    values = projection(hidden, positions, BASE)
    if selected is None:
        selected = reference.select(values["index_query"], values["index_weights"], values["index_key"],
                                    positions, TOP_K)
    records = torch.cat((values["latent"], values["key_rotary"]), -1)
    attended, lse = reference.attend(values["absorbed"], records, selected, module.scaling)
    return projection.finish(attended), selected, values, attended, lse


def visible(selected, row):
    return {int(position) for position in selected[row] if 0 <= position <= row}


def test_selection_matches_the_canonical_indexer(model, hidden):
    _, expected = canonical(model, 0, hidden)
    _, selected, *_ = decomposed(model, 0, hidden)
    for row in range(TOKENS):
        assert visible(selected, row) == visible(expected[0], row)
        assert len(visible(selected, row)) == min(TOP_K, row + 1)
        live = selected[row][selected[row] >= 0]
        assert torch.equal(live, live.sort().values)


@pytest.mark.parametrize("layer", [0])
def test_latent_attention_matches_canonical_attention(model, hidden, layer):
    expected, _ = canonical(model, layer, hidden)
    output, *_ = decomposed(model, layer, hidden)
    assert torch.allclose(output, expected, rtol=1e-4, atol=2e-5)


def test_shared_layer_attends_the_previous_selection(model, hidden):
    _, previous = canonical(model, 0, hidden)
    expected, reused = canonical(model, 1, hidden, previous=previous)
    assert reused is previous and model.model.layers[1].self_attn.indexer is None
    _, selected, *_ = decomposed(model, 0, hidden)
    output, _, values, *_ = decomposed(model, 1, hidden, selected=selected)
    assert values["index_query"] is None
    assert torch.allclose(output, expected, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("world", [8])
def test_context_ranks_combine_to_the_whole_context(model, hidden, world):
    module = model.model.layers[0].self_attn
    _, selected, values, attended, _ = decomposed(model, 0, hidden)
    records = torch.cat((values["latent"], values["key_rotary"]), -1)
    partials, lses = [], []
    for rank in range(world):
        # A rank stores global position p as its local record p // world.
        local_records = records[rank::world]
        partial, lse = reference.attend(values["absorbed"], local_records,
                                        reference.local_selection(selected, rank, world),
                                        module.scaling)
        partials.append(partial)
        lses.append(lse)
    combined = reference.combine(torch.stack(partials), torch.stack(lses))
    assert torch.allclose(combined, attended, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("world", [8])
def test_head_partitions_sum_to_the_whole_output(model, hidden, world):
    module = model.model.layers[0].self_attn
    expected, selected, values, _, _ = decomposed(model, 0, hidden)
    records = torch.cat((values["latent"], values["key_rotary"]), -1)
    positions = torch.arange(TOKENS)
    total = torch.zeros_like(expected)
    width = module.num_heads // world
    for rank in range(world):
        projection = reference.AttentionProjection(module, heads=(rank * width, (rank + 1) * width))
        local = projection(hidden, positions, BASE)
        assert torch.equal(local["latent"], values["latent"])
        attended, _ = reference.attend(local["absorbed"], records, selected, module.scaling)
        total += projection.finish(attended)
    assert torch.allclose(total, expected, rtol=1e-4, atol=2e-5)
