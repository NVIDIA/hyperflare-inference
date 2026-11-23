# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Known E4M3 encodings and canonical-attention checks for the independent GLM cache adapter."""

import pytest
import torch

from ignition.reference.glm_moe_dsa_cache import (
    QuantizedIndexer, ReferenceCache, decode_latent, dequantize_index, encode_latent, quantize_index,
)


def _known_latent():
    # E4M3 rounds halfway values to an even significand, including subnormals.
    block = torch.tensor([448, 1.0625, 1.1875, 2**-10, 3 * 2**-10, -1.0625, -448, 0], dtype=torch.bfloat16).repeat(16)
    return torch.cat((block, block / 2, block * 2, torch.zeros_like(block))).reshape(1, 1, 1, 512)


def test_latent_bytes_scales_and_rotary_are_independently_checkable():
    latent = _known_latent()
    rotary = torch.arange(64, dtype=torch.bfloat16).reshape(1, 1, 1, 64)
    packed = encode_latent(latent, rotary)
    expected_block = torch.tensor([0x7E, 0x38, 0x3A, 0, 2, 0xB8, 0xFE, 0], dtype=torch.uint8).repeat(16)
    for group in range(3):
        assert torch.equal(packed.flatten()[group * 128:(group + 1) * 128], expected_block)
    assert not packed[..., 384:512].any()
    assert torch.equal(packed[..., 512:528].contiguous().view(torch.float32).flatten(), torch.tensor([1., .5, 2., 1.]))
    actual, rope = decode_latent(packed)
    expected_values = torch.tensor([448, 1, 1.25, 0, 2**-8, -1, -448, 0]).repeat(16)
    assert torch.equal(actual.flatten()[:128], expected_values)
    assert torch.equal(actual.flatten()[128:256], expected_values / 2)
    assert torch.equal(actual.flatten()[256:384], expected_values * 2)
    assert torch.equal(rope, rotary)


def test_latent_comparison_rejects_wrong_scales_and_group_placement():
    packed = encode_latent(_known_latent(), torch.zeros((1, 1, 1, 64), dtype=torch.bfloat16))
    correct, _ = decode_latent(packed)
    wrong_scale = packed.clone()
    wrong_scale[..., 512:528].view(torch.float32)[..., 1] *= 2
    swapped = packed.clone()
    swapped[..., 512:528] = packed[..., 512:528].view(torch.float32).flip(-1).contiguous().view(torch.uint8)
    assert not torch.equal(decode_latent(wrong_scale)[0], correct)
    assert not torch.equal(decode_latent(swapped)[0], correct)


def test_index_queries_and_keys_have_different_scales_and_zero_handling():
    rows = torch.zeros((2, 128), dtype=torch.bfloat16)
    rows[0, :3] = torch.tensor([336, .796875, .890625], dtype=torch.bfloat16)
    keys, key_scales = quantize_index(rows, query=False)
    queries, query_scales = quantize_index(rows, query=True)
    assert key_scales[0] == .75 and query_scales[0] == 1
    assert keys[0, :3].tolist() == [0x7E, 0x38, 0x3A]
    assert queries[0, :3].tolist() == [0x7A, 0x35, 0x36]
    assert key_scales[1] == torch.tensor(1e-4, dtype=torch.float32) / 448
    assert query_scales[1] == 2**-22
    assert not keys[1].any() and not queries[1].any()
    assert not torch.equal(dequantize_index(keys, key_scales)[0], dequantize_index(queries, query_scales)[0])


@pytest.mark.parametrize("precision", ["bf16", "fp8-latent", "fp8-latent-index"])
def test_cache_writes_are_immediately_visible_and_preserve_prefix_on_branch(precision):
    cache = ReferenceCache(precision)
    latent = _known_latent()
    rotary = torch.ones((1, 1, 1, 64), dtype=torch.bfloat16)
    first, _ = cache.update(latent, rotary, 6)
    if precision == "bf16":
        assert torch.equal(first, latent)
    else:
        assert first[0, 0, 0, 1] == 1 and latent[0, 0, 0, 1] == 1.0625
    key = torch.full((1, 1, 128), 1.0625, dtype=torch.bfloat16)
    key[..., 0] = 448
    first_key = cache.update_indexer(key, 6).clone()
    if precision == "fp8-latent-index":
        assert first_key[0, 0, 1] == 1
    else:
        assert first_key[0, 0, 1] == 1.0625
    fork = cache.fork()
    appended, _ = fork.update(latent * 16, rotary * 2, 6)
    appended_key = fork.update_indexer(key * 16, 6)
    assert cache.length == 1 and fork.length == 2
    assert torch.equal(appended[..., :1, :], first)
    assert torch.equal(appended_key[:, :1], first_key)
    with pytest.raises(ValueError, match="one decoder layer"):
        cache.update(latent, rotary, 7)


@torch.no_grad()
def test_bf16_cache_preserves_canonical_attention_and_fp8_changes_first_read():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention, GlmMoeDsaRotaryEmbedding

    torch.manual_seed(53)
    config = GlmMoeDsaConfig(hidden_size=32, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=16, kv_lora_rank=512,
                           qk_nope_head_dim=8, qk_rope_head_dim=64, v_head_dim=16, index_head_dim=128,
                           index_n_heads=2, index_topk=2, num_hidden_layers=1, indexer_types=["full"])
    config._attn_implementation = "eager"
    module = GlmMoeDsaAttention(config, 0).bfloat16().eval()
    module.indexer.weights_proj.float()
    hidden = torch.randn((1, 4, 32), dtype=torch.bfloat16)
    positions = torch.arange(4).reshape(1, -1)
    mask = torch.arange(4)[None, None, None, :] <= positions[:, None, :, None]
    kwargs = dict(position_embeddings=GlmMoeDsaRotaryEmbedding(config)(hidden, positions),
                  attention_mask=torch.where(mask, 0., float("-inf")).to(torch.bfloat16), position_ids=positions)
    expected = module(hidden, **kwargs)[0]
    actual = module(hidden, **kwargs, past_key_values=ReferenceCache("bf16"))[0]
    quantized = module(hidden, **kwargs, past_key_values=ReferenceCache("fp8-latent"))[0]
    assert torch.equal(actual, expected)
    assert not torch.equal(quantized, expected)
    assert torch.isfinite(quantized).all()
    module.indexer = QuantizedIndexer(module.indexer)
    combined = module(hidden, **kwargs, past_key_values=ReferenceCache("fp8-latent-index"))
    repeated = module(hidden, **kwargs, past_key_values=ReferenceCache("fp8-latent-index"))
    assert torch.equal(combined[0], repeated[0])
    assert torch.equal(combined[2], repeated[2])
    assert torch.isfinite(combined[0]).all()
    assert torch.isneginf(module.indexer.scores[0][~mask[0, 0]]).all()
