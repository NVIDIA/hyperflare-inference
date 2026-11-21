# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM MoE DSA configuration, storage declarations, declared costs, and weight partitioning."""

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition import planning
from ignition.errors import UsageError
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.checkpoint import TensorMetadata, inspect_indexed_checkpoint
from ignition.recipes import resolve
from ignition.recipes.glm_dsa import pool
from ignition.recipes.glm_dsa.config import ModelConfig
from ignition.recipes.glm_dsa.recipe import RECIPE
from ignition.recipes.glm_dsa.weights import GlmDsaWeightLoader, checkpoint_quantization

GIB = 1 << 30
# GLM-5.3 geometry as its checkpoint configuration declares it.
GLM53 = dict(
    architectures=["GlmMoeDsaForCausalLM"], model_type="glm_moe_dsa", attention_bias=False, hidden_act="silu",
    hidden_size=6144, intermediate_size=12288, moe_intermediate_size=2048, num_hidden_layers=78,
    num_attention_heads=64, num_key_value_heads=64, q_lora_rank=2048, kv_lora_rank=512, qk_nope_head_dim=192,
    qk_rope_head_dim=64, qk_head_dim=256, v_head_dim=256, index_n_heads=32, index_head_dim=128, index_topk=2048,
    indexer_rope_interleave=True, rope_interleave=True, rope_parameters=dict(rope_theta=8000000, rope_type="default"),
    indexer_types=["full" if layer < 3 or layer % 4 == 2 and layer < 75 else "shared" for layer in range(78)],
    mlp_layer_types=["dense"] * 3 + ["sparse"] * 75, first_k_dense_replace=3, n_routed_experts=256,
    n_shared_experts=1, num_experts_per_tok=8, n_group=1, topk_group=1, norm_topk_prob=True,
    routed_scaling_factor=2.5, scoring_func="sigmoid", topk_method="noaux_tc", num_nextn_predict_layers=1,
    max_position_embeddings=1048576, vocab_size=154880, rms_norm_eps=1e-05, tie_word_embeddings=False,
    eos_token_id=[154820, 154827, 154829],
    quantization_config=dict(quant_method="modelopt", quant_algo="NVFP4", config_groups=dict(group_0=dict(
        input_activations=dict(dynamic=False, num_bits=4, type="float", group_size=16),
        weights=dict(dynamic=False, num_bits=4, type="float", group_size=16), targets=["Linear"])),
        ignore=["lm_head", "model.embed_tokens", "model.layers.0*", "model.layers.1.*", "model.layers.2.*",
                *(f"model.layers.{layer}.{module}*" for layer in range(3, 78)
                  for module in ("mlp.shared_experts", "self_attn"))]))


def test_configuration_reads_glm53_geometry():
    config = ModelConfig.from_dict(copy.deepcopy(GLM53))
    # The draft layer, numbered after the decoder, owns an indexer.
    assert config.indexed(78) and not config.indexed(77)
    assert not config.routed(2) and config.routed(3)
    assert not config.quantized("model.layers.3.self_attn.q_a_proj")
    assert config.quantized("model.layers.3.mlp.experts.0.gate_proj")
    # The exclusions do not name the draft layer, whose experts are nevertheless stored in BF16.
    assert config.quantized("model.layers.78.mlp.experts.0.gate_proj")


@pytest.mark.parametrize("change,message", [
    (dict(kv_lora_rank=256), "unsupported GLM MoE DSA kv_lora_rank"),
    (dict(scoring_func="softmax"), "unsupported GLM MoE DSA scoring_func"),
    (dict(num_key_value_heads=8), "one key head per query head"),
    (dict(model_type="glm5_next"), "preparation requires GlmMoeDsaForCausalLM"),
    (dict(num_nextn_predict_layers=2), "at most one multi-token prediction layer"),
    (dict(indexer_types=["shared"] * 78), "indexer_types must name"),
    (dict(mlp_layer_types=["sparse"] * 78), "mlp_layer_types must be"),
    (dict(rope_parameters=dict(rope_type="yarn", rope_theta=1e4)), "default rotary embeddings"),
    (dict(index_share_for_mtp_iteration=False), "unsupported GLM MoE DSA index_share_for_mtp_iteration"),
])
def test_configuration_rejects_unsupported_geometry(change, message):
    with pytest.raises(ValueError, match=message):
        ModelConfig.from_dict({**copy.deepcopy(GLM53), **change})


def test_configuration_rejects_other_quantization():
    raw = copy.deepcopy(GLM53)
    raw["quantization_config"]["quant_algo"] = "FP8"
    with pytest.raises(ValueError, match="ModelOpt NVFP4"):
        ModelConfig.from_dict(raw)


def test_shard_validation_requires_whole_head_groups():
    config = ModelConfig.from_dict(copy.deepcopy(GLM53))
    config.validate_shard(TensorParallelShard(0, 8))
    # Sixteen ranks of 8-head groups need 128 heads.
    with pytest.raises(ValueError, match="8-head groups"):
        config.validate_shard(TensorParallelShard(0, 16))


def _expert_tensors(config, layer, *, storage):
    hidden, width = config.hidden, config.expert_intermediate
    tensors = {}
    for expert in range(config.experts):
        for projection, (rows, columns) in (("gate_proj", (width, hidden)), ("up_proj", (width, hidden)),
                                            ("down_proj", (hidden, width))):
            module = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            if storage == "bf16":
                entries = {"weight": ("BF16", (rows, columns))}
            else:
                entries = {"weight": ("U8", (rows, columns // 2))}
                if storage == "nvfp4_csf":
                    entries["weight_scale.nvfp4_csf_fixed"] = ("U8", (rows * (1 + columns // 32),))
                    entries["weight_scale.nvfp4_csf_exceptions"] = ("U32", (7,))
                else:
                    entries["weight_scale"] = ("F8_E4M3", (rows, columns // 16))
            for suffix, (dtype, shape) in entries.items():
                name = f"{module}.{suffix}"
                tensors[name] = TensorMetadata(name, "shard.safetensors", 0, 0, shape, dtype)
    return tensors


def test_expert_storage_declarations_follow_stored_tensors():
    config = ModelConfig.from_dict(copy.deepcopy(GLM53))
    tensors = {**_expert_tensors(config, 3, storage="nvfp4_csf"), **_expert_tensors(config, 4, storage="nvfp4"),
               **_expert_tensors(config, 78, storage="bf16")}
    declared = checkpoint_quantization(config, tensors, layers=(3, 4, 78))
    assert declared == {
        "model.layers.3.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"},
        "model.layers.4.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16}}
    from ignition.preparation.quantization import expert_format

    assert expert_format(declared, "model.layers.3.mlp.experts") == ("nvfp4_csf", "a4")
    assert expert_format(declared, "model.layers.4.mlp.experts") == ("nvfp4", "a4")


def test_expert_storage_declarations_reject_contradictions():
    config = ModelConfig.from_dict(copy.deepcopy(GLM53))
    # The configuration packs decoder experts, so BF16 storage there contradicts it.
    with pytest.raises(ValueError, match="contradict"):
        checkpoint_quantization(config, _expert_tensors(config, 3, storage="bf16"), layers=(3,))
    mixed = _expert_tensors(config, 3, storage="nvfp4_csf")
    mixed.update((name, tensor) for name, tensor in _expert_tensors(config, 3, storage="nvfp4").items()
                 if ".experts.5." in name)
    del mixed["model.layers.3.mlp.experts.5.gate_proj.weight_scale.nvfp4_csf_fixed"]
    with pytest.raises(ValueError, match="one stored weight format"):
        checkpoint_quantization(config, mixed, layers=(3,))
    missing = _expert_tensors(config, 3, storage="nvfp4")
    del missing["model.layers.3.mlp.experts.9.down_proj.weight_scale"]
    with pytest.raises(ValueError, match="no E4M3 or CSF block scales"):
        checkpoint_quantization(config, missing, layers=(3,))


def test_expert_storage_declarations_reject_excluded_bf16_decoder_experts():
    raw = copy.deepcopy(GLM53)
    raw["quantization_config"]["ignore"].append("model.layers.3.mlp.experts*")
    config = ModelConfig.from_dict(raw)
    with pytest.raises(ValueError, match="BF16 routed experts are prepared only in the draft layer"):
        checkpoint_quantization(config, _expert_tensors(config, 3, storage="bf16"), layers=(3,))


def test_declared_costs_match_the_prepared_layouts():
    config = ModelConfig.from_dict(copy.deepcopy(GLM53))
    costs = RECIPE.costs(8, None)
    # A rank stores the latent records and index keys of the 256 positions it owns in a page:
    # 256 * (78 * 656 + 21 * 132) bytes.
    assert costs.page_bytes == pool.page_bytes(config) == 13_808_640
    assert costs.state_row_bytes == pool.continuation_layout(config).row_bytes
    # The draft layer adds one latent layer that owns an indexer: 256 * (79 * 656 + 22 * 132) bytes.
    drafting = RECIPE.costs(8, RECIPE.speculation)
    assert drafting.page_bytes == pool.page_bytes(config, draft=True) == 14_010_368
    assert drafting.state_row_bytes == pool.continuation_layout(config, draft=True).row_bytes
    drafted = pool.ContinuationPool.for_model(config, 3, 2, "cpu", draft=True)
    # Plain maintenance graphs exclude the draft regions by this prefix.
    assert {name for name in drafted.pages if name.startswith("draft")} == {"draft.latent", "draft.indexer"}


def _gpu(index, free_gib):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell Server Edition", f"GPU-{index:04d}", 97887 << 20,
                        int(free_gib * GIB), (12, 0))


def test_plan_shares_each_page_across_eight_ranks():
    model = resolve("glm-5.3")
    assert model.recipe is RECIPE and model.checkpoint.architecture == "GlmMoeDsaForCausalLM"
    gpus = tuple(_gpu(index, 76) for index in range(8))
    plan = planning.plan(RECIPE, model.checkpoint, gpus, selected=tuple(range(8)), concurrency=16,
                         reserve_bytes=4 * GIB, speculation=None)
    assert plan.fits and plan.page_tokens == 2048 and plan.context_parallel
    assert plan.tokens == plan.pages * 2048 and plan.context % 2048 == 0
    whole = planning.plan(RECIPE, model.checkpoint, gpus, selected=tuple(range(8)), concurrency=16,
                          reserve_bytes=4 * GIB, speculation=None, context_parallel=False)
    # Without context parallelism every rank holds every position's records, so an eighth of the positions fit.
    assert whole.fits and whole.page_tokens == 256 and not whole.context_parallel
    assert whole.pages == plan.pages and whole.tokens * 8 == plan.tokens
    # Two groups of four ranks each hold a complete context: a page covers half as many positions.
    grouped = planning.plan(RECIPE, model.checkpoint, gpus, selected=tuple(range(8)), concurrency=16,
                            reserve_bytes=4 * GIB, speculation=None, context_parallel_size=4)
    assert grouped.fits and grouped.page_tokens == 1024 and grouped.context_parallel_size == 4
    assert grouped.pages == plan.pages and grouped.tokens * 2 == plan.tokens
    with pytest.raises(UsageError, match="hybrid context parallelism"):
        planning.plan(RECIPE, model.checkpoint, gpus, selected=tuple(range(8)), concurrency=16,
                      reserve_bytes=4 * GIB, speculation=None, context_parallel_size=2)


def test_preparation_rejects_a_plan_that_requests_prompt_logprobs(monkeypatch, tmp_path):
    model = resolve("glm-5.3")
    gpus = tuple(_gpu(index, 76) for index in range(8))
    plain = planning.plan(RECIPE, model.checkpoint, gpus, selected=tuple(range(8)), concurrency=16,
                          reserve_bytes=4 * GIB, speculation=None)
    # The rank group starts one process per CUDA device.
    monkeypatch.setattr("ignition.preparation.rank_group.prepare_group", lambda *args, **kwargs: "prepared")
    assert RECIPE.prepare(tmp_path, plain, compile_forwards=True, kernel_cache=None, progress=None) == "prepared"
    with pytest.raises(ValueError, match="GLM MoE DSA preparation does not capture prompt scoring graphs"):
        RECIPE.prepare(tmp_path, replace(plain, prompt_logprobs=True), compile_forwards=True, kernel_cache=None,
                       progress=None)


def test_frontend_contract_matches_the_native_fixture():
    from ignition.recipes.glm_dsa import contract

    pool = SimpleNamespace(pages={}, layout=SimpleNamespace(row_bytes=32,
        region=lambda name: SimpleNamespace(offset=24, nbytes=8)))
    prepared = contract.prepare({}, {}, pool, vocabulary=154880, page_tokens=2048)
    # The native frontend parity test reads this fixture and cannot read the recipe's declaration.
    fixture = json.loads((Path(__file__).parents[2] / "helios/tests/support/contracts/glm_dsa.json").read_text())
    assert fixture["frontend"] == prepared["frontend"]


SNAPSHOT = Path(os.environ.get("FLARE_GLM_DSA_SNAPSHOT", str(
    Path.home() / ".cache/huggingface/hub/models--local-inference-lab--GLM-5.3-NVFP4-CSF/snapshots"
    / RECIPE.checkpoints[0].revision)))


@pytest.fixture(scope="module")
def manifest():
    if not (SNAPSHOT / "manifest.json").is_file():
        pytest.skip("GLM-5.3 NVFP4-CSF checkpoint is not available")
    return inspect_indexed_checkpoint(SNAPSHOT)


def test_checkpoint_container_matches_the_declaration(manifest):
    assert type(manifest).__name__ == "CsfCheckpoint" and len(manifest.compressed_scales) == 75 * 256 * 3
    assert sum(shard.file_size for shard in manifest.shards) == RECIPE.checkpoints[0].stored_bytes
    config = ModelConfig.load(manifest.metadata_path)
    geometry = GLM53.keys() - {"quantization_config"}
    assert {key: config.text[key] for key in geometry} == {key: GLM53[key] for key in geometry}
    assert (sorted(config.text["quantization_config"]["ignore"])
            == sorted(GLM53["quantization_config"]["ignore"]))
    declared = checkpoint_quantization(config, manifest.tensors, layers=range(3, 79))
    assert set(declared) == {f"model.layers.{layer}.mlp.experts" for layer in range(3, 78)}
    assert all(entry["weight_scale_encoding"] == "csf" for entry in declared.values())


def test_loader_without_drafting_leaves_the_draft_layer_unread(manifest):
    loader = GlmDsaWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 8))
    selected = loader._selected_names()
    assert not any(name.startswith("model.layers.78.") for name in selected)
    assert "model.layers.77.self_attn.o_proj.weight" in selected and "lm_head.weight" in selected


def test_rotation_rounds_as_the_checkpoint_rotation():
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (GlmMoeDsaRotaryEmbedding,
                                                                      apply_rotary_pos_emb_interleave)

    from ignition.recipes.glm_dsa import records

    torch.manual_seed(3)
    values = torch.randn(4, 8, 64).to(torch.bfloat16)
    cos, sin = records.rotary_angles(torch.tensor([0, 7, 123456, -1]), records.rotary_frequencies(8e6, "cpu"))
    rotated = records.rotate(values, cos, sin)
    rotary = GlmMoeDsaRotaryEmbedding(GlmMoeDsaConfig(rope_parameters=dict(rope_type="default", rope_theta=8e6)))
    # BF16 input makes Transformers round cosine, sine, and each product to BF16.
    queries = values.transpose(0, 1)[None]
    # An inactive row rotates as position zero.
    canonical, _ = apply_rotary_pos_emb_interleave(queries, queries,
                                                   *rotary(queries, position_ids=torch.tensor([[0, 7, 123456, 0]])))
    # Transformers concatenates the rotated first elements of every pair, then the second elements.
    canonical = torch.stack((canonical[..., :32], canonical[..., 32:]), -1).flatten(-2)[0].transpose(0, 1)
    assert rotated.dtype == torch.bfloat16 and torch.equal(rotated, canonical)
    # Position zero and inactive rows keep their values.
    assert torch.equal(rotated[0], values[0])
    assert torch.equal(rotated[3], values[3])


def test_record_addresses_follow_the_page_tables():
    from ignition.recipes.glm_dsa.dcp import ContextShards
    from ignition.reference import glm_dsa as reference

    whole = ContextShards()
    tables = torch.tensor([[5, 2, -1], [9, -1, -1]], dtype=torch.int32)
    positions = torch.tensor([0, 255, 256, 511, 512, 768, -1, 300])
    requests = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])
    slots = whole.record_slots(positions, tables[requests])
    # A position's record is its page's first record plus its offset. Unmapped and inactive rows have none.
    # Position 768 lies past the table's width, and its column clamps to the unmapped last one.
    assert slots.tolist() == [5 * 256, 5 * 256 + 255, 2 * 256, 2 * 256 + 255, -1, -1, -1, -1]
    # Rank 7 of eight owns position 7 and would also own an inactive position -1 by its residue.
    assert ContextShards.on(7, 8, torch.device("cpu")).record_slots(
        torch.tensor([7, -1]), tables[torch.tensor([0, 0])]).tolist() == [5 * 256, -1]
    assert whole.index_pages(tables).tolist() == [
        [20, 21, 22, 23, 8, 9, 10, 11, -1, -1, -1, -1], [36, 37, 38, 39, -1, -1, -1, -1, -1, -1, -1, -1]]
    selected = torch.full((3, 2048), -1, dtype=torch.int32)
    selected[0, :4] = torch.tensor([0, 3, 256, 600])
    selected[1, :3] = torch.tensor([1, 2, 300])
    selected[2, :2] = torch.tensor([4, 9])
    physical, counts = reference.local_records(selected, torch.tensor([300, 2, 9]), tables[torch.tensor([0, 0, 1])])
    # Row 0 drops the unmapped third page, and row 1 drops a selection after its own position.
    assert counts.tolist() == [3, 2, 2] and physical.dtype == torch.int32
    assert physical[0, :4].tolist() == [5 * 256, 5 * 256 + 3, 2 * 256, -1]
    assert physical[1, :3].tolist() == [5 * 256 + 1, 5 * 256 + 2, -1]
    assert physical[2, :3].tolist() == [9 * 256 + 4, 9 * 256 + 9, -1]


def test_preparation_configuration_validates_its_capacities():
    from ignition.recipes.glm_dsa.bootstrap import PreparationConfig

    # A 4160-token prefill chunk spans three 2048-position pages or seventeen 256-position pages, and each of
    # the 16 request rows adds two boundary pages.
    config = PreparationConfig(context=1048576, max_active=16, prefill_chunk_size=4160)
    assert config.context_ranks == 8 and config.page_tokens == 2048 and config.preparation_pages == 35
    whole = PreparationConfig(context=1048576, max_active=16, prefill_chunk_size=4160, context_parallel=False)
    assert whole.context_ranks == 1 and whole.page_tokens == 256 and whole.preparation_pages == 49
    with pytest.raises(ValueError, match="eight distinct CUDA devices"):
        PreparationConfig(devices=tuple(range(4)))
    PreparationConfig(context=160000, context_parallel=False)
    with pytest.raises(ValueError, match="2048-aligned"):
        PreparationConfig(context=160000)
    with pytest.raises(ValueError, match="64-aligned"):
        PreparationConfig(prefill_chunk_size=100)
    with pytest.raises(ValueError, match="a4 or a16"):
        PreparationConfig(expert_activation="a8")


@pytest.mark.parametrize("rows,bounds,prepared", [
    (8, ((0, 8),), [(8, "")]),
    (9, ((0, 5), (5, 9)), [(5, ".rows5"), (4, ".rows4")]),
    (16, ((0, 8), (8, 16)), [(8, ".rows8")]),
    (17, ((0, 6), (6, 12), (12, 17)), [(6, ".rows6"), (5, ".rows5")]),
])
def test_sliced_reduction_covers_every_row_in_slices_of_at_most_eight(rows, bounds, prepared):
    from ignition.recipes.glm_dsa.decoder import SlicedReduction

    calls = []
    reduction = SlicedReduction(rows, lambda length, suffix: calls.append((length, suffix)), normalizes=True)
    assert reduction.bounds == bounds
    # Equally long slices share one prepared reduction, named by its length when there are several.
    assert calls == prepared


def test_sliced_reduction_assembles_the_rows_of_its_slices():
    from ignition.recipes.glm_dsa.decoder import SlicedReduction

    class Reduction:
        """Stands for a fused reduction: adds the source to the residual and returns twice the sum."""

        def __init__(self, length):
            self.length = length

        def __call__(self, source, residual, output=None):
            assert source.shape[0] == residual.shape[0] == self.length
            residual += source
            if output is None:
                return residual * 2
            output.copy_(residual * 2)
            return output

    source = torch.arange(9.).view(9, 1)
    residual = torch.ones(9, 1)
    normalized = SlicedReduction(9, lambda length, suffix: Reduction(length), normalizes=True)(source, residual)
    assert torch.equal(residual, torch.arange(1., 10.).view(9, 1))
    assert torch.equal(normalized, torch.arange(2., 20., 2.).view(9, 1))
    residual = torch.ones(9, 1)
    reduced = SlicedReduction(9, lambda length, suffix: Reduction(length), normalizes=False)(source, residual)
    assert torch.equal(reduced, torch.arange(1., 10.).view(9, 1))
