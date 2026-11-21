# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM MoE DSA geometry, cache record layouts, and checkpoint storage declarations.

`GlmMoeDsaForCausalLM` is the text-only GLM architecture in which every decoder
layer runs multi-head latent attention over the positions a DSA indexer selects.
A layer whose `indexer_types` entry is `full` owns an indexer, and a `shared`
layer attends the selection of the nearest preceding `full` layer.
"""

from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
from pathlib import Path

from ignition.components.parallel import TensorParallelShard

ARCHITECTURE = "GlmMoeDsaForCausalLM"
# b12x sparse MLA and the DSA indexer address 64-record kernel pages, and one KV page holds four.
KERNEL_PAGE_RECORDS = 64
PAGE_RECORDS = 4 * KERNEL_PAGE_RECORDS
LATENT_RANK = 512
ROTARY_DIM = 64
# 512 E4M3 latent bytes, four FP32 scales of 128 values each, and the BF16 rotary key.
LATENT_RECORD_BYTES = LATENT_RANK + 4 * (LATENT_RANK // 128) + 2 * ROTARY_DIM
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 2048
# 128 E4M3 index-key bytes and their FP32 scale.
INDEX_RECORD_BYTES = INDEX_DIM + 4
# b12x sparse MLA runs query heads in groups of eight.
HEAD_GROUP = 8


@dataclass(frozen=True)
class ModelConfig:
    """Validated `config.json` of one checkpoint. `text` is the unmodified mapping."""

    text: dict

    @classmethod
    def load(cls, metadata):
        """Read the configuration from a checkpoint's metadata directory."""
        return cls.from_dict(json.loads((Path(metadata) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw):
        if raw.get("model_type") != "glm_moe_dsa" or raw.get("architectures") != [ARCHITECTURE]:
            raise ValueError(f"GLM MoE DSA preparation requires {ARCHITECTURE}")
        for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "q_lora_rank",
                     "qk_nope_head_dim", "v_head_dim", "intermediate_size", "moe_intermediate_size",
                     "n_routed_experts", "num_experts_per_tok", "vocab_size", "max_position_embeddings"):
            if type(raw.get(name)) is not int or raw[name] <= 0:
                raise ValueError(f"GLM MoE DSA requires positive {name}")
        # Absent optional fields keep the Transformers configuration defaults.
        required = dict(kv_lora_rank=LATENT_RANK, qk_rope_head_dim=ROTARY_DIM, index_n_heads=INDEX_HEADS,
            index_head_dim=INDEX_DIM, index_topk=INDEX_TOPK, hidden_act="silu", n_group=1, topk_group=1,
            norm_topk_prob=True, n_shared_experts=1, attention_bias=False, tie_word_embeddings=False)
        optional = dict(scoring_func="sigmoid", topk_method="noaux_tc", rope_interleave=True,
            indexer_rope_interleave=True, moe_layer_freq=1, ep_size=1, mlp_bias=False,
            index_share_for_mtp_iteration=True)
        for name, value in required.items():
            if raw.get(name) != value or type(raw[name]) is not type(value):
                raise ValueError(f"unsupported GLM MoE DSA {name}: {raw.get(name)!r}")
        for name, value in optional.items():
            if name in raw and (raw[name] != value or type(raw[name]) is not type(value)):
                raise ValueError(f"unsupported GLM MoE DSA {name}: {raw[name]!r}")
        if raw["num_key_value_heads"] != raw["num_attention_heads"]:
            raise ValueError("GLM MoE DSA latent attention requires one key head per query head")
        if raw.get("qk_head_dim", raw["qk_nope_head_dim"] + ROTARY_DIM) != raw["qk_nope_head_dim"] + ROTARY_DIM:
            raise ValueError("GLM MoE DSA query width must be its content and rotary widths")
        rope = raw.get("rope_parameters")
        if (not isinstance(rope, dict) or rope.get("rope_type") != "default"
                or type(rope.get("rope_theta")) not in (int, float) or rope["rope_theta"] <= 0):
            raise ValueError("GLM MoE DSA requires default rotary embeddings with a positive base")
        layers = raw["num_hidden_layers"]
        indexers, mlps = raw.get("indexer_types"), raw.get("mlp_layer_types")
        if (not isinstance(indexers, list) or len(indexers) != layers
                or any(kind not in ("full", "shared") for kind in indexers) or indexers[0] != "full"):
            raise ValueError("indexer_types must name a full or shared indexer for every layer, starting full")
        dense = raw.get("first_k_dense_replace")
        if (type(dense) is not int or not 0 <= dense <= layers
                or mlps != ["dense"] * dense + ["sparse"] * (layers - dense)):
            raise ValueError("mlp_layer_types must be first_k_dense_replace dense layers followed by sparse layers")
        if not 0 < raw["num_experts_per_tok"] <= raw["n_routed_experts"]:
            raise ValueError("GLM MoE DSA routing requires top_k within the expert count")
        if raw.get("num_nextn_predict_layers", 0) not in (0, 1):
            raise ValueError("GLM MoE DSA supports at most one multi-token prediction layer")
        eos = raw.get("eos_token_id")
        if not (type(eos) is int or isinstance(eos, list) and eos and all(type(token) is int for token in eos)):
            raise ValueError("GLM MoE DSA requires integer end-of-sequence token ids")
        quantization = raw.get("quantization_config")
        if quantization is not None:
            group = quantization.get("config_groups", {}).get("group_0", {})
            packed = dict(dynamic=False, num_bits=4, type="float", group_size=16)
            if (quantization.get("quant_method") != "modelopt" or quantization.get("quant_algo") != "NVFP4"
                    or quantization.get("kv_cache_quant_algo") is not None
                    or not isinstance(quantization.get("ignore"), list)
                    or list(quantization.get("config_groups", {})) != ["group_0"]
                    or group.get("targets") != ["Linear"] or group.get("weights") != packed
                    or group.get("input_activations") != packed):
                raise ValueError("GLM MoE DSA requires ModelOpt NVFP4 with explicit exclusions and unquantized KV")
        return cls(raw)

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def draft_layers(self):
        return self.text.get("num_nextn_predict_layers", 0)

    @property
    def heads(self):
        return self.text["num_attention_heads"]

    @property
    def query_rank(self):
        return self.text["q_lora_rank"]

    @property
    def content_dim(self):
        return self.text["qk_nope_head_dim"]

    @property
    def value_dim(self):
        return self.text["v_head_dim"]

    @property
    def rotary_base(self):
        return float(self.text["rope_parameters"]["rope_theta"])

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def dense_layers(self):
        return self.text["first_k_dense_replace"]

    @property
    def experts(self):
        return self.text["n_routed_experts"]

    @property
    def intermediate(self):
        return self.text["intermediate_size"]

    @property
    def expert_intermediate(self):
        return self.text["moe_intermediate_size"]

    @property
    def norm_eps(self):
        return self.text.get("rms_norm_eps", 1e-5)

    @property
    def eos_token_ids(self):
        eos = self.text["eos_token_id"]
        return [eos] if type(eos) is int else list(eos)

    def indexed(self, layer):
        """Whether `layer` owns an indexer. The draft layer, numbered after the decoder, always does."""
        if not 0 <= layer < self.layers + self.draft_layers:
            raise ValueError("GLM MoE DSA layer index is outside the decoder and its draft layer")
        return layer >= self.layers or self.text["indexer_types"][layer] == "full"

    def routed(self, layer):
        return layer >= self.dense_layers

    def quantized(self, module):
        """Whether the ModelOpt declaration packs the named Linear module as NVFP4."""
        quantization = self.text.get("quantization_config")
        return quantization is not None and not any(
            fnmatchcase(module, pattern) for pattern in quantization["ignore"])

    def vocabulary_partition(self, shard: TensorParallelShard):
        """One rank's vocabulary rows: 64-aligned equal widths whose tail past the vocabulary is zero."""
        return shard.padded_partition(self.vocabulary, alignment=64)

    def validate_shard(self, shard: TensorParallelShard):
        """Check that every sharded axis divides into this rank count's aligned partitions."""
        if self.heads % (shard.world_size * HEAD_GROUP):
            raise ValueError(f"GLM MoE DSA requires {HEAD_GROUP}-head groups on every rank")
        for extent, alignment in ((self.hidden, 128), (self.intermediate, 32), (self.expert_intermediate, 32)):
            if extent % alignment:
                raise ValueError("GLM MoE DSA projection widths must be aligned for packed weights")
        shard.partition(self.heads)
        shard.partition(self.intermediate, alignment=32)
        shard.partition(self.expert_intermediate, alignment=32)
