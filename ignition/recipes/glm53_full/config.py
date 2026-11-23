# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 geometry and checkpoint storage declarations read from `config.json`."""

from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
from pathlib import Path

from ignition.components.parallel import TensorParallelShard

ARCHITECTURE = "GlmMoeDsaForCausalLM"

# Prepared kernels and the cache record layout bind these values. A checkpoint that
# declares anything else is a different model, not a configuration of this recipe.
_REQUIRED = {
    "model_type": "glm_moe_dsa", "hidden_size": 6144, "num_attention_heads": 64, "num_key_value_heads": 64,
    "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64,
    "v_head_dim": 256, "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
    "indexer_rope_interleave": True, "rope_interleave": True, "attention_bias": False,
    "intermediate_size": 12288, "moe_intermediate_size": 2048, "n_routed_experts": 256,
    "n_shared_experts": 1, "num_experts_per_tok": 8, "n_group": 1, "topk_group": 1,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True, "hidden_act": "silu",
    "num_nextn_predict_layers": 1, "vocab_size": 154880, "tie_word_embeddings": False,
}


@dataclass(frozen=True)
class ModelConfig:
    """The validated text configuration of one `GlmMoeDsaForCausalLM` checkpoint."""

    raw: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, raw):
        if raw.get("architectures") != [ARCHITECTURE]:
            raise ValueError(f"GLM 5.3 preparation requires {ARCHITECTURE}")
        for key, expected in _REQUIRED.items():
            if raw.get(key) != expected or type(raw.get(key)) is not type(expected):
                raise ValueError(f"GLM 5.3 {key} requires {expected!r}, got {raw.get(key)!r}")
        if type(raw.get("index_share_for_mtp_iteration", False)) is not bool:
            raise ValueError("GLM 5.3 index_share_for_mtp_iteration must be boolean")
        rope = raw.get("rope_parameters")
        if (not isinstance(rope, dict) or rope.get("rope_type") != "default"
                or type(rope.get("rope_theta")) not in (int, float) or rope["rope_theta"] <= 0):
            raise ValueError("GLM 5.3 requires default rotary embeddings with a positive base")
        layers = raw.get("num_hidden_layers")
        indexers, mlps = raw.get("indexer_types"), raw.get("mlp_layer_types")
        if (type(layers) is not int or layers <= 0 or not isinstance(indexers, list) or len(indexers) != layers
                or not isinstance(mlps, list) or len(mlps) != layers
                or any(kind not in ("full", "shared") for kind in indexers)
                or any(kind not in ("dense", "sparse") for kind in mlps)):
            raise ValueError("GLM 5.3 requires one indexer type and one MLP type per decoder layer")
        if indexers[0] != "full":
            raise ValueError("the first GLM 5.3 layer must select its own sparse-attention positions")
        quantization = raw.get("quantization_config")
        if (not isinstance(quantization, dict) or quantization.get("quant_method") != "modelopt"
                or quantization.get("quant_algo") != "NVFP4"
                or quantization.get("kv_cache_quant_algo") is not None
                or not isinstance(quantization.get("ignore"), list)
                or not all(isinstance(pattern, str) for pattern in quantization["ignore"])):
            raise ValueError("GLM 5.3 requires ModelOpt NVFP4 with explicit exclusions and unquantized KV")
        groups = list(quantization.get("config_groups", {}).values())
        expected = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
        if (len(groups) != 1 or groups[0].get("targets") != ["Linear"] or groups[0].get("weights") != expected
                or groups[0].get("input_activations") != expected):
            raise ValueError("GLM 5.3 requires one group-16 NVFP4 weight and static activation declaration")
        return cls(raw)

    @property
    def hidden(self):
        return self.raw["hidden_size"]

    @property
    def vocabulary(self):
        return self.raw["vocab_size"]

    @property
    def layers(self):
        """Target decoder layers; the draft layer follows them in the checkpoint."""
        return self.raw["num_hidden_layers"]

    @property
    def draft_layer(self):
        return self.layers

    @property
    def rope_theta(self):
        return float(self.raw["rope_parameters"]["rope_theta"])

    @property
    def norm_eps(self):
        return self.raw["rms_norm_eps"]

    @property
    def routed_scale(self):
        return self.raw["routed_scaling_factor"]

    def selects(self, layer):
        """Whether a layer computes its own sparse-attention selection. The draft layer does."""
        return layer == self.draft_layer or self.raw["indexer_types"][layer] == "full"

    def routed(self, layer):
        """Whether a layer's MLP is routed experts plus a shared expert. The draft layer's is."""
        return layer == self.draft_layer or self.raw["mlp_layer_types"][layer] == "sparse"

    def projection_format(self, prefix):
        """Declared storage of the weight at `prefix`: "nvfp4" for a Linear module no exclusion matches.

        The declaration targets Linear modules. A router gate is a parameter of
        its router module, so the declaration leaves it unquantized.
        """
        if prefix.endswith(".mlp.gate"):
            return "bf16"
        ignored = any(fnmatchcase(prefix, pattern) for pattern in self.raw["quantization_config"]["ignore"])
        return "bf16" if ignored else "nvfp4"

    def validate_shard(self, shard: TensorParallelShard):
        """Every partitioned axis must divide into the alignment its prepared operator requires."""
        raw = self.raw
        for extent, alignment in ((raw["num_attention_heads"], 1), (raw["moe_intermediate_size"], 128),
                                  (raw["intermediate_size"], 128)):
            shard.partition(extent, alignment=alignment)
