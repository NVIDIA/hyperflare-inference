# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical GLM MoE DSA structure from a checkpoint configuration."""

from .config import ModelConfig


def create_model_tree(config: ModelConfig, *, draft=False):
    """Construct the canonical Transformers tree on meta storage, without checkpoint transfers.

    The tree supplies module geometry and normalization constants. Packed weight
    loading, tensor-parallel partitioning, and prepared component replacement
    precede execution. This function initializes no CUDA state. Transformers
    builds no multi-token prediction layer, so with `draft` the tree carries one
    more routed decoder layer that owns an indexer, the draft layer's structure.
    """
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

    from ignition.components.model import canonical_tree

    # The stored quantization declaration describes packed storage, which the recipe loads itself.
    raw = {key: value for key, value in config.text.items() if key != "quantization_config"}
    if draft:
        if not config.draft_layers:
            raise ValueError("this GLM MoE DSA checkpoint has no multi-token prediction layer")
        raw.update(num_hidden_layers=config.layers + 1, indexer_types=[*raw["indexer_types"], "full"],
                   mlp_layer_types=[*raw["mlp_layer_types"], "sparse"])
    return canonical_tree(GlmMoeDsaForCausalLM, GlmMoeDsaConfig.from_dict(raw))
