# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The canonical GLM 5.3 module tree on meta storage."""

from pathlib import Path

from .config import ModelConfig


def create_model_tree(snapshot: Path):
    """Construct `GlmMoeDsaForCausalLM` without tensor storage or checkpoint transfers.

    The tree supplies module structure and configuration to weight loading and
    prepared components. It holds no weights and provides no serving forward.
    Transformers omits the checkpoint's draft layer from this class.
    """
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

    from ignition.components.model import canonical_tree

    return canonical_tree(GlmMoeDsaForCausalLM, GlmMoeDsaConfig.from_dict(ModelConfig.load(snapshot).raw))
