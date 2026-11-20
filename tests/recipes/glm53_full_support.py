# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared inputs of GLM 5.3 GPU tests: the checkpoint, reference fixtures, and canonical layers.

`FLARE_GLM53_FULL_SNAPSHOT` names a `GlmMoeDsaForCausalLM` checkpoint.
`FLARE_GLM53_FULL_REFERENCE` names a directory holding `short.pt` and
`long.pt`, fixtures written by `python -m ignition.reference.glm_moe_dsa`
with every layer recorded; the long sequence exceeds 2048 tokens.
"""

import json
import os
from pathlib import Path

import pytest
import torch

SNAPSHOT = os.environ.get("FLARE_GLM53_FULL_SNAPSHOT")
REFERENCE = os.environ.get("FLARE_GLM53_FULL_REFERENCE")
requires_reference = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not SNAPSHOT or not REFERENCE,
    reason="requires FLARE_RUN_CUDA_TESTS=1, FLARE_GLM53_FULL_SNAPSHOT, and FLARE_GLM53_FULL_REFERENCE")


def fixture(name):
    return torch.load(Path(REFERENCE) / f"{name}.pt")


def layer_input(reference, layer, device):
    """The reference input of a decoder layer: the preceding layer's output."""
    if layer == 0:
        raise ValueError("layer zero's input is the embedding, which fixtures do not record")
    return reference["layers"][layer - 1].to(device)


def prior_selection(reference, layer):
    """The reference selection a layer without an indexer reuses."""
    return reference["selections"][max(index for index in reference["selections"] if index < layer)]


@torch.no_grad()
def canonical_layer(manifest, layer, device, dtype=torch.float32):
    """One Transformers decoder layer with checkpoint weights; routed experts are not installed."""
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer

    from ignition.reference.glm_moe_dsa import CheckpointReader

    config = GlmMoeDsaConfig.from_dict(json.loads((manifest.snapshot / "config.json").read_text()))
    config._attn_implementation = "eager"
    with torch.device("meta"):
        module = GlmMoeDsaDecoderLayer(config, layer)
    reader = CheckpointReader(manifest, torch.device(device), dtype)
    reader.install(module, f"model.layers.{layer}")
    reader.close()
    return config, module.eval()


@torch.no_grad()
def canonical_attention(config, module, hidden, selection=None, *, position_hook=None):
    """Normalized layer input, attention output, and selection of one canonical layer, in its dtype."""
    from transformers.masking_utils import create_causal_mask
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRotaryEmbedding

    hidden = hidden.to(next(module.parameters()).dtype).unsqueeze(0)
    positions = torch.arange(hidden.shape[1], device=hidden.device).reshape(1, -1)
    embeddings = GlmMoeDsaRotaryEmbedding(config).to(hidden.device)(hidden, positions)
    if position_hook is not None:
        embeddings = position_hook(*embeddings)
    mask = create_causal_mask(config=config, inputs_embeds=hidden, attention_mask=None, past_key_values=None,
                              position_ids=positions, allow_is_causal_skip=False)
    normalized = module.input_layernorm(hidden)
    output, _, selected = module.self_attn(normalized, attention_mask=mask, position_embeddings=embeddings,
        position_ids=positions, prev_topk_indices=None if selection is None else selection.to(hidden.device).unsqueeze(0))
    return normalized.squeeze(0), output.squeeze(0), selected.squeeze(0)


def causal_selection(selected, device):
    """Pack a reference selection as ascending positions at or before each row, padded with -1, and lengths."""
    rows, width = selected.shape
    row = torch.arange(rows, device=selected.device)[:, None]
    packed = torch.where(selected.long() <= row, selected.long(), torch.iinfo(torch.int32).max).sort(1).values
    lengths = (packed != torch.iinfo(torch.int32).max).sum(1)
    packed = torch.where(packed == torch.iinfo(torch.int32).max, -1, packed)
    records = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    records[:, :width] = packed.to(device=device, dtype=torch.int32)
    return records, lengths.to(device=device, dtype=torch.int32)


def row_errors(actual, expected):
    """Relative L2 error over all rows, and the median and 99th percentile of the per-row relative error."""
    actual, expected = actual.double().cpu(), expected.double().cpu()
    rows = (actual - expected).norm(dim=-1) / expected.norm(dim=-1)
    return dict(relative_l2=float((actual - expected).norm() / expected.norm()), median=float(rows.median()),
                p99=float(rows.quantile(0.99)), maximum=float(rows.max()))
