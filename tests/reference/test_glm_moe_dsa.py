# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The streamed GLM 5.3 reference reads routed experts as the Transformers expert module computes them."""

import json
import os
from pathlib import Path

import pytest
import torch

SNAPSHOT = os.environ.get("FLARE_GLM53_FULL_SNAPSHOT")
pytestmark = pytest.mark.skipif(not SNAPSHOT, reason="requires FLARE_GLM53_FULL_SNAPSHOT, a GlmMoeDsaForCausalLM checkpoint")

EXPERTS, ROWS, SLOTS = 8, 12, 3


def _canonical(reader, config, prefix, decode):
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaExperts

    module = GlmMoeDsaExperts(GlmMoeDsaConfig.from_dict({**config, "n_routed_experts": EXPERTS})).float()
    assert module.gate_up_proj.shape[0] == EXPERTS
    for expert in range(EXPERTS):
        gate, up, down = (decode(f"{prefix}.{expert}.{name}") for name in ("gate_proj", "up_proj", "down_proj"))
        module.gate_up_proj.data[expert].copy_(torch.cat((gate, up)))
        module.down_proj.data[expert].copy_(down)
    return module.requires_grad_(False)


@torch.no_grad()
@pytest.mark.parametrize("layer", [3, 78])
def test_streamed_experts_match_the_transformers_expert_module(layer):
    """Layer 3 stores NVFP4 experts and layer 78 stores BF16 experts."""
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.reference.glm_moe_dsa import CheckpointReader, StreamingExperts
    from ignition.reference.qwen38_quant import decode_nvfp4

    snapshot = Path(SNAPSHOT)
    config = json.loads((snapshot / "config.json").read_text())
    reader = CheckpointReader(inspect_indexed_checkpoint(snapshot), torch.device("cpu"), torch.float32)
    prefix = f"model.layers.{layer}.mlp.experts"
    quantized = f"{prefix}.0.gate_proj.weight_scale" in reader.manifest.tensors
    assert quantized == (layer == 3)

    def decode(name):
        if not quantized:
            return reader.floating(name + ".weight")
        return decode_nvfp4(reader.raw(name + ".weight"), reader.raw(name + ".weight_scale"),
                            reader.raw(name + ".weight_scale_2"))

    canonical = _canonical(reader, config, prefix, decode)
    streamed = StreamingExperts(canonical, reader, prefix, activation="a16")
    generator = torch.Generator().manual_seed(53)
    # BF16-representable inputs: the W4A16 reference rounds activations to BF16.
    hidden = torch.randn((ROWS, config["hidden_size"]), generator=generator).to(torch.bfloat16).float()
    index = torch.stack([torch.randperm(EXPERTS, generator=generator)[:SLOTS] for _ in range(ROWS)])
    weights = torch.rand((ROWS, SLOTS), generator=generator)
    expected = canonical(hidden, index, weights)
    actual = streamed(hidden, index, weights)
    error = float((actual - expected).norm() / expected.norm())
    # BF16 experts are the same FP32 arithmetic. NVFP4 experts round the gated intermediate to
    # BF16 before the down projection: a relative perturbation of at most 2**-9 per element.
    assert error < (2 ** -9 if quantized else 1e-6), error
    # A wrong expert assignment must be visible to this comparison.
    wrong = streamed(hidden, (index + 1) % EXPERTS, weights)
    assert float((wrong - expected).norm() / expected.norm()) > 0.5
    reader.close()
