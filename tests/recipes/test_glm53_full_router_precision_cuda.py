# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM's shared projection keeps router logits in FP32 at the top-eight cutoff."""

from pathlib import Path

import pytest
import torch
from transformers import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaRMSNorm, GlmMoeDsaTopkRouter

from ignition.components.moe import PreparedRouter, RouterScratch
from ignition.components.weight_first_projection import PreparedWeightFirstProjection
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.reference.glm_moe_dsa import CheckpointReader
from ignition.reference.glm_moe_dsa_bounds import projection_interval, sigmoid_relative_allowance

from . import glm53_full_support as support

pytestmark = support.requires_reference


def _within(actual, bounds):
    values = actual.double()
    lower, upper = bounds
    assert bool(((lower <= values) & (values <= upper)).all())


@torch.no_grad()
def test_compiled_weight_first_router_preserves_expert_selection():
    from b12x import gemm

    if not hasattr(gemm, "weight_first_projection"):
        pytest.skip("requires b12x weight_first_projection")
    config = GlmMoeDsaConfig.from_pretrained(support.SNAPSHOT, local_files_only=True)
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("the GLM TP8 weight-first path requires SM120")
    reader = CheckpointReader(inspect_indexed_checkpoint(Path(support.SNAPSHOT)), torch.device("cpu"), torch.bfloat16)
    prefix = "model.layers.6."
    weight = reader.raw(prefix + "mlp.gate.weight")
    bias = reader.raw(prefix + "mlp.gate.e_score_correction_bias")
    local_channels = config.moe_intermediate_size // 8
    shared_weight = torch.cat([reader.raw(prefix + f"mlp.shared_experts.{side}_proj.weight")[:local_channels]
                               for side in ("gate", "up")])
    norm = GlmMoeDsaRMSNorm(config.hidden_size, config.rms_norm_eps).bfloat16()
    norm.weight.copy_(reader.raw(prefix + "post_attention_layernorm.weight"))
    reader.close()
    reference = GlmMoeDsaTopkRouter(config)
    reference.weight.copy_(weight)
    reference.e_score_correction_bias.copy_(bias)

    fixture = torch.load(Path(support.REFERENCE) / "long.pt", mmap=True, map_location="cpu", weights_only=True)
    residuals = fixture["layers"][5][::12].bfloat16() + fixture["layers"][4][::12].bfloat16()
    samples = torch.cat((norm(residuals), weight[:32]))
    reference_logits, _, expert_ids = reference(samples)
    rounded = reference_logits.bfloat16().float().sigmoid() + bias
    top = config.num_experts_per_tok
    rounded_ids = rounded.topk(top, dim=-1).indices.sort(dim=-1).values
    expert_ids = expert_ids.sort(dim=-1).values
    sensitive = (rounded_ids != expert_ids).any(dim=-1).nonzero().flatten()
    rows = 16
    assert sensitive.numel() >= rows, "the fixture must detect a BF16 round-trip before routing"

    source = torch.zeros((rows, config.hidden_size), device=device, dtype=torch.bfloat16)
    experts = weight.shape[0]
    with RecordedPreparationSession(device=device, checkpoint="glm53-layer6", workload_schema="tp8-router",
            numerical_policy="fp32-router-bf16-shared", compile_workers=1) as session:
        projection = PreparedWeightFirstProjection(source, weight.to(device), shared_weight.to(device),
                                                   session=session, name="moe.input_projections")
        router = PreparedRouter(torch.empty((rows, experts), device=device, dtype=torch.float32),
            RouterScratch.allocate(rows, device, experts=experts, top_k=top), session=session,
            name="moe.router", score_func="sigmoid", renormalize=True,
            routed_scaling_factor=config.routed_scaling_factor, correction_bias=bias.to(device)).functional(session=session)
        session.tune()

        def forward():
            logits, shared = projection(source)
            routes = router(logits)
            return logits, shared, routes.ids, routes.weights

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            outputs = compiled()
        session.freeze()
        for indices in (sensitive[:rows], torch.arange(rows), sensitive[:rows]):
            value = samples[indices]
            source.copy_(value)
            graph.replay()
            logits, shared, ids, probabilities = (output.cpu() for output in outputs)
            assert logits.dtype == torch.float32 and shared.dtype == torch.bfloat16
            _within(logits, projection_interval(value, weight, bf16_output=False))
            _within(shared, projection_interval(value, shared_weight, bf16_output=True))
            assert torch.equal(ids.long().sort(dim=-1).values, expert_ids[indices])
            scores = logits.double().sigmoid().gather(1, ids.long())
            expected = scores / scores.sum(dim=-1, keepdim=True) * config.routed_scaling_factor
            unit = 2.**-23
            relative = 2 * sigmoid_relative_allowance(logits.double()).max() + 12 * unit / (1 - 12 * unit)
            _within(probabilities, (expected * (1 - relative), expected * (1 + relative)))
