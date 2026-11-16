# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""An Inkling 2 unembedding shard stored as MXFP8 yields FP32 softcapped logits from the stored values."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.parallel import TensorParallelShard

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an assigned SM120 GPU")


def _expanded(weight):
    """The BF16 matrix an MXFP8 weight stores, in FP64."""
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch, materialize_mxfp8_scale_rows

    rows = weight.weight
    return dequantize_mxfp8_rows_torch(rows.values, materialize_mxfp8_scale_rows(rows))[:, :weight.in_features].double()


@pytest.fixture(scope="module", params=[0, 3])
def shards(request):
    """One rank's weights with a BF16 and with an MXFP8 unembedding shard; rank 3 holds the padded vocabulary rows."""
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.inkling2.weights import ModelWeights

    snapshot = os.environ.get("FLARE_INKLING2_SNAPSHOT")
    if not snapshot:
        pytest.skip("requires FLARE_INKLING2_SNAPSHOT")
    device = torch.device("cuda", 0)

    def load(mxfp8_vocabulary_head):
        with torch.no_grad():
            return ModelWeights.load(inspect_indexed_checkpoint(Path(snapshot)), device,
                shard=TensorParallelShard(request.param, 4), vision=False, layers=1, draft_head_quantization="mxfp8",
                mxfp8_vocabulary_head=mxfp8_vocabulary_head)

    return load(False), load(True)


@torch.no_grad()
def test_checkpoint_unembedding_loads_as_mxfp8_and_proposals_share_it(shards):
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    checkpoint, weights = shards
    assert checkpoint.head.dtype == torch.bfloat16 and isinstance(checkpoint.draft.draft_head, MXFP8LinearWeight)
    assert isinstance(weights.head, MXFP8LinearWeight)
    assert (weights.head.out_features, weights.head.in_features) == tuple(checkpoint.head.shape)
    expanded, original = _expanded(weights.head), checkpoint.head.double()
    assert ((expanded - original).norm() / original.norm()).item() < .035
    # An MXFP8 proposal head would repeat the target's stored values, so the drafter holds no copy.
    assert weights.draft.draft_head is None
    assert torch.equal(weights.head.weight.values.view(torch.uint8),
                       checkpoint.draft.draft_head.weight.values.view(torch.uint8))
    # The embedding shard and the recurrent correction head, folded through the BF16 rows, keep their bits.
    assert torch.equal(weights.embedding, checkpoint.embedding)
    assert torch.equal(weights.draft.recurrent.head, checkpoint.draft.recurrent.head)
    assert torch.equal(weights.draft.recurrent.head_bias, checkpoint.draft.recurrent.head_bias)


@pytest.mark.parametrize("tokens,compiled", [(1, False), (4, True), (8, False), (128, True)])
@torch.no_grad()
def test_checkpoint_mxfp8_unembedding_projects_stored_values_into_fp32_softcapped_logits(shards, tokens, compiled):
    """Decode, verification and packed-round row counts: each kernel multiplies BF16 hidden rows by the stored
    values and rounds the FP32 sum to BF16; softcap and the valid-vocabulary mask then run in FP32, as they do
    for the BF16 shard, whose logits differ by no more than the stored weights' distance allows."""
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.inkling2.vocabulary import PreparedLogits

    checkpoint, weights = shards
    text, device = weights.config.text, weights.embedding.device
    torch.manual_seed(tokens)
    # Rows in the final hidden state's layout: unit RMS scaled by the checkpoint's final norm weight.
    hidden = torch.randn(tokens, text.hidden, device=device)
    hidden = (hidden * hidden.square().mean(-1, keepdim=True).rsqrt() * weights.final_norm.float()).bfloat16()
    with RecordedPreparationSession(device=device, checkpoint="inkling2-checkpoint-unembedding-mxfp8",
            workload_schema="inkling2-checkpoint-unembedding", numerical_policy="default",
            autotune_effort="off") as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        stored = PreparedLogits(weights.head, text, weights.rank, tokens=tokens, session=session,
                                name="stored", workspace=arena)
        reference = PreparedLogits(checkpoint.head, text, checkpoint.rank, tokens=tokens, session=session,
                                   name="reference", workspace=arena)
        session.tune()

        def forward():
            return stored.projection(hidden), stored(hidden), reference(hidden)

        if compiled:
            from ignition.preparation.compilation import compile_forward
            forward = compile_forward(forward)
        raw, logits, expected = forward()

    columns = checkpoint.head.shape[0]
    assert raw.dtype == logits.dtype == expected.dtype == torch.float32
    assert raw.shape == logits.shape == expected.shape == (tokens, columns)
    # The kernel's result is a BF16 value, widened without change.
    assert torch.equal(raw, raw.bfloat16().float())
    source, expanded = hidden.double(), _expanded(weights.head)
    exact, magnitude = source @ expanded.T, source.abs() @ expanded.abs().T
    gamma = text.hidden * 2 ** -24 / (1 - text.hidden * 2 ** -24)
    # FP32 accumulation of the products, then one rounding to BF16.
    assert (((raw.double() - exact).abs() - gamma * magnitude - 2 ** -8 * exact.abs()).max() <= 0).item()
    valid = weights.rank * columns + torch.arange(columns, device=device) < text.valid_vocabulary
    assert valid.any() and bool(valid.all()) == (weights.rank < 3)
    assert torch.equal(logits[:, ~valid], torch.full_like(logits[:, ~valid], float("-inf")))
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected))
    capped = text.softcap * torch.tanh(raw / text.softcap)
    torch.testing.assert_close(logits[:, valid], capped[:, valid], rtol=1e-6, atol=1e-6)
    # Softcap has slope at most one, so the two heads' logits differ by at most the products of the hidden rows
    # with the weight differences, plus each head's accumulation and the stored head's BF16 rounding.
    original = checkpoint.head.double()
    # Each softcap evaluation is within a few FP32 units in the last place of the softcap.
    bound = (source.abs() @ (expanded - original).abs().T + gamma * (magnitude + source.abs() @ original.abs().T)
             + 2 ** -8 * exact.abs() + 8 * 2 ** -23 * text.softcap)
    assert (((logits.double() - expected.double()).abs() - bound)[:, valid].max() <= 0).item()
    assert (logits[:, valid] - expected[:, valid]).abs().max().item() > 0
