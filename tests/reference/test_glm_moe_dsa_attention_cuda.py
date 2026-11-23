# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Independent stored-operand bounds for native SM120 eight-head sparse attention."""

import os

import pytest
import torch

from ignition.reference.glm_moe_dsa_attention_bounds import sparse_attention_allowance
from ignition.reference.glm_moe_dsa_cache import encode_latent

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1")


@pytest.mark.parametrize("capacity", [1, 8])
@pytest.mark.parametrize("seed", [5300, 5301, 5302])
@torch.no_grad()
def test_native_attention_obeys_stored_operand_allowance(capacity, seed):
    from b12x.attention import sparse_mla
    from b12x.attention._shared.mla.kernel import LAST_DECODE_PLAN
    from ignition.components.sparse_mla import PreparedSparseMLA
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession

    device = torch.device("cuda:0")
    assert torch.cuda.get_device_capability(device) == (12, 0), "the arithmetic contract is qualified on SM120"
    generator = torch.Generator().manual_seed(seed)
    latent = (torch.randn((2304, 512), generator=generator) / 8 + 0.25).bfloat16()
    rotary = (torch.randn((2304, 64), generator=generator) / 8).bfloat16()
    stored = encode_latent(latent, rotary)
    queries = (torch.randn((capacity, 8, 576), generator=generator) / 8).bfloat16()
    chosen = torch.full((capacity, 2048), -1, dtype=torch.int32)
    counts = torch.zeros(capacity, dtype=torch.int32)
    visible = torch.full((capacity,), 2049, dtype=torch.int32)
    for row in range(capacity):
        count = 2048 if row % 2 == 0 else 257
        chosen[row, :count] = torch.randperm(2049, generator=generator)[:count].sort().values
        counts[row] = count
    if capacity > 1:
        chosen[-1].fill_(-1)
        counts[-1] = visible[-1] = 0
    query, cache, indices, lengths, visibility = (queries.to(device), stored.to(device).view(-1, 64, 656),
                                               chosen.to(device), counts.to(device), visible.to(device))
    arena = ScratchArena(device, overlap=False)
    caps = sparse_mla.Caps(device=device, num_q_heads=8, max_q_rows=capacity, max_width=2048,
        softmax_scale=1/16, max_batch=capacity, max_kv_rows=2304, kv_dtype=torch.uint8,
        head_dim=576, v_head_dim=512, page_size=64, model_type=sparse_mla.ModelType.GLM_NSA,
        cache_record_bytes=656, mode="decode")
    with RecordedPreparationSession(device=device, checkpoint="glm53-attention-arithmetic",
            workload_schema="glm53-attention-arithmetic", numerical_policy="default") as session:
        session.register_scratch(arena)
        operator = PreparedSparseMLA(query, cache, indices, visibility, lengths,
            torch.zeros((capacity, 8, 512), dtype=torch.bfloat16, device=device),
            caps=caps, session=session, name="attention", workspace=arena)
        operator.functional(session=session)
        session.tune()
        arena.release_preparation()
        for _ in range(2):
            operator(query, indices, visibility, lengths, cache=cache)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operator(query, indices, visibility, lengths, cache=cache)
        plan = dict(LAST_DECODE_PLAN)
        assert plan["native_glm_h8"]
        graph.replay()
        actual = output.cpu().clone()
        for _ in range(3):
            graph.replay()
            assert torch.equal(actual, output.cpu())
        session.freeze()
    for row in range(capacity):
        allowance = sparse_attention_allowance(queries[row], stored, chosen[row],
            chunks_per_split=plan["chunks_per_split"], softmax_scale=1/16)
        error = (actual[row].double() - allowance.expected).abs()
        assert torch.isfinite(actual[row]).all() and (error <= allowance.absolute).all(), {
            "row": row, "plan": plan, "maximum_error_ratio": float((error / allowance.absolute.clamp_min(1e-30)).max())}
        if counts[row]:
            assert ((1.2 * actual[row].double() - allowance.expected).abs() > allowance.absolute).any()
        else:
            assert not actual[row].any()
    print("attention arithmetic plan:", plan)
