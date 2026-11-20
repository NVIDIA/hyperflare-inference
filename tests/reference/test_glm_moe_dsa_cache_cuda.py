# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare deployed GLM cache writers with independent storage arithmetic on fixed operands."""

import os

import pytest
import torch

from ignition.reference.glm_moe_dsa_cache import encode_latent, quantize_index


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires FLARE_RUN_CUDA_TESTS=1")


@torch.no_grad()
def test_native_cache_writes_and_query_rounding_match_independent_reference():
    from b12x.attention.dsa_indexer import glm_pool
    from b12x.preparation import PreparedCall
    from ignition.components.fp8_indexer import write_fp8_index_keys
    from ignition.components.sparse_mla import PreparedLatentCacheWriter
    from ignition.preparation.tuning import RecordedPreparationSession, declare

    generator = torch.Generator().manual_seed(53)
    latent = torch.randn((67, 512), generator=generator).bfloat16()
    latent[0].zero_()
    latent[1].fill_(1.0625)
    latent[1, ::128] = 448
    expected = encode_latent(latent, torch.zeros((67, 64), dtype=torch.bfloat16))[:, :528]
    source = latent.cuda()
    slots = torch.arange(67, dtype=torch.int64, device="cuda")
    slots[-1] = -1
    cache = torch.zeros((2, 64, 528), dtype=torch.uint8, device="cuda")
    keys = latent[:, :128].contiguous()
    # Multiplication by FP32(1/448) changes this key's E4M3 code from 0xeb to 0xea.
    keys[2].zero_()
    keys[2, :2] = torch.tensor([2.59375, -0.486328125], dtype=torch.bfloat16)
    query = keys.cuda()
    query_bytes = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    query_scales = torch.empty(67, device="cuda")
    head_weights = torch.ones(67, device="cuda")
    index_cache = torch.zeros((2, 64 * 132), dtype=torch.uint8, device="cuda")
    with RecordedPreparationSession(device=torch.device("cuda:0"), checkpoint="glm53-storage-reference",
            workload_schema="glm53-storage-reference", numerical_policy="default") as session:
        writer = PreparedLatentCacheWriter(source, cache, slots, session=session, name="latent")
        query_plan = glm_pool.plan_queries(query, query_bytes, query_scales, head_weights, hadamard=False)
        declare(session, (query_plan.request(name="index-query", prepare_call=lambda state: PreparedCall(
            run=lambda: state.quantize(query, query_bytes, query_scales, head_weights))),))
        session.tune()

        def run():
            writer()
            write_fp8_index_keys(query, slots, index_cache, hadamard=False, power_of_two=False)
            head_weights.fill_(1)
            glm_pool.quantize_queries(query, query_bytes, query_scales, head_weights, plan=query_plan)

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        captured = tuple(x.cpu().clone() for x in (cache, index_cache, query_bytes.view(torch.uint8), query_scales, head_weights))
        for _ in range(3):
            graph.replay()
            assert all(torch.equal(actual.cpu(), saved) for actual, saved in
                       zip((cache, index_cache, query_bytes.view(torch.uint8), query_scales, head_weights), captured))
        session.freeze()

    actual = cache.cpu().reshape(-1, 528)
    assert not actual[66:].any(), "inactive slots and unwritten records must remain unchanged"
    assert torch.equal(actual[:2], expected[:2]), "zero groups and exact halfway values have known encodings"
    scales = expected[:66, 512:].contiguous().view(torch.float32)
    assert torch.equal(actual[:66, 512:], expected[:66, 512:]), "record scale computation and placement are exact"
    # PTX rcp.approx.f32 is within one ulp of the exact reciprocal. Two
    # adjacent FP32 values around the correctly rounded reciprocal enclose
    # that interval, including either rounding direction. Test each value's
    # possible E4M3 outputs, rather than allowing an arbitrary byte mismatch.
    inverse = scales.reciprocal()
    lower, upper = inverse, inverse
    for _ in range(2):
        lower = torch.nextafter(lower, torch.full_like(lower, -torch.inf))
        upper = torch.nextafter(upper, torch.full_like(upper, torch.inf))
    values = latent[:66].float().reshape(66, 4, 128)
    endpoints = torch.stack((values * lower[..., None], values * upper[..., None])).clamp(-448, 448)
    rounded = endpoints.to(torch.float8_e4m3fn).float()
    stored = actual[:66, :512].contiguous().view(torch.float8_e4m3fn).float().reshape_as(values)
    assert bool(((stored >= rounded.amin(0)) & (stored <= rounded.amax(0))).all())
    print("latent FP8 cells differing from correctly rounded reciprocal:", int((actual[:66, :512] != expected[:66, :512]).sum()))

    expected_keys, expected_scales = quantize_index(keys, query=False)
    device_keys, device_scales = quantize_index(query, query=False)
    assert expected_scales[2].view(torch.int32).item() == 1002288859
    assert expected_keys[2, 1].item() == 0xeb
    assert torch.equal(device_keys.cpu(), expected_keys) and torch.equal(device_scales.cpu(), expected_scales)
    key_cache = index_cache.cpu()
    assert torch.equal(key_cache[:, :8192].reshape(-1, 128)[:66], expected_keys[:66])
    assert torch.equal(key_cache[:, 8192:].contiguous().view(torch.float32).flatten()[:66], expected_scales[:66])
    expected_queries, expected_scales = quantize_index(keys, query=True)
    assert torch.equal(query_bytes.view(torch.uint8).cpu(), expected_queries)
    assert torch.equal(query_scales.cpu(), expected_scales)
    assert torch.equal(head_weights.cpu(), expected_scales / 64)
