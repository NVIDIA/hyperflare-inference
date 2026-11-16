# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused attention weight layouts preserve native bytes and match FlashMLA."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from ignition.components.mega_attention_weights import query_planes, output_planes, pack_output_planes

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.fixture(scope="module")
def reference():
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    library = os.environ.get("FLARE_FLASHMLA_LIBRARY")
    if not library:
        pytest.skip("requires pinned FlashMLA reference library")
    torch.ops.load_library(library)
    import deep_gemm
    return deep_gemm


def check_query(values, scales, reference):
    actual, actual_sf = query_planes(values, scales)
    # Reconstruct the canonical head order independently from output coordinates.
    restored = actual.view(torch.uint8).view(32, 64, 16, values.shape[1]).transpose(0, 1).reshape_as(values)
    restored_sf = actual_sf.view(32, 64, 16, scales.shape[1]).transpose(0, 1).reshape_as(scales)
    torch.testing.assert_close(restored, values.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(restored_sf, scales, rtol=0, atol=0)
    native = (scales.to(torch.int32) << 23).view(torch.float32)
    packed = reference.transform_sf_into_required_layout(native, values.shape[0], values.shape[1], (1, 32))
    expected, expected_sf = torch.ops._flashmla_C.permute_q_b_proj(values, packed, 64, 512)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual_sf, expected_sf.contiguous().view(torch.uint8), rtol=0, atol=0)


def check_output(values, scales, reference):
    actual, actual_sf = output_planes(values, scales)
    groups, rows, columns = values.shape
    restored = actual.view(torch.uint8).view(groups, rows, 16, 8, 32).transpose(2, 3).reshape_as(values)
    restored_sf = actual_sf.view(groups, rows, 16, 8).transpose(2, 3).reshape_as(scales)
    torch.testing.assert_close(restored, values.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(restored_sf, scales, rtol=0, atol=0)
    native = (scales.to(torch.int32) << 23).view(torch.float32)
    packed = reference.transform_sf_into_required_layout(native, rows, columns, (1, 32), groups)
    expected, expected_sf = torch.ops._flashmla_C.permute_wv_proj(values, packed, 8, 512)
    result, result_sf = pack_output_planes(actual, actual_sf)
    torch.testing.assert_close(result.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(result_sf, expected_sf, rtol=0, atol=0)
    assert result_sf.stride() == expected_sf.stride()


def test_synthetic_planes_match_reference_and_roundtrip(reference):
    generator = torch.Generator(device="cuda").manual_seed(3141)
    def planes(shape):
        values = torch.randn(shape, device="cuda", generator=generator).to(torch.float8_e4m3fn)
        scales = torch.randint(0, 256, (*shape[:-1], shape[-1] // 32), dtype=torch.uint8,
                               device="cuda", generator=generator)
        return values, scales
    check_query(*planes((64 * 512, 1280)), reference)
    check_output(*planes((8, 1024, 4096)), reference)


@pytest.mark.parametrize("layer", [0, 20, 39])
def test_checkpoint_attention_weights_match_reference(reference, layer):
    source = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not source:
        pytest.skip("requires the DeepSeek V4.1 1017 lossless-CSF checkpoint")
    source = Path(source)
    mapping = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    def read(name):
        with safe_open(source / mapping[name], framework="pt", device="cpu") as reader:
            return reader.get_tensor(name)
    prefix = f"layers.{layer}.attn"
    def planes(projection):
        values = read(prefix + projection + ".weight")
        scales = read(prefix + projection + ".scale").view(torch.uint8)
        assert scales.shape == (values.shape[0] // 32, values.shape[1] // 32)
        return values.cuda(), scales.repeat_interleave(32, dim=0).cuda()
    q, qs = planes(".wq_b")
    check_query(q, qs, reference)
    a, sa = planes(".wo_a")
    check_output(a.view(8, 1024, 4096), sa.view(8, 1024, 128), reference)


def test_rejects_scale_geometry():
    values = torch.empty((64 * 512, 1280), dtype=torch.float8_e4m3fn)
    scales = torch.empty((64 * 512, 10), dtype=torch.uint8)
    with pytest.raises(ValueError, match="per-row native"):
        query_planes(values, scales)
