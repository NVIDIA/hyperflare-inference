# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit-storage index scoring matches the reference across captured transitions."""

import os

import pytest
import torch

from ignition.components.mxfp4_logits import PreparedPagedMXFP4Logits
from ignition.components.megamoe_kernel import initialize
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.mark.parametrize("page", [64, 128])
def test_paged_logits_reference_and_captured_transitions(page, tmp_path):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    reference = initialize(tmp_path)
    generator = torch.Generator(device="cuda").manual_seed(431 + page)
    columns, physical_pages = 2049, 50
    page_bytes = (page * 68 + 511) // 512 * 512
    backing = torch.zeros((physical_pages, page_bytes), device="cuda", dtype=torch.uint8)
    backing[:, :page * 64].random_(0, 256, generator=generator)
    backing[:, page * 64:page * 68].random_(122, 129, generator=generator)
    cache = backing.as_strided((physical_pages, page, 68), (page_bytes, 68, 1))
    calls, data, outputs, graphs = {}, {}, {}, []
    with RecordedPreparationSession(device="cuda:0", checkpoint="generated-mxfp4-index",
            workload_schema="paged-mqa", numerical_policy="reference-bitwise",
            compile_workers=2, autotune_effort="off") as session:
        for rows in (1, 17, 257):
            query = torch.randint(-128, 128, (rows, 32, 64), device="cuda", dtype=torch.int8, generator=generator)
            scales = torch.randint(122, 129, (rows, 32, 4), device="cuda", dtype=torch.uint8, generator=generator)
            scales = scales.view(torch.int32).squeeze(-1)
            weights = torch.randn((rows, 32), device="cuda", generator=generator)
            lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
            pages = torch.randint(0, physical_pages, (rows, (columns + page - 1) // page),
                                  device="cuda", dtype=torch.int32, generator=generator)
            data[rows] = query, scales, cache, weights, lengths, pages
            calls[rows] = PreparedPagedMXFP4Logits(*data[rows], columns=columns, kernel_cache=tmp_path,
                                                 session=session, name=f"logits-{rows}")
            outputs[rows] = torch.empty((rows, columns), device="cuda", dtype=torch.float32)
        session.tune()
        for rows in data:
            def forward(rows=rows):
                outputs[rows].copy_(calls[rows](*data[rows]))
            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            graphs.append((rows, graph))
        session.freeze()
        original = backing.clone()
        try:
            for step in range(3):
                for rows, graph in reversed(graphs):
                    query, scales, cache, weights, lengths, pages = data[rows]
                    query.bitwise_xor_(17)
                    pages.copy_(pages.flip(1))
                    lengths.copy_((torch.arange(rows, device="cuda", dtype=torch.int32) * 131 + step * 23) % columns)
                    if step == 0:
                        lengths.fill_(columns)
                    elif step == 2:
                        lengths.zero_()
                    lens = lengths[:, None]
                    schedule = reference.get_paged_mqa_logits_metadata(lens, page, reference.get_num_sms())
                    expected = reference.fp8_fp4_paged_mqa_logits(
                        (query[:, None], scales[:, None]), cache.unsqueeze(2), weights,
                        lens, pages, schedule, columns, clean_logits=False)
                    outputs[rows].fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                    mask = torch.arange(columns, device="cuda")[None, :] < lens
                    torch.testing.assert_close(outputs[rows][mask], expected[mask], rtol=0, atol=0)
            assert torch.equal(backing, original)
            with pytest.raises(ValueError, match="prepared geometry"):
                calls[257](*data[1])
        finally:
            for _, graph in graphs:
                graph.reset()
