# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Contiguous prefill scoring retains reference arithmetic through worker capture."""

import os

import pytest
import torch

from ignition.components.megamoe_kernel import initialize
from ignition.components.mxfp4_logits import PreparedContiguousMXFP4Logits
from ignition.preparation.capture import capture_forwards
from ignition.preparation.execution import GraphSignature
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


class ContiguousScoringForward:
    def __init__(self, operands, output, *, session, kernel_cache, name):
        self.operands, self.output = operands, output
        self.scoring = PreparedContiguousMXFP4Logits(*operands, session=session,
                                                    kernel_cache=kernel_cache, name=name)

    def __call__(self):
        self.output.copy_(self.scoring(*self.operands))


def test_contiguous_scoring_reference_worker_capture_and_changing_intervals(tmp_path):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    reference = initialize(tmp_path)
    generator = torch.Generator(device="cuda").manual_seed(1117)
    forwards, signatures = {}, {}
    with RecordedPreparationSession(device="cuda:0", checkpoint="synthetic-packed-mxfp4",
            workload_schema="contiguous-prefill-mqa", numerical_policy="reference-bitwise",
            compile_workers=2, autotune_effort="off") as session:
        for rows, columns in ((1, 65), (17, 2049), (257, 4096), (8192, 2049), (128, 131072)):
            query = torch.randint(-128, 128, (rows, 32, 64), dtype=torch.int8,
                                  device="cuda", generator=generator)
            scales = torch.randint(122, 129, (rows, 32, 4), dtype=torch.uint8,
                                   device="cuda", generator=generator).view(torch.int32).squeeze(-1)
            keys = torch.randint(-128, 128, (columns, 64), dtype=torch.int8,
                                 device="cuda", generator=generator)
            key_scales = torch.randint(122, 129, (columns, 4), dtype=torch.uint8,
                                       device="cuda", generator=generator).view(torch.int32).squeeze(-1)
            weights = torch.randn((rows, 32), device="cuda", generator=generator)
            starts = torch.zeros((rows,), dtype=torch.int32, device="cuda")
            ends = torch.full_like(starts, columns)
            operands = query, scales, keys, key_scales, weights, starts, ends
            output = torch.empty((rows, columns), dtype=torch.float32, device="cuda")
            name = f"scores-{rows}-{columns}"
            forwards[name] = ContiguousScoringForward(operands, output,
                session=session, kernel_cache=tmp_path, name=name)
            signatures[name] = GraphSignature("prefill", 1, rows)
        session.tune()
        graphs, compiled = capture_forwards(forwards, signatures, session=session,
            device=torch.device("cuda:0"), stream=torch.cuda.Stream(),
            pool=torch.cuda.graph_pool_handle(), cache_root=tmp_path)
        try:
            assert all(not forward.cache_hit for forward in compiled.values())
            session.freeze()
            for step in range(3):
                for name, forward in reversed(forwards.items()):
                    query, scales, keys, key_scales, weights, starts, ends = forward.operands
                    rows, columns = query.shape[0], keys.shape[0]
                    query.bitwise_xor_(17)
                    keys.bitwise_xor_(31)
                    weights.neg_()
                    if step == 1:
                        index = torch.arange(rows, device="cuda", dtype=torch.int32)
                        starts.copy_(torch.where(index < rows // 2, 0, columns // 2))
                        ends.copy_(starts + (index * 31) % (columns // 2))
                    elif step == 2:
                        ends.copy_(starts)
                    expected = reference.fp8_fp4_mqa_logits((query, scales), (keys, key_scales),
                        weights, starts, ends, clean_logits=False)
                    original_keys = keys.clone()
                    forward.output.fill_(float("nan"))
                    torch.cuda.synchronize()
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graphs[name].replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                    columns_index = torch.arange(columns, device="cuda")[None, :]
                    visible = (columns_index >= starts[:, None]) & (columns_index < ends[:, None])
                    torch.testing.assert_close(forward.output[visible], expected[visible], rtol=0, atol=0)
                    assert torch.equal(keys, original_keys)
            forward = forwards["scores-17-2049"]
            invalid = list(forward.operands)
            invalid[4] = invalid[4].to(torch.bfloat16)
            with pytest.raises(ValueError, match="prepared geometry"):
                forward.scoring(*invalid)
        finally:
            for graph in graphs.values():
                graph.reset()
