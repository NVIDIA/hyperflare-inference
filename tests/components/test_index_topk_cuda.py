# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared selection preserves ragged prefixes and reference choices under replay."""

import os
from pathlib import Path

import pytest
import torch

from ignition.components.index_topk import PreparedIndexTopK
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.mark.parametrize("rows,dtype,prefill", [
    (17, torch.float32, False), (65, torch.float32, False), (257, torch.float32, True),
    (17, torch.bfloat16, False), (257, torch.bfloat16, True),
])
def test_ragged_selection_and_captured_input_changes(rows, dtype, prefill, tmp_path):
    library = os.environ.get("FLARE_DEEPSELECT_LIBRARY")
    if dtype == torch.bfloat16 and not library:
        pytest.skip("requires the pinned DeepSelect library")
    torch.manual_seed(1737)
    columns = 16384
    logits = torch.randn((rows, columns), device="cuda", dtype=dtype)
    lengths = torch.full((rows,), columns, device="cuda", dtype=torch.int32)
    output = torch.empty((rows, 512), device="cuda", dtype=torch.int32)
    with RecordedPreparationSession(device="cuda:0", checkpoint="generated-index-scores",
            workload_schema="ragged-topk", numerical_policy="exact-selection",
            compile_workers=2, autotune_effort="off") as session:
        select = PreparedIndexTopK(logits, lengths, prefill=prefill, kernel_cache=tmp_path,
            library=None if library is None else Path(library), session=session, name="select")
        session.tune()

        def forward():
            output.copy_(select(logits, lengths))

        compiled = compile_forward(forward)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            for step in range(3):
                logits.neg_()
                lengths.copy_((torch.arange(rows, device="cuda", dtype=torch.int32) * 331 + step * 223) % columns)
                if step == 2:
                    lengths.zero_()
                expected = select(logits, lengths)
                output.fill_(-99)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
                # Reference cooperative selection scatters indices with atomics.
                # Compare token sets; score values are checked independently below.
                assert torch.equal(output.sort(-1).values, expected.sort(-1).values)
                for row, length in enumerate(lengths.tolist()):
                    n = min(length, 512)
                    ids = output[row, :n].long()
                    assert torch.all((ids >= 0) & (ids < length))
                    assert ids.unique().numel() == n
                    assert torch.all(output[row, n:] == -1)
                    wanted = logits[row, :length].float().topk(n).values
                    found = logits[row, ids].float().sort(descending=True).values
                    torch.testing.assert_close(found, wanted, rtol=0, atol=0)
            with pytest.raises(ValueError, match="prepared geometry"):
                select(logits[:, :100], lengths)
        finally:
            graph.reset()
