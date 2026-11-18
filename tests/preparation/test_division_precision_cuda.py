# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compiled division retains eager rounding through serialization and graph replay."""

import os

import pytest
import torch

from ignition import runtime_patches
from ignition.preparation.aot import aot_forward

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


class DivisionForward:
    def __init__(self, operation, dtype):
        self.operation = operation
        self.source = torch.empty((4, 4096), dtype=dtype, device="cuda")
        self.denominator = torch.tensor(1.234567, dtype=torch.float32, device="cuda")
        self.output = torch.empty_like(self.source)

    def __call__(self):
        if self.operation == "silu":
            result = torch.nn.functional.silu(self.source)
        else:
            result = self.source / self.denominator
        self.output.copy_(result)


@torch.no_grad()
@pytest.mark.parametrize("operation", ["division", "silu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_eager_division_rounding_survives_graph_replay(tmp_path, operation, dtype):
    runtime_patches.apply()
    forward = DivisionForward(operation, dtype)
    forward.source.fill_(1)
    compiled = aot_forward(forward)
    compiled(forward)
    artifact = tmp_path / "division.bin"
    compiled.save_compiled_function(str(artifact))
    with artifact.open("rb") as file:
        restored = torch.compiler.load_compiled_function(file)
    restored(forward)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        restored(forward)
    try:
        for seed, denominator in ((42, 1.234567), (43, 0.18215)):
            source = torch.randn(forward.source.shape, generator=torch.Generator().manual_seed(seed)) * 3
            # This representable FP16 input crosses a SiLU rounding boundary with approximate division.
            source[0, 0] = -2.724609375
            forward.source.copy_(source)
            forward.denominator.fill_(denominator)
            forward()
            expected = forward.output.clone()
            forward()
            assert torch.equal(forward.output.view(torch.uint8), expected.view(torch.uint8))
            for _ in range(2):
                forward.output.fill_(float("nan"))
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                assert torch.equal(forward.output.view(torch.uint8), expected.view(torch.uint8))
    finally:
        graph.reset()
