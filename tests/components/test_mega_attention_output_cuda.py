# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused attention feeds reference grouped projection without activation repacking."""

import json
import os
import weakref
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from ignition.components.flashmla_attention import PreparedFlashMLA
from ignition.components.mega_attention import _outputs
from ignition.components.mega_attention_output import PreparedGroupedFP8Projection
from ignition.components.mega_attention_weights import output_planes, pack_output_planes
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession
from tests.components.test_flashmla_attention_cuda import operands, reference

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def test_compiled_attention_to_checkpoint_output_projection(tmp_path):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    source, library = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT"), os.environ.get("FLARE_FLASHMLA_LIBRARY")
    if not source or not library:
        pytest.skip("requires DeepSeek V4.1 1017 checkpoint and pinned FlashMLA")
    source = Path(source)
    mapping = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    def read(name):
        with safe_open(source / mapping[name], framework="pt", device="cpu") as reader:
            return reader.get_tensor(name)
    w = read("layers.0.attn.wo_a.weight").cuda().view(8, 1024, 4096)
    sf = read("layers.0.attn.wo_a.scale").view(torch.uint8).repeat_interleave(32, 0).cuda().view(8, 1024, 128)
    weights = pack_output_planes(*output_planes(w, sf))
    del w, sf
    data, attention, projection, outputs, graphs, functions = {}, {}, {}, {}, {}, {}
    weak_inputs = []
    cases = ((1, False, False), (17, True, True), (192, True, True), (257, False, False), (8192, False, False))
    with RecordedPreparationSession(device="cuda:0", checkpoint="deepseek41-layer0-output",
            workload_schema="fused-attention-projection", numerical_policy="reference-bitwise",
            compile_workers=2, autotune_effort="off") as session:
        for rows, paged, compressed in cases:
            args = operands(rows, paged, compressed)
            data[rows] = args
            attention[rows] = PreparedFlashMLA(*args, scale=512 ** -.5, kernel_cache=tmp_path,
                library=Path(library), session=session, name=f"attention-{rows}")
            x, xs = _outputs(rows, args[0].device)
            weak_inputs.extend((weakref.ref(x), weakref.ref(xs)))
            projection[rows] = PreparedGroupedFP8Projection(x, xs, *weights, kernel_cache=tmp_path,
                                                           session=session, name=f"woa-{rows}")
            outputs[rows] = torch.empty((rows, 8, 1024), device="cuda", dtype=torch.bfloat16)
        del x, xs
        session.tune()
        assert all(ref() is None for ref in weak_inputs)
        for rows, _, _ in cases:
            def forward(rows=rows):
                x, xs = attention[rows](*data[rows])
                outputs[rows].copy_(projection[rows](x, xs))
            functions[rows] = compile_forward(forward)
            functions[rows]()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                functions[rows]()
            graphs[rows] = graph
        session.freeze()
        import deep_gemm
        for step in range(3):
            for rows, _, _ in reversed(cases):
                args = data[rows]
                args[0].neg_()
                args[1].add_(1)
                args[6].fill_(0 if step == 2 else (17 if step == 1 else args[5].shape[-1]))
                if args[9] is not None:
                    args[9].fill_(0 if step == 2 else (61 if step == 1 else 512))
                x, xs = _outputs(rows, args[0].device)
                reference(args, x, xs)
                expected = torch.empty_like(outputs[rows])
                deep_gemm.fp8_einsum("bhr,hdr->bhd", (x, xs), weights, expected, recipe=(1, 1, 32))
                outputs[rows].fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graphs[rows].replay()
                torch.cuda.synchronize()
                assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                torch.testing.assert_close(outputs[rows], expected, rtol=0, atol=0)
        with pytest.raises(ValueError, match="prepared geometry"):
            projection[257](x[:1], xs[:1])
        for graph in graphs.values():
            graph.reset()
