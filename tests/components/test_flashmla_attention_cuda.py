# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit-buffer FlashMLA matches its allocating API through compiled graph replay."""

import os
import weakref
from pathlib import Path

import pytest
import torch

from ignition.components.flashmla_attention import PreparedFlashMLA
from ignition.components.flashmla_kernel import prepare
from ignition.components.mega_attention import _outputs
from ignition.components.mega_attention_cache import MegaCacheLayout
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


@pytest.fixture(scope="module")
def binding(tmp_path_factory):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    library = os.environ.get("FLARE_FLASHMLA_LIBRARY")
    if not library:
        pytest.skip("requires pinned FlashMLA 0eee43b with SM103 support")
    cache = tmp_path_factory.mktemp("flashmla-binding")
    return prepare(cache, Path(library)), cache, Path(library)


def operands(rows, paged, compressed):
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference

    generator = torch.Generator(device="cuda").manual_seed(313)

    def random(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator) * .2

    def cache(page, kind):
        layout = MegaCacheLayout(page, 528 if kind == "swa" else 288)
        records = pack_deepseek_v41_cache_reference(random((page * 8, 512)), page_size=page, cache_kind=kind)
        records = records.view(8, page, layout.record_bytes)
        planes = torch.cat((records[:, :, :layout.payload_bytes].flatten(1),
                            records[:, :, layout.payload_bytes:].flatten(1)), -1)
        backing = torch.zeros((8, layout.page_bytes), dtype=torch.uint8, device="cuda")
        result = layout.view(backing)
        result.copy_(planes.view_as(result))
        return result

    q = random((rows, 64, 512))
    q = q.reshape(rows, 64, 32, 16).transpose(1, 2).contiguous().reshape_as(q)
    pos = torch.arange(2048, 2048 + rows, dtype=torch.int32, device="cuda")
    angles = torch.arange(8192, device="cuda").float()[:, None] / (
        10000 ** (torch.arange(32, device="cuda").float() / 32))[None, :]
    cs = torch.cat((angles.cos(), angles.sin()), -1)
    sink = random((64,)).float()
    kv = cache(136, "swa") if paged else random((2048, 1, 512))
    width = 128 if paged else 640
    indices = torch.randint(0, 1024, (rows, width), dtype=torch.int32, device="cuda", generator=generator)
    if not paged:
        indices = indices.unsqueeze(1)
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    extra = cache(64, "indexed") if compressed else None
    ci = torch.arange(512, dtype=torch.int32, device="cuda").repeat(rows, 1) if compressed else None
    cl = torch.full((rows,), 512, dtype=torch.int32, device="cuda") if compressed else None
    return q, pos, cs, sink, kv, indices, lengths, extra, ci, cl


def reference(args, output, scales):
    q, pos, cs, sink, kv, indices, lengths, extra, ci, cl = args
    if kv.dtype == torch.bfloat16:
        return torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_fwd(q, kv, indices, 512 ** -.5, 512,
            sink, lengths, False, 0., pos, False, 64, cs, 8, 32, True, True, True, output, scales)
    return torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(q, kv.unsqueeze(-2), indices,
        512 ** -.5, 512, sink, lengths, None if extra is None else extra.unsqueeze(-2), ci, cl,
        False, 0., pos, False, 64, cs, 8, 32, True, True, True, output, scales)


def test_compiled_catalog_exposes_storage_and_preserves_results(binding):
    _, cache, library = binding
    cases = ((1, False, False), (257, False, False), (1024, False, False),
             (25, True, False), (77, True, True), (192, True, True))
    operations, data, outputs, references, functions, graphs = {}, {}, {}, {}, {}, {}
    declaration_refs = []
    with RecordedPreparationSession(device="cuda:0", checkpoint="synthetic-flashmla",
            workload_schema="prefill-and-paged-decode", numerical_policy="flashmla-bitwise",
            compile_workers=2, autotune_effort="off") as session:
        for case in cases:
            args = operands(*case)
            data[case] = args
            examples = list(args)
            examples[0] = args[0].clone()
            declaration_refs.append(weakref.ref(examples[0]))
            operations[case] = PreparedFlashMLA(*examples, scale=512 ** -.5, kernel_cache=cache,
                                               library=library, session=session, name=f"attention-{case}")
            outputs[case] = _outputs(case[0], args[0].device)
            references[case] = _outputs(case[0], args[0].device)
        del examples
        session.tune()
        assert all(ref() is None for ref in declaration_refs)
        for case in cases:
            def forward(case=case):
                result, scales = operations[case](*data[case])
                outputs[case][0].copy_(result)
                outputs[case][1].copy_(scales)
            functions[case] = compile_forward(forward)
            functions[case]()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                functions[case]()
            graphs[case] = graph
        session.freeze()
        for step in range(3):
            for case in reversed(cases):
                args = data[case]
                args[0].neg_()
                args[1].add_(1)
                args[5].copy_(torch.roll(args[5], 17, -1))
                args[6].fill_(0 if step == 2 else (17 if step == 1 else args[5].shape[-1]))
                if args[9] is not None:
                    args[8].copy_(torch.roll(args[8], 13, -1))
                    args[9].fill_(0 if step == 2 else (61 if step == 1 else 512))
                reference(args, *references[case])
                before_inputs = [t.clone() for t in args if t is not None]
                outputs[case][0].view(torch.uint8).fill_(255)
                outputs[case][1].fill_(-999)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graphs[case].replay()
                torch.cuda.synchronize()
                assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                torch.testing.assert_close(outputs[case][0].view(torch.uint8), references[case][0].view(torch.uint8), rtol=0, atol=0)
                torch.testing.assert_close(outputs[case][1], references[case][1], rtol=0, atol=0)
                assert torch.isfinite(outputs[case][0].float()).all()
                for actual, expected in zip((t for t in args if t is not None), before_inputs):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        args = data[(257, False, False)]
        with pytest.raises(ValueError, match="prepared geometry"):
            operations[(257, False, False)](args[0][:1], *args[1:])
        for graph in graphs.values():
            graph.reset()


def test_binding_writes_reference_statistics(binding):
    module, _, _ = binding
    for rows, paged, compressed in ((7, False, False), (77, True, True), (25, True, False)):
        args = operands(rows, paged, compressed)
        q, pos, cs, sink, kv, indices, lengths, extra, ci, cl = args
        out, sf = _outputs(rows, q.device)
        expected = reference(args, *_outputs(rows, q.device))
        stats = [torch.full((rows, 64), float("nan"), device="cuda") for _ in expected[2:]]
        if paged:
            module.decode(q, kv, indices, lengths, sink, pos, cs, extra, ci, cl, out, sf, *stats, 512 ** -.5)
        else:
            module.prefill(q, kv, indices, lengths, sink, pos, cs, out, sf, *stats, 512 ** -.5)
        torch.cuda.synchronize()
        for actual, wanted in zip(stats, expected[2:]):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0, equal_nan=True)


def test_rejects_changed_geometry_and_preparation_during_capture(binding):
    _, cache, library = binding
    args = list(operands(77, True, True))
    args[5] = args[5][:, :127].contiguous()
    with pytest.raises(ValueError, match="divisible by eight"):
        PreparedFlashMLA(*args, scale=512 ** -.5, kernel_cache=cache, library=library,
                        session=None, name="invalid")
    graph = torch.cuda.CUDAGraph()
    value = torch.zeros(1, device="cuda")
    with torch.cuda.graph(graph):
        value.add_(1)
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            prepare(cache, library)
    graph.reset()
