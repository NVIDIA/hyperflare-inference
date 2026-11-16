# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Paged cache insertion preserves inactive slots, compression boundaries and padding."""

import os
import weakref

import pytest
import torch

from ignition.components.mega_attention_cache import MegaCacheLayout, PreparedMegaCacheWriter
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def expected_cache(backing, values, positions, slots, layout, ratio):
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference

    # Identity rotary table makes the independently encoded source exact. The
    # kernel's non-identity rotation is checked against the pinned vLLM build.
    result = backing.cpu().clone()
    records = pack_deepseek_v41_cache_reference(values.cpu(), page_size=1,
        cache_kind="swa" if layout.record_bytes == 528 else "indexed")
    for row, (position, slot) in enumerate(zip(positions.cpu().tolist(), slots.cpu().tolist())):
        if slot < 0 or (layout.record_bytes == 288 and (position + 1) % ratio):
            continue
        page, offset = divmod(slot, layout.page_tokens)
        data = offset * layout.payload_bytes
        scales = layout.page_tokens * layout.payload_bytes + offset * layout.scale_bytes
        result[page, data:data + layout.payload_bytes] = records[row, :layout.payload_bytes]
        result[page, scales:scales + layout.scale_bytes] = records[row, layout.payload_bytes:]
    return result


def test_compiled_writers_preserve_layout_and_unwritten_state(tmp_path):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    generator = torch.Generator(device="cuda").manual_seed(8321)
    cs = torch.cat((torch.ones((4096, 32), device="cuda"), torch.zeros((4096, 32), device="cuda")), -1)
    operations, operands, functions, graphs = {}, {}, {}, {}
    declarations = []
    cases = ((1, 128, 528, 1), (17, 136, 528, 1), (257, 128, 528, 1),
             (1, 64, 288, 1), (17, 136, 288, 1), (257, 64, 288, 2))
    with RecordedPreparationSession(device="cuda:0", checkpoint="synthetic-cache",
            workload_schema="sm103-page-separated", numerical_policy="exact-encoded-bytes",
            compile_workers=2, autotune_effort="off") as session:
        for rows, page, width, ratio in cases:
            layout = MegaCacheLayout(page, width)
            pages = (2 * rows + page - 1) // page + 1
            backing = torch.randint(0, 256, (pages, layout.page_bytes), device="cuda", dtype=torch.uint8, generator=generator)
            cache = layout.view(backing)
            values = torch.randn((rows, 512), device="cuda", dtype=torch.bfloat16, generator=generator) * .2
            positions = torch.arange(rows, device="cuda", dtype=torch.int64)
            slots = torch.arange(rows, device="cuda", dtype=torch.int64) * 2 + 1
            if rows > 1:
                slots[::7] = -1
            key = rows, page, width, ratio
            operands[key] = values, positions, cs, cache, slots, backing, layout
            examples = values.clone(), positions.clone(), cs.clone(), cache, slots.clone()
            declarations.extend(weakref.ref(tensor) for tensor in (examples[0], examples[1], examples[2], examples[4]))
            operations[key] = PreparedMegaCacheWriter(*examples, ratio=ratio, kernel_cache=tmp_path,
                                                      session=session, name=f"write-{key}")
        del examples
        session.tune()
        assert all(reference() is None for reference in declarations)
        for key, operation in operations.items():
            args = operands[key][:5]
            functions[key] = compile_forward(lambda args=args, operation=operation: operation(*args))
            functions[key]()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                functions[key]()
            graphs[key] = graph
        session.freeze()
        for step in range(3):
            for key in (*reversed(cases), *cases):
                values, positions, cos_sin, cache, slots, backing, layout = operands[key]
                values.neg_()
                positions.add_(1)
                if step == 2:
                    slots.fill_(-1)
                reference = expected_cache(backing, values, positions, slots, layout, key[-1])
                saved_inputs = [tensor.clone() for tensor in (values, positions, cos_sin, slots)]
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()["allocation.all.allocated"]
                graphs[key].replay()
                torch.cuda.synchronize()
                assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
                torch.testing.assert_close(backing.cpu(), reference, rtol=0, atol=0)
                for actual, saved in zip((values, positions, cos_sin, slots), saved_inputs):
                    torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        args = operands[(17, 136, 528, 1)][:5]
        with pytest.raises(ValueError, match="prepared contiguous"):
            operations[(17, 136, 528, 1)](args[0][:1], *args[1:])
        for graph in graphs.values():
            graph.reset()


def test_cache_layout_requires_page_padding():
    layout = MegaCacheLayout(136, 528)
    assert layout.page_bytes == 72192
    backing = torch.empty((2, layout.page_bytes), dtype=torch.uint8)
    cache = layout.view(backing)
    assert cache.shape == (2, 136, 528)
    assert cache.stride() == (72192, 528, 1)
    layout.validate(cache)
    with pytest.raises(ValueError, match="aligned byte row"):
        layout.view(backing[:, :136 * 528])
    with pytest.raises(ValueError, match="page and scale layout"):
        layout.validate(torch.empty((2, 136, 528), dtype=torch.uint8))


def test_window_snapshot_and_partial_commit_preserve_complete_ring():
    from ignition.components.mega_attention_cache import snapshot_window, commit_window

    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    generator = torch.Generator(device="cuda").manual_seed(1112)
    history_layout = MegaCacheLayout(128, 528)
    history_bytes = torch.randint(0, 256, (5, history_layout.page_bytes), dtype=torch.uint8,
                                 device="cuda", generator=generator)
    history = history_layout.view(history_bytes)
    rows = torch.tensor([3, 0, 1], dtype=torch.int64, device="cuda")
    starts = torch.tensor([126, 20, 5], dtype=torch.int64, device="cuda")
    counts = torch.tensor([1, 0, 0], dtype=torch.int64, device="cuda")
    graphs, buffers, functions = {}, {}, {}
    for suffix in (8, 144):
        layout = MegaCacheLayout(128 + suffix, 528)
        backing = torch.randint(0, 256, (3, layout.page_bytes), dtype=torch.uint8,
                                device="cuda", generator=generator)
        staged = layout.view(backing)
        buffers[suffix] = backing, staged

        def forward(staged=staged):
            snapshot_window(history, staged, rows)
            commit_window(staged, history, rows, starts, counts)

        functions[suffix] = compile_forward(forward)
        functions[suffix]()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            functions[suffix]()
        graphs[suffix] = graph
    for suffix, accepted in ((8, 7), (144, 140), (8, 0), (144, 1), (8, 8)):
        backing, staged = buffers[suffix]
        backing.random_(0, 256, generator=generator)
        counts.copy_(torch.tensor([1, 0, accepted], dtype=torch.int64, device="cuda"))
        expected_history, expected_staged = history_bytes.cpu(), backing.cpu()
        state_ids, offsets, valid = rows.cpu().tolist(), starts.cpu().tolist(), counts.cpu().tolist()
        for request, (row, start, count) in enumerate(zip(state_ids, offsets, valid)):
            if row == 0:
                continue
            expected_staged[request, :128 * 512] = expected_history[row, :128 * 512]
            expected_staged[request, (128 + suffix) * 512:(128 + suffix) * 512 + 128 * 16] = expected_history[row, 128 * 512:]
            for token in range(max(0, count - 128), count):
                ring = (start + token) % 128
                source_data, target_data = (128 + token) * 512, ring * 512
                source_scale = (128 + suffix) * 512 + (128 + token) * 16
                target_scale = 128 * 512 + ring * 16
                expected_history[row, target_data:target_data + 512] = expected_staged[request, source_data:source_data + 512]
                expected_history[row, target_scale:target_scale + 16] = expected_staged[request, source_scale:source_scale + 16]
        torch.cuda.synchronize()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        graphs[suffix].replay()
        torch.cuda.synchronize()
        assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
        torch.testing.assert_close(history_bytes.cpu(), expected_history, rtol=0, atol=0)
        torch.testing.assert_close(backing.cpu(), expected_staged, rtol=0, atol=0)
    for graph in graphs.values():
        graph.reset()


@pytest.mark.parametrize("rows", [1, 17])
def test_cache_writers_feed_compiled_fused_attention(tmp_path, rows):
    from ignition.components.mega_attention import MegaAttentionWorkspace, PreparedMegaAttention, _outputs

    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    library = os.environ.get("FLARE_FLASHMLA_LIBRARY")
    if not library:
        pytest.skip("requires the pinned FlashMLA reference library")
    torch.ops.load_library(library)
    generator = torch.Generator(device="cuda").manual_seed(1221)

    def randn(shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator) * .2

    wl, cl = MegaCacheLayout(128, 528), MegaCacheLayout(64, 288)
    window = wl.view(torch.zeros((4, wl.page_bytes), device="cuda", dtype=torch.uint8))
    compressed = cl.view(torch.zeros((8, cl.page_bytes), device="cuda", dtype=torch.uint8))
    values, latent, query = randn((rows, 512)), randn((rows, 512)), randn((rows, 64, 512))
    positions = torch.arange(256, 256 + rows, device="cuda", dtype=torch.int64)
    slots = torch.arange(128, 128 + rows, device="cuda", dtype=torch.int64)
    compressed_slots = slots - 128
    angles = torch.arange(4096, device="cuda").float()[:, None] / (
        10000 ** (torch.arange(32, device="cuda").float() / 32))[None, :]
    cs = torch.cat((angles.cos(), angles.sin()), -1)
    sink = randn((64,)).float()
    wi = torch.arange(128, 256, device="cuda", dtype=torch.int32).repeat(rows, 1)
    ci = torch.arange(512, device="cuda", dtype=torch.int32).repeat(rows, 1)
    window_lengths = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
    compressed_lengths = torch.full((rows,), 512, device="cuda", dtype=torch.int32)
    out, sf = _outputs(rows, query.device)
    expected, expected_sf = _outputs(rows, query.device)
    workspace = MegaAttentionWorkspace(rows, device="cuda:0")
    with RecordedPreparationSession(device="cuda:0", checkpoint="synthetic-cache-attention",
            workload_schema="sm103-writer-reader", numerical_policy="flashmla-bitwise",
            compile_workers=2, autotune_effort="off") as session:
        window_writer = PreparedMegaCacheWriter(values, positions, cs, window, slots, ratio=1,
                                                kernel_cache=tmp_path, session=session, name="window-write")
        compressed_writer = PreparedMegaCacheWriter(latent, positions, cs, compressed, compressed_slots, ratio=1,
                                                    kernel_cache=tmp_path, session=session, name="compressed-write")
        attention = PreparedMegaAttention(query, positions.int(), cs, sink, window, wi, window_lengths,
            compressed, ci, compressed_lengths, workspace=workspace, scale=512 ** -.5,
            kernel_cache=tmp_path, session=session, name="attention")
        session.tune()

        def forward():
            window_writer(values, positions, cs, window, slots)
            compressed_writer(latent, positions, cs, compressed, compressed_slots)
            result, scales = attention(query, positions.int(), cs, sink, window, wi, window_lengths,
                                       compressed, ci, compressed_lengths)
            out.copy_(result)
            sf.copy_(scales)

        function = compile_forward(forward)
        function()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            function()
        session.freeze()
        for _ in range(3):
            positions.add_(1)
            values.neg_()
            latent.neg_()
            out.view(torch.uint8).fill_(255)
            sf.fill_(-99)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert before == torch.cuda.memory_stats()["allocation.all.allocated"]
            torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
                query, window.unsqueeze(-2), wi, 512 ** -.5, 512, sink, window_lengths,
                compressed.unsqueeze(-2), ci, compressed_lengths, False, 0., positions.int(),
                False, 64, cs, 8, 32, True, True, True, expected, expected_sf)
            torch.testing.assert_close(out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(sf, expected_sf, rtol=0, atol=0)
        graph.reset()
