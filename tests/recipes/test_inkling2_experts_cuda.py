# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-backed A4/A16 SiTU expert graph replay with one NVFP4 weight owner."""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@torch.no_grad()
def test_checkpoint_tp4_experts_share_a4_a16_storage_and_replay():
    from b12x.preparation import PreparationSession
    from b12x.moe._shared.kernels.w4a16.host import unswizzle_block_scale
    from ignition import runtime_patches
    from ignition.components.moe import PreparedRoutedExperts
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.inkling2.experts import load_experts
    from ignition.reference.qwen38_quant import CheckpointTensors, decode_nvfp4, quantize_nvfp4_activation

    snapshot = os.getenv("FLARE_INKLING2_SNAPSHOT")
    if snapshot is None:
        pytest.skip("requires FLARE_INKLING2_SNAPSHOT")
    runtime_patches.apply()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    manifest = inspect_indexed_checkpoint(Path(snapshot))
    root = "model.language_model.layers.0.mlp.experts"
    with DirectWeightLoader(manifest, device, prefixes=(root + ".",)) as loader:
        experts = load_experts(loader, 0, shard=TensorParallelShard(0, 4))
        assert loader._session.torch_copy_bytes == 0
    assert experts.plan._impl.quant_modes == {"nvfp4", "w4a16"}
    native = experts._impl.representation_for("w4a16")
    for tensor, canonical in ((native.w13, experts._impl.w1_fp4), (native.w2, experts._impl.w2_fp4),
                              (native.w13_scale, experts._impl.w1_blockscale),
                              (native.w2_scale, experts._impl.w2_blockscale)):
        assert tensor.data_ptr() == canonical.data_ptr()
    selected = (0, 1, 17, 63, 127, 255, 382, 383)
    decoded = []
    with CheckpointTensors(manifest, byte_budget=192 << 20) as reader:
        for expert in selected:
            projections, input_scales = [], []
            canonical_scales = (
                unswizzle_block_scale(experts._impl.w1_blockscale[expert].cpu(), 1152, 288),
                unswizzle_block_scale(experts._impl.w2_blockscale[expert].cpu(), 4608, 36),
            )
            for projection in ("up_proj", "gate_proj", "down_proj"):
                prefix = f"{root}.{expert}.{projection}"
                rows = None if projection == "down_proj" else (0, 576)
                packed = reader.read(prefix + ".weight", rows=rows)
                scales = reader.read(prefix + ".weight_scale", rows=rows)
                if projection == "down_proj":
                    packed, scales = packed[:, :288], scales[:, :36]
                    expected_packed = experts._impl.w2_fp4[expert]
                    expected_scales = canonical_scales[1]
                else:
                    first = 0 if projection == "up_proj" else 576
                    expected_packed = experts._impl.w1_fp4[expert, first:first + 576]
                    expected_scales = canonical_scales[0][first:first + 576]
                assert torch.equal(packed, expected_packed.cpu())
                assert torch.equal(scales.float(), expected_scales)
                projections.append(decode_nvfp4(packed, scales, reader.read(prefix + ".weight_scale_2")))
                input_scales.append(reader.read(prefix + ".input_scale"))
            decoded.append((projections, input_scales))
    generator = torch.Generator().manual_seed(791)
    cpu = (torch.randn(4096, 4608, generator=generator) * .125).bfloat16()
    source = cpu.to(device)
    ids_cpu = torch.stack([torch.randperm(384, generator=generator)[:8] for _ in range(4096)]).int()
    reference_rows = (tuple(range(8)), (0, 31, 63), (0, 2047, 4095))
    ids_cpu[sorted({row for rows in reference_rows for row in rows})] = torch.tensor(selected, dtype=torch.int32)
    ids = ids_cpu.to(device)
    weights_cpu = torch.softmax(torch.randn(4096, 8, generator=generator), -1)
    weights = weights_cpu.to(device)
    cases = {(8, "a4"): reference_rows[0], (8, "a16"): reference_rows[0],
             (64, "a16"): reference_rows[1], (4096, "a4"): reference_rows[2]}
    outputs = {case: torch.empty_like(source[:case[0]]) for case in cases}
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        operations = {case: PreparedRoutedExperts(source[:case[0]], experts, ids[:case[0]], weights[:case[0]], output,
                      session=session, name=f"inkling2-{case[0]}-{case[1]}", activation_mode=case[1])
                      for case, output in outputs.items()}
        for (tokens, mode), operation in operations.items():
            assert operation.plan.query.quant_mode == ("nvfp4" if mode == "a4" else "w4a16")
            assert operation.experts is experts
            print(f"tokens={tokens} {mode} backend={operation.plan.selection.config.backend}")
        graphs = {}
        for case, operation in operations.items():
            tokens, mode = case
            operation()
            operation.functional(session=session)
            operation(source[:tokens], ids[:tokens], weights[:tokens])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[case] = operation(source[:tokens], ids[:tokens], weights[:tokens])
            assert outputs[case].dtype == torch.bfloat16
            graphs[case] = graph
        for factor in (1., -.5):
            live = (cpu * factor).bfloat16()
            source.copy_(live)
            for case, graph in graphs.items():
                tokens, mode = case
                outputs[case].fill_(float("nan"))
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                rows = list(cases[case])
                expected = torch.zeros(len(rows), 4608)
                for route, (projections, scales) in enumerate(decoded):
                    up, gate, down = projections
                    operand = live[rows].float()
                    if mode == "a4":
                        operand = quantize_nvfp4_activation(operand, scales[0])
                    up_values, gate_values = operand @ up.T, operand @ gate.T
                    if mode == "a16":
                        up_values, gate_values = up_values.bfloat16().float(), gate_values.bfloat16().float()
                    intermediate = (4 * torch.tanh(gate_values / 4) * torch.sigmoid(gate_values)
                                    * (25 * torch.tanh(up_values / 25))).bfloat16().float()
                    if mode == "a4":
                        intermediate = quantize_nvfp4_activation(intermediate, scales[2])
                    expected += (intermediate @ down.T) * weights_cpu[rows, route, None]
                assert torch.isfinite(outputs[case]).all()
                actual = outputs[case][rows].float().cpu()
                relative = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
                cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=-1).mean()
                print(f"tokens={tokens} {mode} factor={factor:g} relative_l2={float(relative):.8f} "
                      f"cosine={float(cosine):.8f} max_abs={float((actual - expected).abs().max()):.8f}")
                assert relative < (.035 if mode == "a4" else .015), (mode, factor, float(relative))
                assert cosine >= .9975, (mode, factor, float(cosine))
