# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Time Mini experts using checkpoint weights and synthetic activations.

Loads one layer at a time and captures only its expert operation. CUDA event
timing includes route packing, activation quantization, both projections,
route-weighted SiTU, and the expert sum. Routing decisions are prepared outside
timing. The default uses direct Triton launches and captured PyTorch reductions;
--compile additionally compiles this component with Inductor. Neither mode
prepares a model catalog or exports a core. Service gains require native serving
measurements after export.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 20])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 32, 128, 8192])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--reference", type=Path,
                        help="also check each selected layer/shape against inkling2_mini_reference.py fixtures")
    parser.add_argument("--compile", action="store_true", help="compile only the expert forward with Inductor")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", type=Path, help="write JSONL to a file that does not already exist")
    args = parser.parse_args()
    if any(not 0 <= layer < 24 for layer in args.layers):
        parser.error("layers must be in [0, 24)")
    if min(*args.rows, args.warmup, args.samples) < 1:
        parser.error("rows, warmup, and samples must be positive")
    if any(not 0 <= seed < 2**63 for seed in args.seeds):
        parser.error("seeds must be in [0, 2**63)")
    if any(len(values) != len(set(values)) for values in (args.layers, args.rows, args.seeds)):
        parser.error("layers, rows, and seeds must each be distinct")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")

    from ignition.cli.config import Configuration
    from ignition.cli.environment import apply
    from ignition.cli.preparation import Host

    available = {gpu.index: gpu for gpu in Host.discover().gpus}
    if args.gpu not in available:
        parser.error(f"GPU {args.gpu} is not visible")
    apply(Configuration.load(args.config), (available[args.gpu],))

    import torch
    from flashinfer.testing.utils import bench_gpu_time_with_cuda_event
    from safetensors.torch import load_file
    from ignition.components.parallel import TensorParallelShard
    from ignition.components.situ_experts import PreparedSiTUExperts
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.inkling2.mini_experts import load_experts
    from ignition.recipes.inkling2.mini_weights import MiniWeightLoader

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        parser.error("this benchmark requires SM120 or SM121")
    manifest = inspect_indexed_checkpoint(args.snapshot)
    reference = {} if args.reference is None else load_file(args.reference)
    if args.reference is not None:
        for layer in args.layers:
            for rows in args.rows:
                for role in ("input", "ids", "routes", "output"):
                    key = f"layer{layer}.{rows}.{role}"
                    if key not in reference:
                        parser.error(f"reference is missing {key}; generate fixtures with matching --rows")
    output_file = None if args.output is None else args.output.open("x")

    def emit(record):
        line = json.dumps(record)
        print(line, flush=True)
        if output_file is not None:
            output_file.write(line + "\n")
            output_file.flush()

    emit(dict(kind="configuration", snapshot=str(manifest.snapshot),
        shard_headers_sha256={shard.path: shard.header_sha256 for shard in manifest.shards},
        metadata_sha256={name: hashlib.sha256((manifest.snapshot / name).read_bytes()).hexdigest()
                         for name in ("config.json", "hf_quant_config.json")},
        source_sha256={name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                       for name in ("scripts/benchmark_inkling2_mini_experts.py",
                                    "ignition/components/situ_experts.py",
                                    "ignition/components/four_six_quantization.py")},
        gpu=available[args.gpu].uuid, device=torch.cuda.get_device_name(device),
        torch=torch.__version__, layers=args.layers, rows=args.rows, seeds=args.seeds,
        compiled=args.compile, reference=None if args.reference is None else str(args.reference),
        activation_distribution="Gaussian(mean=0,std=1), BF16; checkpoint sigmoid/top-k router",
        timing="CUDA events around expert graph replay; L2 flushed before each sample; routing excluded",
        warmup=args.warmup, samples=args.samples))
    try:
        with torch.no_grad():
            for layer in args.layers:
                load_started = time.perf_counter()
                shard = TensorParallelShard(0, 1)
                with MiniWeightLoader(manifest, device, shard=shard, vision=False) as loader:
                    weights = load_experts(loader, layer, shard=shard)
                    text = loader.config.text
                    prefix = f"model.llm.layers.{layer}.mlp.gate."
                    gate = loader.load_tensor(prefix + "weight")
                    bias = loader.load_tensor(prefix + "bias")
                torch.cuda.synchronize()
                emit(dict(kind="layer", layer=layer, precision="nvfp4" if weights.nvfp4 else "mxfp8",
                          load_seconds=time.perf_counter() - load_started))
                for rows in args.rows:
                    prepare_started = time.perf_counter()
                    operation = PreparedSiTUExperts(weights, tokens=rows, top_k=text.experts_per_token)
                    source = torch.zeros((rows, text.hidden), dtype=torch.bfloat16, device=device)
                    ids = torch.zeros((rows, text.experts_per_token), dtype=torch.int32, device=device)
                    routes = torch.ones(ids.shape, dtype=torch.float32, device=device)
                    forward = lambda: operation(source, ids, routes)
                    if args.compile:
                        forward = compile_forward(forward)
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        forward()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    try:
                        with torch.cuda.graph(graph, stream=stream):
                            output = forward()
                        torch.cuda.synchronize()
                        emit(dict(kind="prepared", layer=layer, rows=rows,
                                  seconds=time.perf_counter() - prepare_started))
                        if reference:
                            prefix = f"layer{layer}.{rows}."
                            source.copy_(reference[prefix + "input"])
                            ids.copy_(reference[prefix + "ids"])
                            routes.copy_(reference[prefix + "routes"].reshape_as(routes))
                            graph.replay()
                            expected = reference[prefix + "output"].to(device)
                            relative_l2 = ((output.float() - expected.float()).norm()
                                           / expected.float().norm().clamp_min(1e-30)).item()
                            if not torch.isfinite(output).all().item() or relative_l2 > 1e-6:
                                raise RuntimeError(f"vLLM reference mismatch: layer={layer}, rows={rows}, relative_l2={relative_l2}")
                            emit(dict(kind="reference", layer=layer, rows=rows, relative_l2=relative_l2,
                                      max_absolute_error=(output.float() - expected.float()).abs().max().item()))
                            del expected
                        for seed in args.seeds:
                            generator = torch.Generator().manual_seed(seed)
                            source.copy_(torch.randn(source.shape, generator=generator).to(torch.bfloat16))
                            scores = torch.nn.functional.linear(source.to(gate.dtype), gate).float().sigmoid()
                            selected = (scores + bias.float()).topk(text.experts_per_token, dim=-1).indices
                            selected_scores = scores.gather(1, selected)
                            ids.copy_(selected)
                            routes.copy_(selected_scores / selected_scores.sum(-1, keepdim=True) * text.route_scale)
                            expected = operation(source, ids, routes)
                            output.fill_(float("nan"))
                            torch.cuda.synchronize()
                            before = torch.cuda.memory_stats(device)
                            for _ in range(3):
                                graph.replay()
                            torch.cuda.synchronize()
                            after = torch.cuda.memory_stats(device)
                            for key in ("allocation.all.allocated", "allocation.all.freed",
                                        "segment.all.allocated", "segment.all.freed"):
                                if before[key] != after[key]:
                                    raise RuntimeError(f"expert replay changed CUDA allocator counter {key}")
                            if not torch.isfinite(output).all().item():
                                raise RuntimeError(f"non-finite expert output: layer={layer}, rows={rows}, seed={seed}")
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                            with torch.cuda.nvtx.range(f"mini_experts.layer{layer}.rows{rows}.seed{seed}"):
                                times = bench_gpu_time_with_cuda_event(graph.replay,
                                    dry_run_iters=args.warmup, repeat_iters=args.samples, cold_l2_cache=True)
                            torch.testing.assert_close(output, expected, rtol=0, atol=0)
                            emit(dict(kind="measurement", layer=layer, rows=rows, seed=seed,
                                unique_routed_experts=ids.unique().numel(), milliseconds=times,
                                median_ms=statistics.median(times)))
                            del expected, scores, selected, selected_scores
                    finally:
                        torch.cuda.synchronize()
                        graph.reset()
                    del graph, forward, operation, source, ids, routes, output
                del weights, gate, bias
                torch.cuda.empty_cache()
        emit(dict(kind="complete", seconds=time.perf_counter() - started))
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
