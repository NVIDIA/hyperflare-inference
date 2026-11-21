# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Write independent vLLM expert outputs for Mini checkpoint qualification.

Run in the reference vLLM environment, with its Inkling2 SM121 implementation.
The output contains synthetic inputs, selected routes, and complete expert sums.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file


@torch.no_grad()
def main():
    from vllm.models.inkling2.nvidia.moe import Inkling2MoE
    from vllm.models.inkling2.nvidia.ops.moe_sm90 import select_experts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 32, 128, 4096, 8192])
    args = parser.parse_args()
    if any(rows < 1 for rows in args.rows) or len(set(args.rows)) != len(args.rows):
        parser.error("rows must be positive and distinct")
    torch.set_num_threads(4)
    torch.manual_seed(917)
    config = SimpleNamespace(**json.loads((args.checkpoint / "config.json").read_text())["text_config"])
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for layer in (0, 20):
        prefix = f"model.llm.layers.{layer}.mlp."
        weights = {}
        for name, file in index.items():
            if name.startswith(prefix):
                with safe_open(args.checkpoint / file, framework="pt", device="cpu") as handle:
                    weights[name.removeprefix(prefix)] = handle.get_tensor(name).cuda()
        fmt = "nvfp4" if weights["experts.w13_weight"].dtype == torch.uint8 else "mxfp8"
        with patch("vllm.models.inkling2.nvidia.moe.get_tensor_model_parallel_rank", return_value=0), patch(
                "vllm.models.inkling2.nvidia.moe.get_tensor_model_parallel_world_size", return_value=1), torch.device("cuda"):
            model = Inkling2MoE(config, fmt)
        for name, param in model.named_parameters():
            param.weight_loader(param, weights[name])
        model.experts.process_weights_after_loading()
        for tokens in args.rows:
            x = torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
            ids, gammas, _ = select_experts(x, model.gate.weight, model.gate.bias, config.num_experts_per_tok,
                                          config.route_scale, model.experts.expert_bounds)
            y = model(x)
            for role, tensor in (("input", x), ("ids", ids), ("routes", gammas), ("output", y)):
                tensors[f"layer{layer}.{tokens}.{role}"] = tensor.cpu().contiguous()
            print(f"layer={layer} tokens={tokens} finite={bool(y.isfinite().all())}", flush=True)
        del model, weights
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)


if __name__ == "__main__":
    main()
