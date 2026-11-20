# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture Mini image features using the independent vLLM vision implementation."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors import safe_open
from safetensors.torch import save_file


@torch.no_grad()
def main():
    from vllm.models.inkling2.nvidia.vision import Inkling2VisionEncoder
    from vllm.models.inkling2.processing import Inkling2ImageProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    raw = json.loads((args.checkpoint / "config.json").read_text())
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.device("cuda"):
        encoder = Inkling2VisionEncoder(raw["vision_config"], raw["text_config"]["hidden_size"])
    weights = []
    for name, file in index.items():
        if name.startswith("model.visual."):
            with safe_open(args.checkpoint / file, framework="pt", device="cpu") as handle:
                weights.append((name.removeprefix("model.visual."), handle.get_tensor(name)))
    loaded = encoder.load_weights(weights)
    assert loaded == set(dict(encoder.named_parameters()))
    result = {}
    for variant, (width, height) in enumerate(((1, 1), (41, 79), (160, 160), (320, 320), (640, 320))):
        pixels = bytes(component for y in range(height) for x in range(width)
                       for component in ((x * 7 + variant * 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
        picture = Image.frombytes("RGB", (width, height), pixels)
        inputs = Inkling2ImageProcessor().preprocess([picture])
        output = encoder(inputs["vision_pixels"].cuda(), inputs["vision_grid_thw"])
        key = f"{width}x{height}"
        result[f"{key}.image"] = torch.from_numpy(np.array(picture))
        result[f"{key}.features"] = output.cpu().contiguous()
        print(f"{key}: {tuple(output.shape)}, finite={bool(output.isfinite().all())}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, args.output)


if __name__ == "__main__":
    main()
