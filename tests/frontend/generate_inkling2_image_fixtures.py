# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate padded-grid reference tensors with Transformers' Inkling 2 processor.

Run with the preparation environment and the checkpoint configuration:

    python tests/frontend/generate_inkling2_image_fixtures.py --snapshot /data/models/Inkling-2-Small-BF16

The fixtures cover unresized cell boundaries, RGB and transparency modes, stored
orientation, static WebP, JPEG decoding, and packed image batches. Binary files
contain ordinary little-endian tensor arrays. JPEG pixel comparisons allow the
maximum absolute channel error of two; other arrays compare exactly.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.models.inkling2 import modeling_inkling2
from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor


def pattern(width, height, seed):
    state = seed
    pixels = bytearray(width * height * 3)
    for index in range(len(pixels)):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        pixels[index] = state & 255
    return np.frombuffer(pixels, np.uint8).reshape(height, width, 3)


class StopBeforeAttention(Exception):
    pass


def vision_inputs(model, pixels, grid):
    recorded = {}

    def patches(module, args):
        recorded["pixel_values"] = args[0].detach().clone()

    def positions(module, args):
        recorded["rotary_positions"] = args[1].detach().clone()

    def attention(module, args, kwargs):
        recorded["cu_seqlens"] = kwargs["cu_seqlens"].detach().clone()
        raise StopBeforeAttention

    handles = [model.embeddings.patch_embedding.register_forward_pre_hook(patches),
               model.rotary_emb.register_forward_pre_hook(positions),
               model.layers[0].register_forward_pre_hook(attention, with_kwargs=True)]
    try:
        with torch.no_grad():
            model(pixel_values=pixels, image_grid_thw=grid)
    except StopBeforeAttention:
        pass
    finally:
        for handle in handles:
            handle.remove()
    return recorded


def generate(snapshot, output):
    torch.set_num_threads(1)
    processor_config = json.loads((snapshot / "processor_config.json").read_text())
    model_config = json.loads((snapshot / "config.json").read_text())
    image_settings = dict(processor_config["image_processor"])
    image_settings.pop("image_processor_type")
    processor = Inkling2ImageProcessor(**image_settings)
    vision = model_config["vision_config"]
    config = Inkling2VisionConfig(patch_size=vision["native_patch_size"],
        spatial_merge_size=vision["adapter_merge_size"], posemb_grid_size=vision["posemb_grid_size"],
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, projection_hidden_dim=8, text_hidden_size=8)
    model = modeling_inkling2.Inkling2VisionModel(config).eval()
    tensors = output.parent / "inkling2_images-tensors"
    tensors.mkdir(exist_ok=True)

    def store(name, value, dtype):
        array = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        path = tensors / f"{name}.{dtype}"
        path.write_bytes(np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<")).tobytes())
        return str(path.relative_to(output.parent))

    sources = {}
    for name, width, height, seed in [("one_pixel", 1, 1, 1), ("one_cell", 40, 40, 2),
            ("cell_plus_one", 41, 41, 3), ("below_cell", 39, 17, 4), ("non_multiple", 53, 71, 5),
            ("landscape", 83, 41, 6), ("wide", 100, 3, 7), ("tall", 2, 81, 8)]:
        sources[name] = ({"pattern": dict(width=width, height=height, seed=seed)},
                         Image.fromarray(pattern(width, height, seed)))
    for path in sorted((output.parent / "inkling2_images-inputs").iterdir()):
        image = Image.open(path)
        image.load()
        name = path.stem.replace("-", "_")
        descriptor = dict(file=str(path.relative_to(output.parent)), format=image.format,
                          width=image.width, height=image.height,
                          max_channel_error=2 if image.format == "JPEG" else 0)
        sources[name] = descriptor, image

    cases = []
    batches = [[name] for name in sources] + [["non_multiple", "one_pixel", "wide"],
        ["one_cell", "one_cell"], ["RGBA", "CMYK", "RGBA_lossless"]]
    for names in batches:
        name = "__".join(names)
        images = [sources[key][1] for key in names]
        result = processor(images=images, return_tensors="pt")
        grid = result["image_grid_thw"]
        native_grid = grid * torch.tensor([1, 2, 2])
        recorded = vision_inputs(model, result["pixel_values"], grid)
        positions = modeling_inkling2.get_vision_position_ids(native_grid, 2)
        indices, weights = modeling_inkling2.get_vision_interpolation_indices_and_weights(
            native_grid, 32, spatial_merge_size=2)
        composites = [processor.process_image(image, do_convert_rgb=True).permute(1, 2, 0).numpy()
                      for image in images]
        cases.append(dict(name=name, inputs=[sources[key][0] for key in names],
            grid_thw=grid.tolist(), tokens=grid.prod(-1).tolist(),
            cu_seqlens=recorded["cu_seqlens"].tolist(),
            composited_rgb=[store(f"{name}_image{i}", rgb, "uint8") for i, rgb in enumerate(composites)],
            pixel_values=store(name, recorded["pixel_values"], "float32"),
            position_ids=positions.tolist(), rotary_positions=recorded["rotary_positions"].tolist(),
            interpolation_indices=indices.tolist(), interpolation_weights=weights.tolist()))
    data = dict(reference={"transformers": importlib.metadata.version("transformers"),
                          "torch": torch.__version__, "processor": "Inkling2ImageProcessor"},
                processor=processor_config, model={"model_type": model_config["model_type"], "vision_config": {key: vision[key] for key in ("native_patch_size", "adapter_merge_size", "posemb_grid_size", "temporal_patch_size")}},
                cases=cases)
    output.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(cases)} image cases to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("inkling2_images.json"))
    args = parser.parse_args()
    generate(args.snapshot, args.output)
