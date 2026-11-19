# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation binding for the unchanged ds41 SM103 fused decode attention."""

from functools import cache
import hashlib
from pathlib import Path


def initialize(kernel_cache: Path):
    """Build the source with Ignition's Torch ABI before compiling or capturing forwards.

    The caller retains this module until captured work drains. Exported cores
    retain its CUDA kernels and do not load this Python extension when serving.
    """
    import torch
    if torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("the reference fused attention build requires an SM103 device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("fused attention must be prepared before CUDA graph capture")
    return _build(Path(kernel_cache).resolve())


@cache
def _build(kernel_cache: Path):
    import torch
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted((*root.rglob("*.cuh"), *root.rglob("*.cu"))):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    digest.update((torch.__version__ + str(torch.version.cuda)).encode())
    build = Path(kernel_cache) / "mega-attention" / digest.hexdigest()
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="flare_mega_attention", sources=[str(root / "csrc/w1_attn.cu")],
        extra_include_paths=[str(root / "include")],
        extra_cuda_cflags=[
            "-O3", "-std=c++20", "-lineinfo",
            "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        build_directory=str(build), verbose=False,
    )


def prepare(kernel_cache: Path):
    """Configure shared-memory limits on the current device before capture."""
    module = initialize(kernel_cache)
    module.w1_attn_prepare()
    return module
