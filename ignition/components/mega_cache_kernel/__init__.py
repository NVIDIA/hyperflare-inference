# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation build for the reference paged MXFP8 rotary-cache writer."""

import hashlib
from pathlib import Path
from threading import Lock

_library = None
_lock = Lock()


def prepare(kernel_cache: Path):
    """Load the CUDA writer once before capture; retain it through model drain."""
    import torch
    if torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("the reference cache writer build requires SM103")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("the cache writer must be prepared before capture")
    global _library
    with _lock:
        if _library is None:
            from torch.utils.cpp_extension import load

            root = Path(__file__).parent
            source = root / "csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"
            digest = hashlib.sha256()
            for path in sorted((*root.rglob("*.cu"), *root.rglob("*.h"),
                                *root.rglob("*.cuh"), *root.glob("*.cpp"))):
                digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
            digest.update((torch.__version__ + str(torch.version.cuda)
                           + str(torch._C._GLIBCXX_USE_CXX11_ABI)).encode())
            build = Path(kernel_cache) / "mega-cache" / digest.hexdigest()
            build.mkdir(parents=True, exist_ok=True)
            _library = load(name="flare_reference_cache", sources=[str(root / "bindings.cpp"), str(source),
                str(root / "csrc/indexer_cache_gather.cu")],
                extra_include_paths=[str(root / "csrc")],
                extra_cuda_cflags=["-DUSE_CUDA", "-O3", "-std=c++20",
                                  "-gencode=arch=compute_103a,code=sm_103a"],
                extra_cflags=["-DUSE_CUDA", "-O3", "-std=c++20"],
                build_directory=str(build), is_python_module=False, verbose=False)
    return torch.ops.flare_reference_cache.rope_insert


def prepare_gather(kernel_cache: Path):
    """Prepare the lossless paged index-key gather alongside cache insertion."""
    prepare(kernel_cache)
    import torch
    return torch.ops.flare_reference_cache.indexer_gather
