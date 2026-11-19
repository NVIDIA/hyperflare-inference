# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation-only build of the reference dense-index selection kernels."""

import hashlib
from pathlib import Path
from threading import Lock

_loaded = False
_lock = Lock()


def prepare(kernel_cache):
    import torch
    from torch.utils.cpp_extension import load

    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("reference top-K kernels must be prepared before capture")
    global _loaded
    with _lock:
        if not _loaded:
            root = Path(__file__).parent
            files = sorted(p for p in root.rglob("*") if p.suffix in (".cu", ".h", ".cuh", ".cpp"))
            digest = hashlib.sha256()
            for path in files:
                digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
            digest.update((torch.__version__ + str(torch.version.cuda)).encode())
            build = Path(kernel_cache) / "index-topk" / digest.hexdigest()
            build.mkdir(parents=True, exist_ok=True)
            flags = ["-O3", "-std=c++20", "-DUSE_CUDA", "-DVLLM_ENABLE_COOPERATIVE_TOPK",
                     "-DTORCH_TARGET_VERSION=0x020A000000000000ULL"]
            load(name="flare_reference_topk", sources=[str(root / "bindings.cpp"),
                *(str(root / "csrc" / n) for n in ("sampler.cu", "topk.cu", "cooperative_topk.cu"))],
                extra_include_paths=[str(root / "csrc")], extra_cflags=flags,
                extra_cuda_cflags=flags + ["-gencode=arch=compute_100f,code=sm_100f", "--expt-relaxed-constexpr"],
                build_directory=str(build), is_python_module=False, verbose=False)
            _loaded = True
    return torch.ops.flare_reference_topk
