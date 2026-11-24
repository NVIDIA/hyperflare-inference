# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile GPU KV-cache codec CUDA C++ kernels into a target-arch ELF cubin at core export."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

GPU_CODEC_MODULE_NAME = "kv_codec_gpu"
_SOURCE = Path(__file__).with_name("kv_codec_gpu.cu")
_CACHE: dict[tuple[int, int], Path] = {}


def _find_nvcc() -> str:
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        CUDA_HOME = None
    candidates = []
    if CUDA_HOME:
        candidates.append(Path(CUDA_HOME) / "bin" / "nvcc")
    which = shutil.which("nvcc")
    if which:
        candidates.append(Path(which))
    candidates.append(Path("/usr/local/cuda/bin/nvcc"))
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    raise RuntimeError(
        "nvcc is required during core export to compile embedded GPU KV codec kernels"
    )


def _target_arch(capability: tuple[int, int]) -> str:
    major, minor = capability
    suffix = "a" if (major, minor) >= (9, 0) else ""
    return f"sm_{major}{minor}{suffix}"


def compile_gpu_codec_cubin(capability: tuple[int, int] | None = None) -> Path:
    """Compile `kv_codec_gpu.cu` into a target-architecture ELF cubin and return its path.

    Caches the compiled cubin per `(arch, source_hash, nvcc_version)` in a shared temporary
    directory so multi-rank TP exports compile once and reuse the verified ELF cubin.
    """
    if capability is None:
        import torch

        capability = tuple(int(x) for x in torch.cuda.get_device_capability())
    cached = _CACHE.get(capability)
    if cached is not None and cached.is_file():
        return cached

    if not _SOURCE.is_file():
        raise FileNotFoundError(f"missing GPU KV codec kernel source: {_SOURCE}")
    source_bytes = _SOURCE.read_bytes()
    nvcc = _find_nvcc()
    version = subprocess.run(
        [nvcc, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    arch = _target_arch(capability)
    digest = hashlib.sha256(
        arch.encode() + b"\0" + version.encode() + b"\0" + source_bytes
    ).hexdigest()[:24]

    if hasattr(os, "getuid"):
        uid = os.getuid()
        candidate = Path(tempfile.gettempdir()) / f"flare-kv-codec-cubins-{uid}"
        try:
            candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
            st = os.lstat(candidate)
            if (
                not os.path.islink(candidate)
                and st.st_uid == uid
                and (st.st_mode & 0o077) == 0
            ):
                cache_dir = candidate
            else:
                cache_dir = Path(tempfile.mkdtemp(prefix="flare-kv-codec-cubins-"))
        except OSError:
            cache_dir = Path(tempfile.mkdtemp(prefix="flare-kv-codec-cubins-"))
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="flare-kv-codec-cubins-"))
    out = cache_dir / f"{GPU_CODEC_MODULE_NAME}-{arch}-{digest}.cubin"
    if out.is_file():
        cached_data = out.read_bytes()
        if cached_data.startswith(b"\x7fELF"):
            _CACHE[capability] = out
            return out
    tmp = cache_dir / f".{out.name}.{os.getpid()}.tmp"
    try:
        subprocess.run(
            [
                nvcc,
                "-O3",
                "--cubin",
                f"-arch={arch}",
                "-o",
                str(tmp),
                str(_SOURCE),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = tmp.read_bytes()
        if not data.startswith(b"\x7fELF"):
            raise RuntimeError(
                f"compiled GPU KV codec image for {arch} is not an ELF cubin"
            )
        os.replace(tmp, out)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"nvcc failed compiling GPU KV codec for {arch}:\n{exc.stderr}"
        ) from exc
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _CACHE[capability] = out
    return out
