# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation binding to the pinned FlashMLA library with explicit scratch."""

from functools import cache
import hashlib
from pathlib import Path


def prepare(kernel_cache: Path, library: Path):
    """Link the supplied FlashMLA build before capture; retain owners through drain.

    The library must be built from the revision in NOTICE with SM103 support.
    It is a preparation input, never a library loaded by native serving.
    """
    import torch
    if torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("the fused FlashMLA binding requires SM103")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FlashMLA must be prepared before CUDA graph capture")
    library = Path(library).resolve(strict=True)
    return _build(Path(kernel_cache).resolve(), library)


@cache
def _build(kernel_cache: Path, library: Path):
    import torch
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in (".h", ".hpp", ".cu"):
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    with library.open("rb") as stream:
        digest.update(hashlib.file_digest(stream, "sha256").digest())
    digest.update((torch.__version__ + str(torch.version.cuda) + str(library)).encode())
    build = kernel_cache / "flashmla-binding" / digest.hexdigest()
    build.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(str(library))
    return load(name="flare_flashmla_binding", sources=[str(root / "bindings.cu")],
        extra_include_paths=[str(root / "csrc"), str(root / "csrc/cutlass/include")],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_103a,code=sm_103a"],
        extra_cflags=["-O3", "-std=c++20"], extra_ldflags=[str(library)],
        build_directory=str(build), verbose=False)
