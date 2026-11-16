# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation binding for reference MXFP4 index logits with caller-owned outputs."""

from functools import cache
import hashlib
from pathlib import Path

from . import header_root


@cache
def prepare(kernel_cache: Path):
    import torch
    import deep_gemm
    from torch.utils.cpp_extension import CUDA_HOME, load

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("MXFP4 index logits must be prepared before capture")
    if torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("the reference MXFP4 index logits binding requires SM103")
    root = Path(__file__).parent
    headers = header_root(kernel_cache)
    digest = hashlib.sha256()
    for path in sorted([root / "mqa_bindings.cpp", *(root / "host").rglob("*.hpp"),
                        *(root / "host").rglob("*.h")]):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    digest.update((str(headers) + torch.__version__ + str(torch.version.cuda)).encode())
    key = digest.hexdigest()
    build = Path(kernel_cache) / "mqa-binding" / key
    build.mkdir(parents=True, exist_ok=True)
    cuda = Path(CUDA_HOME)
    module = load(name="flare_mqa_" + key[:16], sources=[str(root / "mqa_bindings.cpp")],
        extra_include_paths=[str(root / "host"), str(root / "host/include"),
            str(Path(deep_gemm.__file__).parent / "include"),
            str(cuda / "include"), str(cuda / "include/cccl")],
        extra_cflags=["-O3", "-std=c++20", "-DDJ_DISABLE_GIL=1", "-fvisibility=hidden", "-fno-gnu-unique"],
        extra_ldflags=[f"-L{cuda / 'lib64'}", "-lcudart", "-lcublasLt"],
        build_directory=str(build), verbose=False)
    module.initialize(str(headers))
    return module
