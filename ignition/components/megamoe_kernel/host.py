# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation-only MegaMoE launcher with explicit physical weight extents."""

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
        raise RuntimeError("MegaMoE host-tier binding must be built before capture")
    root = Path(__file__).parent
    headers = header_root(kernel_cache)
    digest = hashlib.sha256()
    for path in sorted([root / "host_bindings.cpp", *(root / "host").rglob("*.hpp"),
                        *(root / "host").rglob("*.h")]):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    digest.update((str(headers) + torch.__version__ + str(torch.version.cuda)).encode())
    key = digest.hexdigest()
    build = Path(kernel_cache) / "megamoe-host-binding" / key
    build.mkdir(parents=True, exist_ok=True)
    cuda = Path(CUDA_HOME)
    module = load(name="flare_megamoe_host_" + key[:16], sources=[str(root / "host_bindings.cpp")],
        extra_include_paths=[str(root / "host"), str(root / "host/include"),
            str(Path(deep_gemm.__file__).parent / "include"),
            str(cuda / "include"), str(cuda / "include/cccl")],
        extra_cflags=["-O3", "-std=c++20", "-DDJ_DISABLE_GIL=1", "-fvisibility=hidden", "-fno-gnu-unique"],
        extra_ldflags=[f"-L{cuda / 'lib64'}", "-lcudart", "-lcublasLt"],
        build_directory=str(build), verbose=False)
    module.initialize(str(headers), str(Path(kernel_cache) / "deepgemm"))
    return module
