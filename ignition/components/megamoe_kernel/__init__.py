# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation-only binding of the unchanged DerpSeek MegaMoE reference kernel."""

from functools import cache
import errno
import hashlib
from pathlib import Path
import shutil
import tempfile


@cache
def initialize(kernel_cache: Path):
    """Bind a private header tree without changing the installed DeepGEMM package.

    All compilation and resource discovery happen during preparation. Exported
    graphs retain compiled CUDA modules; Helios neither imports this package nor
    compiles kernels when serving them.
    """
    import deep_gemm

    source = Path(deep_gemm.__file__).parent / "include"
    overlay = Path(__file__).parent / "include"
    files = {path.relative_to(source): path for path in source.rglob("*") if path.is_file()}
    files.update({path.relative_to(overlay): path for path in overlay.rglob("*") if path.is_file()})
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(str(name).encode() + b"\0")
        digest.update(path.read_bytes())
    parent = Path(kernel_cache) / "megamoe-headers"
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / digest.hexdigest()
    if not root.exists():
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            candidate = Path(temporary) / "root"
            for name, path in files.items():
                target = candidate / "include" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            try:
                candidate.rename(root)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not root.is_dir():
                    raise
    deep_gemm._C.init(str(root))
    return deep_gemm
