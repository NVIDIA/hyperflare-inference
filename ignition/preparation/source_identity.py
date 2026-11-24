# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Source compatibility identity for preparation and core rebuild metadata."""

from contextlib import contextmanager
import contextvars
import hashlib
from pathlib import Path

_EXTRA_ROOTS: contextvars.ContextVar[tuple[tuple[str, Path], ...]] = contextvars.ContextVar(
    "flare_source_identity_roots", default=()
)


@contextmanager
def source_roots(roots=()):
    """Bind external `(label, path)` source roots for the duration of a CLI invocation."""
    normalized = tuple((str(label), Path(path)) for label, path in roots)
    token = _EXTRA_ROOTS.set(normalized)
    try:
        yield
    finally:
        _EXTRA_ROOTS.reset(token)


def source_identity(*, extra_roots=()):
    """Digest of the Python and native sources every rank must run identically.

    `extra_roots` supplies optional `(label, path)` pairs for external recipe
    packages so multi-host ranks verify identical recipe code alongside core
    Ignition, Heliopause/Helios, and FlashInfer.
    """
    from importlib import util
    import flare_heliopause

    # Locate FlashInfer without importing it: CUDA device selection must precede PyTorch's import.
    flashinfer = util.find_spec("flashinfer")
    if flashinfer is None:
        raise RuntimeError("the preparation environment requires FlashInfer")
    ignition_root = Path(__file__).resolve().parents[1]
    combined_roots = (*_EXTRA_ROOTS.get(), *extra_roots)
    return _source_identity(
        ignition_root,
        Path(flashinfer.origin).resolve().parent,
        native_identities=flare_heliopause.build_source_identity(),
        extra_roots=combined_roots,
    )


def _source_identity(
    source_root: Path,
    flashinfer_package: Path,
    *,
    native_identities=None,
    extra_roots=(),
) -> str:
    """Hash labeled source trees independently of checkout and install layout.

    When `native_identities` supplies precomputed `helios` and `heliopause`
    digests (from `flare_heliopause.build_source_identity()`), `source_root` is
    the installed `ignition` package directory and no Rust checkout is required
    on disk. When `native_identities` is omitted, `source_root` is treated as a
    monorepo root containing `ignition/`, `helios/src`, and `heliopause/src`.

    FlashInfer's ``data/include`` and ``data/csrc`` contain the native sources
    used by its JIT specs. Editable installs link those directories; wheels
    copy them. Exclude ``data`` from the Python-package tree and include the
    two resolved native trees separately, so each contributes exactly once.
    Dependency data and compilation caches are outside this source set.
    """
    import flare_heliopause

    source_root, flashinfer_package = source_root.resolve(), flashinfer_package.resolve()
    data = flashinfer_package / "data"
    native = {name: (data / name).resolve() for name in ("include", "csrc")}
    for name, source in native.items():
        if not source.is_dir():
            raise RuntimeError(f"FlashInfer native source directory is unavailable: {data / name}; "
                               "install the selected FlashInfer package before preparation")

    digest = hashlib.sha256()

    def mix_hex(label: str, hex_digest: str):
        digest.update(label.encode("ascii") + b"\0")
        digest.update(bytes.fromhex(hex_digest))

    def mix_tree(label: str, path: Path, excluded=()):
        identity = flare_heliopause.export_source_identity(
            [path], excluded_directories=list(excluded))
        mix_hex(label, identity)

    if native_identities is not None:
        mix_tree("ignition", source_root)
        mix_hex("helios", native_identities["helios"])
        mix_hex("heliopause", native_identities["heliopause"])
    else:
        mix_tree("ignition", source_root / "ignition")
        mix_tree("helios", source_root / "helios" / "src")
        mix_tree("heliopause", source_root / "heliopause" / "src")

    mix_tree("flashinfer.python", flashinfer_package, (data,))
    mix_tree("flashinfer.include", native["include"])
    mix_tree("flashinfer.csrc", native["csrc"])

    for label, path in extra_roots:
        mix_tree(label, Path(path).resolve())

    return digest.hexdigest()
