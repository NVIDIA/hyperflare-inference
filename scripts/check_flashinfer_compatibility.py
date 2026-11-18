# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that FlashInfer's RoCE proxy ABI matches the native runtime's ABI.

The script reads four files and imports nothing from FlashInfer or CUDA. From
FlashInfer it reads `ABI_VERSION` in
`experimental/b12x/comm/roce/_proxy.py` with `ast` and `ROCE_ABI_VERSION` in
`experimental/b12x/comm/roce/_roce_proxy.c` with a regular expression. From the
repository it reads `ROCE_ABI_VERSION` in `helios/native/roce_proxy.c` and
`PROXY_ABI` in `helios/src/execution/roce.rs`. Prepared cores record the
FlashInfer proxy ABI and the runtime rejects a core whose ABI it does not
provide, so the two must agree.

FlashInfer is located through `importlib.util.find_spec("flashinfer")` or an
explicit `--flashinfer-root`. On success the script prints every value and the
installed FlashInfer commit when `direct_url.json` records one. On disagreement
it exits 1 with one line that names every value and file.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import pathlib
import re
import sys


ROCE_MODULE = pathlib.Path("experimental") / "b12x" / "comm" / "roce"
PROXY_PY = "_proxy.py"
PROXY_C = "_roce_proxy.c"
RUNTIME_C = pathlib.Path("helios") / "native" / "roce_proxy.c"
RUNTIME_RS = pathlib.Path("helios") / "src" / "execution" / "roce.rs"

DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+ROCE_ABI_VERSION[ \t]+(\d+)\b", re.MULTILINE)
PROXY_ABI = re.compile(r"\bpub\s+const\s+PROXY_ABI\s*:\s*u32\s*=\s*(\d+)\s*;")
ACCEPTED = re.compile(r"matches!\s*\(\s*spec\.proxy_abi\s*,\s*([0-9][0-9 \t|]*)\)")


class CheckError(Exception):
    """A file could not be read or did not contain the expected value."""


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckError(f"cannot read {path}: {exc}") from exc


def _read_abi_version(path: pathlib.Path) -> int:
    """Read the module-level integer `ABI_VERSION` assignment with `ast`."""
    try:
        tree = ast.parse(_read_text(path), filename=str(path))
    except SyntaxError as exc:
        raise CheckError(f"cannot parse {path}: {exc}") from exc
    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "ABI_VERSION" for target in targets):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, int):
            raise CheckError(f"{path}: ABI_VERSION is not an integer literal")
        values.append(node.value.value)
    return _one_value(path, "ABI_VERSION", values)


def _read_define(path: pathlib.Path) -> int:
    """Read `#define ROCE_ABI_VERSION` with a regular expression."""
    values = [int(value) for value in DEFINE.findall(_read_text(path))]
    return _one_value(path, "ROCE_ABI_VERSION", values)


def _read_proxy_abi(path: pathlib.Path) -> int:
    """Read `pub const PROXY_ABI: u32` with a regular expression."""
    values = [int(value) for value in PROXY_ABI.findall(_read_text(path))]
    return _one_value(path, "PROXY_ABI", values)


def _read_accepted_abis(path: pathlib.Path) -> list[int]:
    """Return the core ABIs the runtime lists in `matches!`, when it does."""
    match = ACCEPTED.search(_read_text(path))
    if match is None:
        return []
    return sorted({int(value) for value in re.findall(r"\d+", match.group(1))})


def _one_value(path: pathlib.Path, name: str, values: list[int]) -> int:
    unique = set(values)
    if len(unique) != 1:
        found = ", ".join(str(value) for value in sorted(unique)) or "none"
        raise CheckError(f"{path}: expected one {name} value, found {found}")
    return unique.pop()


def _flashinfer_roce_dir(explicit_root: str | None) -> pathlib.Path:
    """Return FlashInfer's RoCE directory from an explicit root or the installation."""
    if explicit_root is not None:
        root = pathlib.Path(explicit_root)
        for candidate in (root / "flashinfer" / ROCE_MODULE, root / ROCE_MODULE, root):
            if (candidate / PROXY_PY).is_file() and (candidate / PROXY_C).is_file():
                return candidate
        raise CheckError(
            f"{root}: no FlashInfer RoCE directory found; expected "
            f"flashinfer/{ROCE_MODULE} or {ROCE_MODULE} with {PROXY_PY} and {PROXY_C}"
        )
    try:
        spec = importlib.util.find_spec("flashinfer")
    except (ImportError, ValueError) as exc:
        raise CheckError(f"cannot locate the installed flashinfer package: {exc}") from exc
    if spec is None or not spec.submodule_search_locations:
        raise CheckError("flashinfer is not installed; pass --flashinfer-root")
    package = pathlib.Path(next(iter(spec.submodule_search_locations)))
    roce = package / ROCE_MODULE
    for name in (PROXY_PY, PROXY_C):
        if not (roce / name).is_file():
            raise CheckError(f"{roce / name}: not found in the installed flashinfer package")
    return roce


def _installed_commit(package: pathlib.Path) -> tuple[str, pathlib.Path] | None:
    """Read the installed FlashInfer commit from `direct_url.json` when present."""
    candidates: list[tuple[pathlib.Path, dict]] = []
    try:
        distribution = importlib.metadata.distribution("flashinfer-python")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        distribution = None
    if distribution is not None:
        text = distribution.read_text("direct_url.json")
        if text:
            try:
                candidates.append((pathlib.Path(distribution.locate_file("")), json.loads(text)))
            except ValueError:
                pass
    for info in package.parent.glob("flashinfer_python-*.dist-info/direct_url.json"):
        try:
            candidates.append((info.parent, json.loads(info.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    for location, data in candidates:
        commit = data.get("vcs_info", {}).get("commit_id")
        if isinstance(commit, str) and commit:
            return commit, location
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flashinfer-root",
        help="FlashInfer checkout or package root; default: the installed package",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Flare repository root; default: the parent of this script's directory",
    )
    args = parser.parse_args(argv)
    roce = None
    try:
        roce = _flashinfer_roce_dir(args.flashinfer_root)
        readings = [
            ("FlashInfer _proxy.py", _read_abi_version(roce / PROXY_PY), roce / PROXY_PY),
            ("FlashInfer _roce_proxy.c", _read_define(roce / PROXY_C), roce / PROXY_C),
            ("runtime roce_proxy.c", _read_define(args.repo_root / RUNTIME_C), args.repo_root / RUNTIME_C),
            ("runtime roce.rs PROXY_ABI", _read_proxy_abi(args.repo_root / RUNTIME_RS), args.repo_root / RUNTIME_RS),
        ]
    except CheckError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    flashinfer_proxy, flashinfer_proxy_c, runtime_c, runtime_rs = (value for _, value, _ in readings)
    failures = []
    if flashinfer_proxy != flashinfer_proxy_c:
        failures.append("the two FlashInfer files disagree")
    if runtime_c != runtime_rs:
        failures.append("the two runtime files disagree")
    if flashinfer_proxy != runtime_rs:
        failures.append("FlashInfer's ABI is not the runtime PROXY_ABI")
    if failures:
        detail = ", ".join(f"{name}={value} ({path})" for name, value, path in readings)
        print(f"FAIL proxy ABI mismatch: {detail}; {'; '.join(failures)}", file=sys.stderr)
        return 1
    for name, value, path in readings:
        print(f"{name}: {value} ({path})")
    accepted = _read_accepted_abis(args.repo_root / RUNTIME_RS)
    if accepted:
        print(f"runtime roce.rs accepted core ABIs: {', '.join(str(value) for value in accepted)}")
    if args.flashinfer_root is None:
        commit = _installed_commit(roce.parents[len(ROCE_MODULE.parts) - 1])
        if commit is None:
            print("FlashInfer commit: not found in direct_url.json")
        else:
            print(f"FlashInfer commit: {commit[0]} (direct_url.json at {commit[1]})")
    print(f"OK FlashInfer proxy ABI {flashinfer_proxy} matches the runtime proxy ABI {runtime_rs}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
