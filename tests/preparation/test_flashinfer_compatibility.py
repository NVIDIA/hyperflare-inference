# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for `scripts/check_flashinfer_compatibility.py`.

Each test builds a small FlashInfer tree and a small repository tree in a
temporary directory. The script reads those files and never imports FlashInfer.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_flashinfer_compatibility.py"


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture(
    tmp_path: pathlib.Path,
    *,
    proxy_py: int = 14,
    proxy_c: int = 14,
    runtime_c: int = 14,
    runtime_rs: int = 14,
) -> tuple[pathlib.Path, pathlib.Path]:
    flashinfer = tmp_path / "flashinfer-root"
    roce = flashinfer / "flashinfer" / "experimental" / "b12x" / "comm" / "roce"
    _write(
        roce / "_proxy.py",
        f'"""Fixture proxy."""\n\n# ABI_VERSION = 99\nABI_VERSION = {proxy_py}\n',
    )
    _write(
        roce / "_roce_proxy.c",
        f"/* Fixture proxy. */\n// #define ROCE_ABI_VERSION 99\n#define ROCE_ABI_VERSION {proxy_c}\n",
    )
    repo = tmp_path / "repo"
    _write(repo / "helios" / "native" / "roce_proxy.c", f"#define ROCE_ABI_VERSION {runtime_c}\n")
    _write(
        repo / "helios" / "src" / "execution" / "roce.rs",
        f"pub const PROXY_ABI: u32 = {runtime_rs};\n",
    )
    return flashinfer, repo


def _run(flashinfer: pathlib.Path, repo: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            "--flashinfer-root",
            str(flashinfer),
            "--repo-root",
            str(repo),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_agreeing_proxy_abi_passes(tmp_path):
    result = _run(*_fixture(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "FlashInfer _proxy.py: 14" in result.stdout
    assert "FlashInfer _roce_proxy.c: 14" in result.stdout
    assert "runtime roce_proxy.c: 14" in result.stdout
    assert "runtime roce.rs PROXY_ABI: 14" in result.stdout
    assert "OK FlashInfer proxy ABI 14 matches the runtime proxy ABI 14." in result.stdout


def test_flashinfer_copy_with_abi_13_fails(tmp_path):
    result = _run(*_fixture(tmp_path, proxy_py=13, proxy_c=13))
    assert result.returncode == 1
    assert len(result.stderr.strip().splitlines()) == 1
    assert "FAIL proxy ABI mismatch" in result.stderr
    assert "FlashInfer _proxy.py=13" in result.stderr
    assert "FlashInfer _roce_proxy.c=13" in result.stderr
    assert "runtime roce_proxy.c=14" in result.stderr
    assert "runtime roce.rs PROXY_ABI=14" in result.stderr
    assert "FlashInfer's ABI is not the runtime PROXY_ABI" in result.stderr


def test_flashinfer_files_disagreeing_fails(tmp_path):
    result = _run(*_fixture(tmp_path, proxy_py=14, proxy_c=13))
    assert result.returncode == 1
    assert len(result.stderr.strip().splitlines()) == 1
    assert "the two FlashInfer files disagree" in result.stderr
    assert "FlashInfer _proxy.py=14" in result.stderr
    assert "FlashInfer _roce_proxy.c=13" in result.stderr


def test_runtime_files_disagreeing_fails(tmp_path):
    result = _run(*_fixture(tmp_path, runtime_c=14, runtime_rs=13))
    assert result.returncode == 1
    assert len(result.stderr.strip().splitlines()) == 1
    assert "the two runtime files disagree" in result.stderr
    assert "runtime roce_proxy.c=14" in result.stderr
    assert "runtime roce.rs PROXY_ABI=13" in result.stderr
