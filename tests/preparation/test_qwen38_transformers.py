# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen preparation requires the Transformers release pinned in pyproject.toml."""

import importlib.metadata

import pytest

from ignition.preparation import qwen38


def test_accepts_pinned_release(monkeypatch):
    monkeypatch.setattr(qwen38.importlib.metadata, "version", lambda name: qwen38.TRANSFORMERS_VERSION)
    qwen38.require_transformers("Qwen preparation")


def test_rejects_other_release(monkeypatch):
    monkeypatch.setattr(qwen38.importlib.metadata, "version", lambda name: "5.22.0")
    with pytest.raises(RuntimeError, match=r"requires Transformers 5\.23\.0; found 5\.22\.0"):
        qwen38.require_transformers("Qwen preparation")


def test_rejects_missing_distribution(monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(qwen38.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="requires Transformers 5.23.0"):
        qwen38.require_transformers("Qwen preparation")
