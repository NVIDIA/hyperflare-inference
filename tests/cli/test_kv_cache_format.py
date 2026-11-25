# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ignition --kv-cache-format declaration, planning, help, and rebuild args."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ignition import planning
from ignition.cli import _parser
from ignition.cli.help import recipe_help
from ignition.declaration import (
    Checkpoint,
    KvCacheFormat,
    RankCosts,
    Recipe,
    SamplingDefaults,
    Speculation,
)
from ignition.errors import UsageError
from ignition.preparation.rebuild_manifest import reproduction_arguments

_GIB = 1 << 30


def _make_recipe(*, with_nvfp4: bool = True) -> Recipe:
    checkpoint = Checkpoint(
        alias="test-model",
        repository="org/test-model",
        revision="abc123",
        architecture="TestArch",
        stored_bytes=16 * _GIB,
        device_bytes=16 * _GIB,
    )
    kv_formats = (
        {
            "nvfp4_4o6k": KvCacheFormat(
                ranks={1: RankCosts(state_row_bytes=4 << 20, page_bytes=1_179_648, graph_bytes=2 * _GIB)},
                plain_ranks={1: RankCosts(state_row_bytes=2 << 20, page_bytes=1_179_648, graph_bytes=1 * _GIB)},
            )
        }
        if with_nvfp4
        else {}
    )
    return Recipe(
        name="test_model",
        served_model_name="test-model",
        checkpoints=(checkpoint,),
        context=131_072,
        page_tokens=128,
        concurrency=16,
        max_concurrency=16,
        max_draft_tokens=4,
        prefill_chunk=2048,
        ranks={1: RankCosts(state_row_bytes=4 << 20, page_bytes=2_097_152, graph_bytes=2 * _GIB)},
        plain_ranks={1: RankCosts(state_row_bytes=2 << 20, page_bytes=2_097_152, graph_bytes=1 * _GIB)},
        speculation=Speculation(draft_tokens=4),
        kv_cache_formats=kv_formats,
        prepare=lambda *args, **kwargs: None,
        usage=lambda *args, **kwargs: (),
        sampling_defaults=SamplingDefaults(temperature=1.0, top_p=1.0, top_k=0),
    )


def _gpu() -> planning.Gpu:
    return planning.Gpu(
        index=0,
        name="RTX 5090",
        uuid="GPU-0",
        capability=(12, 0),
        free_bytes=31 * _GIB,
        total_bytes=32 * _GIB,
    )


def test_kv_cache_format_planning_scales_pages_and_to_dict():
    recipe = _make_recipe(with_nvfp4=True)
    cp = recipe.checkpoints[0]
    gpus = (_gpu(),)

    fp8_plan = planning.plan(recipe, cp, gpus, selected=(0,), concurrency=16, reserve_bytes=0, speculation=None)
    nvfp4_plan = planning.plan(
        recipe,
        cp,
        gpus,
        selected=(0,),
        concurrency=16,
        reserve_bytes=0,
        speculation=None,
        kv_cache_format="nvfp4_4o6k",
    )
    assert fp8_plan.kv_cache_format == "fp8"
    assert nvfp4_plan.kv_cache_format == "nvfp4_4o6k"
    assert fp8_plan.to_dict()["kv_cache_format"] == "fp8"
    assert nvfp4_plan.to_dict()["kv_cache_format"] == "nvfp4_4o6k"
    assert nvfp4_plan.costs.page_bytes == 1_179_648
    assert nvfp4_plan.pages > int(fp8_plan.pages * 1.7)

    # With speculation enabled, ranks (not plain_ranks) are used.
    nvfp4_mtp = planning.plan(
        recipe,
        cp,
        gpus,
        selected=(0,),
        concurrency=16,
        reserve_bytes=0,
        speculation=recipe.speculation,
        kv_cache_format="nvfp4_4o6k",
    )
    assert nvfp4_mtp.costs.state_row_bytes == 4 << 20
    assert nvfp4_mtp.costs.page_bytes == 1_179_648


def test_kv_cache_format_rejects_unsupported_recipe():
    recipe = _make_recipe(with_nvfp4=False)
    cp = recipe.checkpoints[0]
    with pytest.raises(UsageError, match="--kv-cache-format nvfp4_4o6k: test-model supports fp8"):
        planning.plan(
            recipe,
            cp,
            (_gpu(),),
            selected=(0,),
            concurrency=16,
            reserve_bytes=0,
            speculation=None,
            kv_cache_format="nvfp4_4o6k",
        )


def test_kv_cache_format_cli_help_and_reproduction_args():
    recipe = _make_recipe(with_nvfp4=True)
    parser = _parser(recipes=(recipe,))
    args = parser.parse_args(["export", "test-model", "--kv-cache-format", "nvfp4_4o6k"])
    assert args.kv_cache_format == "nvfp4_4o6k"

    help_text = recipe_help("test-model", recipes=(recipe,))
    assert "--kv-cache-format: fp8, nvfp4_4o6k; default fp8." in help_text
    assert "--kv-cache-format nvfp4_4o6k (44% smaller KV pages)" in help_text

    cp = recipe.checkpoints[0]
    nvfp4_plan = planning.plan(
        recipe,
        cp,
        (_gpu(),),
        selected=(0,),
        concurrency=16,
        reserve_bytes=0,
        speculation=None,
        kv_cache_format="nvfp4_4o6k",
    )
    fp8_plan = replace(nvfp4_plan, kv_cache_format="fp8")
    model = SimpleNamespace(directory=None)
    nvfp4_args = reproduction_arguments(model, nvfp4_plan, compiled_forwards=True)
    fp8_args = reproduction_arguments(model, fp8_plan, compiled_forwards=True)
    assert "--kv-cache-format" in nvfp4_args
    assert nvfp4_args[nvfp4_args.index("--kv-cache-format") + 1] == "nvfp4_4o6k"
    assert "--kv-cache-format" not in fp8_args
