# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ignition --kv-cache-format declaration, planning, help, forwarding, and manifest."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
from google.protobuf import json_format

from ignition import planning
from ignition.cli import _parser
from ignition.cli.help import recipe_help
from ignition.declaration import KvCacheFormat, RankCosts, Speculation, SpeculativeMethod
from ignition.errors import UsageError
from ignition.preparation import rebuild_manifest, runtime_core_pb2 as proto
from ignition.preparation.bootstrap import BasePreparationConfig, common_preparation_kwargs

from tests.cli.test_planning import GIB, MIB, _gpu, _recipe


def _kv_recipe(*, with_nvfp4: bool = True, ranks=(1, 2)):
    base = _recipe("kvtest", device_gib=16, ranks=ranks)
    if not with_nvfp4:
        return base
    return replace(
        base,
        kv_cache_format="fp8",
        kv_cache_formats=MappingProxyType(
            {
                "nvfp4_4o6k": KvCacheFormat(
                    page_bytes=MappingProxyType({size: (1_179_648 // size) for size in ranks}),
                    plain_page_bytes=MappingProxyType({size: (1_048_576 // size) for size in ranks}),
                )
            }
        ),
    )


def _manifest_params(plan):
    manifest = rebuild_manifest.from_plan(SimpleNamespace(directory=None), plan, compiled_forwards=True)
    encoded = json_format.ParseDict(manifest, proto.PreparationManifest())
    decoded = proto.PreparationManifest.FromString(encoded.SerializeToString(deterministic=True))
    return {parameter.key: parameter for parameter in decoded.parameters}


def test_undeclared_recipe_defaults_to_none_and_omits_manifest_parameter():
    recipe = _kv_recipe(with_nvfp4=False, ranks=(1,))
    cp = recipe.checkpoints[0]
    gpus = (_gpu(0, free_gib=31),)

    plan = planning.plan(recipe, cp, gpus, selected=(0,), concurrency=4, reserve_bytes=0, speculation=None)
    assert plan.kv_cache_format is None
    assert not plan.has_custom_kv_cache_format
    assert plan.to_dict()["kv_cache_format"] is None
    assert "kv_cache_format" not in _manifest_params(plan)
    assert "--kv-cache-format" not in reproduction_arguments_for(plan)
    assert "--kv-cache-format" not in recipe_help(cp.alias, recipes=(recipe,))

    # Even --kv-cache-format fp8 is rejected when the recipe declares no KV format selection.
    with pytest.raises(UsageError, match="does not declare KV cache format selection"):
        planning.plan(
            recipe,
            cp,
            gpus,
            selected=(0,),
            concurrency=4,
            reserve_bytes=0,
            speculation=None,
            kv_cache_format="fp8",
        )


def reproduction_arguments_for(plan):
    return rebuild_manifest.reproduction_arguments(SimpleNamespace(directory=None), plan, compiled_forwards=True)


def test_kv_cache_format_planning_scales_pages_forwards_and_records_manifest():
    recipe = _kv_recipe(with_nvfp4=True, ranks=(1, 2))
    cp = recipe.checkpoints[0]
    gpus = (_gpu(0, free_gib=31),)

    fp8_plan = planning.plan(recipe, cp, gpus, selected=(0,), concurrency=4, reserve_bytes=0, speculation=None)
    nvfp4_plan = planning.plan(
        recipe,
        cp,
        gpus,
        selected=(0,),
        concurrency=4,
        reserve_bytes=0,
        speculation=None,
        kv_cache_format="nvfp4_4o6k",
    )
    assert fp8_plan.kv_cache_format == "fp8"
    assert not fp8_plan.has_custom_kv_cache_format
    assert nvfp4_plan.kv_cache_format == "nvfp4_4o6k"
    assert nvfp4_plan.has_custom_kv_cache_format
    assert fp8_plan.to_dict()["kv_cache_format"] == "fp8"
    assert nvfp4_plan.to_dict()["kv_cache_format"] == "nvfp4_4o6k"
    assert nvfp4_plan.costs.page_bytes == 1_048_576
    assert nvfp4_plan.costs.state_row_bytes == fp8_plan.costs.state_row_bytes
    assert nvfp4_plan.costs.graph_bytes == fp8_plan.costs.graph_bytes
    assert nvfp4_plan.pages > fp8_plan.pages

    # Speculative plan uses page_bytes (not plain_page_bytes) and preserves speculative costs.
    nvfp4_mtp = planning.plan(
        recipe,
        cp,
        gpus,
        selected=(0,),
        concurrency=4,
        reserve_bytes=0,
        speculation=recipe.speculation,
        kv_cache_format="nvfp4_4o6k",
    )
    assert nvfp4_mtp.costs.page_bytes == 1_179_648
    assert nvfp4_mtp.costs.state_row_bytes == recipe.ranks[1].state_row_bytes

    # common_preparation_kwargs forwards kv_cache_format into BasePreparationConfig.
    cfg = BasePreparationConfig(**common_preparation_kwargs(
        nvfp4_plan, compile_forwards=True, kernel_cache=None, retain_graphs=True
    ))
    assert cfg.kv_cache_format == "nvfp4_4o6k"

    # PreparationManifest records declared formats under SCOPE_DEPLOYMENT.
    fp8_params = _manifest_params(fp8_plan)
    nvfp4_params = _manifest_params(nvfp4_plan)
    assert fp8_params["kv_cache_format"].scope == proto.PREPARATION_PARAMETER_SCOPE_DEPLOYMENT
    assert fp8_params["kv_cache_format"].value.string_value == "fp8"
    assert nvfp4_params["kv_cache_format"].scope == proto.PREPARATION_PARAMETER_SCOPE_DEPLOYMENT
    assert nvfp4_params["kv_cache_format"].value.string_value == "nvfp4_4o6k"

    # reproduction_arguments emits the flag only when it differs from the recipe default.
    fp8_args = reproduction_arguments_for(fp8_plan)
    nvfp4_args = reproduction_arguments_for(nvfp4_plan)
    assert "--kv-cache-format" not in fp8_args
    assert "--kv-cache-format" in nvfp4_args
    assert nvfp4_args[nvfp4_args.index("--kv-cache-format") + 1] == "nvfp4_4o6k"


def test_kv_cache_format_preserves_checkpoint_and_speculative_method_cost_layering():
    base = _kv_recipe(with_nvfp4=True, ranks=(1, 2))
    cp_ranks = MappingProxyType({1: RankCosts(120 * MIB, 2 * MIB, 5 * GIB)})
    cp_plain = MappingProxyType({1: RankCosts(90 * MIB, 3 * MIB // 2, 4 * GIB)})
    cp = replace(base.checkpoints[0], ranks=cp_ranks, plain_ranks=cp_plain)
    dflash_ranks = MappingProxyType({
        1: RankCosts(140 * MIB, 2 * MIB, 6 * GIB),
        2: RankCosts(70 * MIB, 1 * MIB, 6 * GIB),
    })
    recipe = replace(
        base,
        checkpoints=(cp,),
        speculative_methods=MappingProxyType({"dflash2": SpeculativeMethod(ranks=dflash_ranks, draft_bytes=1 * GIB)}),
    )

    # Checkpoint plain_ranks state_row_bytes and graph_bytes survive under kv_cache_format.
    plain_costs = recipe.costs(1, None, cp, kv_cache_format="nvfp4_4o6k")
    assert plain_costs.state_row_bytes == 90 * MIB
    assert plain_costs.graph_bytes == 4 * GIB
    assert plain_costs.page_bytes == 1_048_576

    # SpeculativeMethod ranks survive under kv_cache_format while overriding page_bytes only.
    dflash_spec = Speculation(draft_tokens=3, method="dflash2")
    dflash_costs = recipe.costs(1, dflash_spec, cp, kv_cache_format="nvfp4_4o6k")
    assert dflash_costs.state_row_bytes == 140 * MIB
    assert dflash_costs.graph_bytes == 6 * GIB
    assert dflash_costs.page_bytes == 1_179_648

    # Unknown format name raises ValueError in Recipe.costs and UsageError in Plan.__post_init__.
    with pytest.raises(ValueError, match="unknown KV cache format 'bogus'"):
        recipe.costs(1, None, cp, kv_cache_format="bogus")
    good_plan = planning.plan(recipe, cp, (_gpu(0, free_gib=31),), selected=(0,), concurrency=4, reserve_bytes=0, speculation=None)
    with pytest.raises(UsageError, match="--kv-cache-format bogus: kvtest-alias supports fp8, nvfp4_4o6k"):
        replace(good_plan, kv_cache_format="bogus")


def test_kv_cache_format_validation_parser_and_help():
    base = _recipe("valtest", device_gib=16, ranks=(1, 2))
    # Alternative formats without a default kv_cache_format are rejected.
    with pytest.raises(ValueError, match="must declare a non-empty default kv_cache_format"):
        replace(base, kv_cache_formats={"nvfp4_4o6k": KvCacheFormat(page_bytes={1: 1024, 2: 512})})

    # Missing TP sizes are named in the validation error.
    with pytest.raises(ValueError, match=r"missing=\[2\]"):
        replace(base, kv_cache_format="fp8", kv_cache_formats={"nvfp4_4o6k": KvCacheFormat(page_bytes={1: 1024})})

    # Non-positive page_bytes are rejected.
    with pytest.raises(ValueError, match="must be positive integers"):
        replace(base, kv_cache_format="fp8", kv_cache_formats={"nvfp4_4o6k": KvCacheFormat(page_bytes={1: 0, 2: 512})})

    # Shared parser accepts arbitrary recipe-declared format names (no hardcoded choices).
    recipe = replace(
        base,
        kv_cache_format="bf16",
        kv_cache_formats=MappingProxyType(
            {"int4_custom": KvCacheFormat(page_bytes=MappingProxyType({1: 1 * MIB, 2: 512 * 1024}))}
        ),
    )
    parser = _parser(recipes=(recipe,))
    args = parser.parse_args(["export", "valtest-alias", "--kv-cache-format", "int4_custom"])
    assert args.kv_cache_format == "int4_custom"

    # Help names the TP size and qualifies the page savings estimate.
    help_text = recipe_help("valtest-alias", recipes=(recipe,))
    assert "--kv-cache-format: bf16, int4_custom; default bf16." in help_text
    assert "--kv-cache-format int4_custom (TP1: estimated 50% smaller KV pages; actual savings vary)" in help_text
