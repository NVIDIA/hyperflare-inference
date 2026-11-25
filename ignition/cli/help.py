# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Offline command help drawn from recipe declarations."""

import argparse


def recipe_help(model=None, *, recipes=(), prog="ignition"):
    catalog = [(recipe, checkpoint) for recipe in recipes for checkpoint in recipe.checkpoints]
    selected = next(((recipe, checkpoint) for recipe, checkpoint in catalog
                     if model in (checkpoint.alias, checkpoint.repository)), None)
    if selected is None:
        if not catalog:
            return (
                "No model recipes are registered in this CLI.\n"
                "Core publishing and rank-join commands (keygen, key-password, sign, edit, combine, convert, join)\n"
                "work without recipes. Build a recipe CLI with ignition.cli.main(recipes=...)."
            )
        lines = ["Available model recipes:",
                 f"  {'MODEL':<26}{'GPUs':<14}Drafts (default/max)"]
        for recipe, checkpoint in catalog:
            gpus = ",".join(map(str, recipe.checkpoint_tensor_parallel(checkpoint)))
            drafts = ("none" if recipe.speculation is None else
                      f"{recipe.speculation.draft_tokens}/{recipe.max_draft_tokens}")
            lines.append(f"  {checkpoint.alias:<26}{gpus:<14}{drafts}")
        lines.extend(("", "Show a recipe's defaults, limits, and optional features:",
                      f"  {prog} export MODEL --help",
                      "MODEL also accepts org/repository[@revision] or a checkpoint directory.",
                      "GPU counts are supported tensor-parallel sizes; memory must still fit."))
        return "\n".join(lines)

    recipe, checkpoint = selected
    lines = [f"Model recipe: {checkpoint.alias}", f"  Checkpoint: {checkpoint.repository}",
             f"  --gpus: select {', '.join(map(str, recipe.checkpoint_tensor_parallel(checkpoint)))} GPUs",
             f"  --max-concurrency: 1..{recipe.max_concurrency}; recipe default {recipe.concurrency}",
             f"  --prefill-chunk-size: 64..{recipe.context}, multiple of 64; default {recipe.prefill_chunk}"]
    if recipe.max_images:
        lines.append(f"  --image-concurrency: 1..max concurrency; default {min(4, recipe.concurrency)}.")
    if len(recipe.mla_compute_modes) > 1:
        lines.append(f"  --mla-compute-mode: {', '.join(recipe.mla_compute_modes)}; default {recipe.mla_compute_mode}.")
    if checkpoint.scale_compressed_device_bytes:
        lines.append(f"  --expert-scale-compression: native, {', '.join(checkpoint.scale_compressed_device_bytes)};"
                     f" default {recipe.expert_scale_compression or 'native'}.")
    if recipe.supported_kv_cache_formats:
        if recipe.kv_cache_formats:
            lines.append(
                f"  --kv-cache-format: {', '.join(recipe.supported_kv_cache_formats)};"
                f" default {recipe.kv_cache_format}."
            )
        else:
            lines.append(f"  --kv-cache-format: {recipe.kv_cache_format} only.")
    speculation = recipe.speculation
    if speculation is not None:
        methods = tuple(dict.fromkeys(name for name in (speculation.method, *recipe.speculative_methods) if name))
        if methods:
            lines.append(f"  --speculative-method: {', '.join(methods)}; default {speculation.method or 'recipe draft'}")
        lines.extend((
            f"  --mtp-draft-tokens: 0..{recipe.max_draft_tokens}; default {speculation.draft_tokens}",
            f"  Verification: {'adaptive' if speculation.adaptive else 'fixed width'} by default;"
            " --no-mtp excludes draft weights and state.",
            "  --mtp-draft-tokens 0 retains draft state and captures width-one rounds."))
        lines.append("  --mtp-only omits MTP-disabled graphs; the package requires speculation at runtime.")
        if speculation.token_capacity is None:
            lines.extend(("  --speculative-token-capacity: at least drafts + 1; default is the larger of",
                          "    power-of-two padded concurrency and drafts + 1."))
        else:
            lines.append(f"  --speculative-token-capacity: at least drafts + 1; default {speculation.token_capacity}.")
        if speculation.packed_verification:
            lines.extend(("    Target rows are shared across retained request prefixes.",
                          "    Packed rounds need more target rows than active requests."))
            if speculation.graph_budget is not None:
                lines.extend((f"  --speculative-graph-budget: positive integer; default {speculation.graph_budget}.",
                              "    Bounds packed round graphs; excludes mandatory width-one and ordinary graphs."))
        else:
            lines.append("    Wider graphs require batch * width <= capacity (width = draft count + 1).")
        layout = "packed" if speculation.packed_verification else "rectangular"
        lines.extend(("    Width one covers every batch.",
                      f"  Prepared verification: {layout} rows with {speculation.draft_prefix_selection.upper()}"
                      " prefix selection."))
    else:
        lines.append("  Multi-token speculation: unavailable.")
    optional = []
    for encoding, compressed in checkpoint.scale_compressed_device_bytes.items():
        saving = 100 * (1 - compressed / checkpoint.device_bytes)
        optional.append(f"--expert-scale-compression {encoding}"
                        f" (estimated {saving:.0f}% less resident weight storage; actual savings vary)")
    if recipe.kv_cache_formats:
        default_tp = recipe.checkpoint_tensor_parallel(checkpoint)[0]
        default_page = recipe.costs(default_tp, recipe.speculation, checkpoint).page_bytes
        plain_default_page = (
            recipe.costs(default_tp, None, checkpoint).page_bytes
            if recipe.speculation is not None else None
        )
        for name in recipe.kv_cache_formats:
            alt_page = recipe.costs(default_tp, recipe.speculation, checkpoint, kv_cache_format=name).page_bytes
            spec_saving = round(100 * (1 - alt_page / default_page)) if default_page > 0 and alt_page < default_page else None
            plain_saving = None
            if plain_default_page is not None and plain_default_page > 0:
                alt_plain = recipe.costs(default_tp, None, checkpoint, kv_cache_format=name).page_bytes
                if alt_plain < plain_default_page:
                    plain_saving = round(100 * (1 - alt_plain / plain_default_page))
            if spec_saving is not None and plain_saving is not None and plain_saving != spec_saving:
                optional.append(
                    f"--kv-cache-format {name}"
                    f" (TP{default_tp}: estimated {spec_saving}% smaller KV pages,"
                    f" {plain_saving}% with --no-mtp; actual savings vary)"
                )
            elif spec_saving is not None:
                optional.append(
                    f"--kv-cache-format {name}"
                    f" (TP{default_tp}: estimated {spec_saving}% smaller KV pages; actual savings vary)"
                )
            elif plain_saving is not None:
                optional.append(
                    f"--kv-cache-format {name}"
                    f" (TP{default_tp} --no-mtp: estimated {plain_saving}% smaller KV pages; actual savings vary)"
                )
            else:
                optional.append(f"--kv-cache-format {name}")
    if recipe.mixed_prefill:
        optional.append("--mixed-prefill (requires concurrency >= 2)")
    if any(cost.page_tokens for cost in recipe.ranks.values()):
        optional.append("--no-context-parallel")
    if recipe.stream_overlap:
        optional.append("--no-stream-overlap")
    lines.append("  Optional features for this recipe:")
    lines.extend(f"    {feature}" for feature in optional)
    if not optional:
        lines.append("    none")
    lines.extend(("  Images per request: bounded by the prepared image-token budget" if recipe.max_images else
                  "  Images: unavailable.",
                  "  Defaults above come from the recipe; configuration and explicit flags override them.",
                  "  Supported GPU counts do not guarantee that the selected capacities fit in memory."))
    return "\n".join(lines)


class RecipeHelp(argparse.Action):
    """Show settings for a MODEL preceding --help without resolving or loading it."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None,
                 recipes=(), prog="ignition"):
        self._recipes = tuple(recipes)
        self._prog = prog
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.epilog = recipe_help(getattr(namespace, "model", None), recipes=self._recipes, prog=self._prog)
        parser.print_help()
        parser.exit()
