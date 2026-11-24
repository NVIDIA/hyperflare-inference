# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch pinned upstream templates and prepare external fixtures for frontend tests."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH_ROOT = ROOT / "helios/tests/fixtures/templates"


def _annotate(source: bytes, patch_path: Path) -> bytes:
    patch = json.loads(patch_path.read_text())
    digest = hashlib.sha256(source).hexdigest()
    if digest != patch["source_sha256"]:
        raise ValueError(
            f"checkpoint chat template does not match {patch_path}: its SHA-256 is {digest},"
            f" and the patch applies to {patch['source_sha256']}"
        )
    parts, previous = [], 0
    for offset in patch["insertions"]:
        if type(offset) is not int or not previous <= offset <= len(source):
            raise ValueError(f"invalid template insertion offset in {patch_path}")
        parts.extend((source[previous:offset], b"{%- mark -%}"))
        previous = offset
    contents = b"".join((*parts, source[previous:]))
    if hashlib.sha256(contents).hexdigest() != patch["result_sha256"]:
        raise ValueError(f"patched chat template digest does not match {patch_path}")
    return contents


def main():
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hub-cache", type=Path)
    parser.add_argument("--recipe", action="append", help="Limit to the named recipe(s)")
    args = parser.parse_args()
    for patch_path in sorted(PATCH_ROOT.glob("*/*.patch.json")):
        recipe = patch_path.parent.name
        if args.recipe and recipe not in args.recipe:
            continue
        patch = json.loads(patch_path.read_text())
        source = patch["source"]
        path = hf_hub_download(source["repository"], source["filename"],
                               revision=source["revision"], cache_dir=args.hub_cache)
        original = Path(path).read_bytes()
        to_patch = original
        if recipe == "step5":
            opening, closing = b"{%- generation %}", b"{%- endgeneration %}"
            if original.count(opening) != 1 or original.count(closing) != 1:
                raise ValueError("Step-5 chat template must contain one assistant generation span")
            to_patch = original.replace(opening, b"{%- if true %}").replace(closing, b"{%- endif %}")
        annotated = _annotate(to_patch, patch_path)
        output = args.output / recipe
        output.mkdir(parents=True, exist_ok=True)
        name = patch_path.name.removesuffix(".patch.json")
        (output / f"{name}.source.jinja").write_bytes(original)
        (output / f"{name}.jinja").write_bytes(annotated)
        print(output / f"{name}.jinja")


if __name__ == "__main__":
    main()
