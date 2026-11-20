# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify exact-source patching, HF selection, and exported template bytes."""

import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ignition.preparation.frontend import annotated_template_asset

ROOT = Path(__file__).resolve().parents[2]
PATCHES = sorted((ROOT / "ignition/recipes").glob("*/assets/*.patch.json"))


def fixture(recipe, filename):
    directory = os.environ.get("FLARE_TEST_TEMPLATE_DIR")
    assert directory, "run tests/frontend/prepare_templates.py and set FLARE_TEST_TEMPLATE_DIR"
    return Path(directory) / recipe / filename


def test_annotations_preserve_utf8_and_require_exact_source(tmp_path):
    source = "{{ messages[0].content }} café 世界".encode()
    expected = source + b"{%- mark -%}"
    patch = dict(source_sha256=hashlib.sha256(source).hexdigest(),
                 result_sha256=hashlib.sha256(expected).hexdigest(), insertions=[len(source)])
    path = tmp_path / "chat_template.patch.json"
    path.write_text(json.dumps(patch))
    assert annotated_template_asset(source, path) == {
        "text": expected.decode(), "sha256": hashlib.sha256(expected).hexdigest()}
    with pytest.raises(ValueError, match="checkpoint chat template does not match"):
        annotated_template_asset(source + b" ", path)
    patch["insertions"] = [0]
    path.write_text(json.dumps(patch))
    with pytest.raises(ValueError, match="patched chat template digest"):
        annotated_template_asset(source, path)
    for offsets in ([len(source) + 1], [-1], [True], [2, 1]):
        patch["insertions"] = offsets
        path.write_text(json.dumps(patch))
        with pytest.raises(ValueError, match="insertion offset"):
            annotated_template_asset(source, path)


@pytest.mark.parametrize("patch_path", PATCHES, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_recipe_patches_preserve_previously_exported_bytes(patch_path, tmp_path):
    recipe = patch_path.parent.parent.name
    filename = patch_path.name.replace(".patch.json", ".jinja")
    source = fixture(recipe, filename.replace(".jinja", ".source.jinja")).read_bytes()
    if recipe == "step5":
        from ignition.recipes.step5.frontend import template_asset

        (tmp_path / "chat_template.jinja").write_bytes(source)
        asset = template_asset(tmp_path)
        source = source.replace(b"{%- generation %}", b"{%- if true %}").replace(
            b"{%- endgeneration %}", b"{%- endif %}")
    else:
        asset = annotated_template_asset(source, patch_path)
    assert asset["text"].encode() == fixture(recipe, filename).read_bytes()
    assert asset["text"].encode().replace(b"{%- mark -%}", b"") == source


@pytest.mark.parametrize("hidden,filename,repository,revision", [
    (hidden, "chat_template.google-e2b-e4b", "google/gemma-4-E4B-it",
     "ee0ef6023621cff504d758262d4e04895a5af4a2") for hidden in (1536, 2560)
] + [
    (hidden, "chat_template.google", "google/gemma-4-26B-A4B-it",
     "4d7ae4984b7db7de8f8457170b3f1a419ee76d52") for hidden in (2816, 3840, 5376)
])
def test_gemma_fetches_pinned_canonical_template(hidden, filename, repository, revision, monkeypatch, tmp_path):
    from ignition.recipes.gemma4.frontend import chat_template_asset

    calls = []
    def download(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        return fixture("gemma4", filename + ".source.jinja")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    asset = chat_template_asset(hidden, cache_dir=tmp_path)
    assert calls == [(repository, "chat_template.jinja", dict(revision=revision, cache_dir=tmp_path))]
    assert asset["text"].encode() == fixture("gemma4", filename + ".jinja").read_bytes()
    with pytest.raises(ValueError, match="unsupported Gemma 4"):
        chat_template_asset(4096, cache_dir=tmp_path)


HUB = Path.home() / ".cache/huggingface/hub"
ZAI = dict(repository="zai-org/GLM-5.3-Flash", filename="chat_template.jinja",
           revision="46a050bec13002ad29ee795c16e61cf4fa3c4a9c")


@pytest.fixture
def calibrated_template(tmp_path):
    """The chat_template.jinja bytes of the calibrated checkpoint's pinned revision, in a snapshot directory."""
    bundled = (HUB / "models--local-inference-lab--GLM-5.3-Flash-NVFP4" / "snapshots"
               / "378ca54585c46542bad1f3cb3ed0d73ae51cdb62" / "chat_template.jinja")
    if not bundled.is_file():
        pytest.skip("the calibrated GLM checkpoint's chat template is not in the hub cache")
    (tmp_path / "chat_template.jinja").write_bytes(bundled.read_bytes())
    return tmp_path


def stub_download(monkeypatch, result):
    calls = []

    def download(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    return calls


def test_glm_nvidia_template_is_annotated_in_place(monkeypatch, tmp_path):
    from ignition.recipes.glm53 import frontend

    calls = stub_download(monkeypatch, fixture("glm53", "chat_template.zai.source.jinja"))
    (tmp_path / "chat_template.jinja").write_bytes(fixture("glm53", "chat_template.source.jinja").read_bytes())
    asset = frontend.chat_template_asset(tmp_path, cache_dir=tmp_path)
    assert asset["text"].encode() == fixture("glm53", "chat_template.jinja").read_bytes()
    assert not calls
    (tmp_path / "chat_template.jinja").write_bytes((tmp_path / "chat_template.jinja").read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not match") as error:
        frontend.chat_template_asset(tmp_path, cache_dir=tmp_path)
    assert hashlib.sha256((tmp_path / "chat_template.jinja").read_bytes()).hexdigest() in str(error.value)
    assert not calls


def test_glm_calibrated_template_is_replaced_by_the_canonical_source(monkeypatch, calibrated_template):
    """The real bytes of the calibrated checkpoint's template select the pinned canonical download."""
    from ignition.recipes.glm53 import frontend

    bundled = (calibrated_template / "chat_template.jinja").read_bytes()
    assert hashlib.sha256(bundled).hexdigest() == frontend.CALIBRATED_TEMPLATE_SHA256
    calls = stub_download(monkeypatch, fixture("glm53", "chat_template.zai.source.jinja"))
    asset = frontend.chat_template_asset(calibrated_template, cache_dir=calibrated_template)
    assert calls == [(ZAI["repository"], ZAI["filename"],
                      dict(revision=ZAI["revision"], cache_dir=calibrated_template))]
    assert asset["text"].encode() == fixture("glm53", "chat_template.zai.jinja").read_bytes()
    assert asset["text"].replace("{%- mark -%}", "").encode() != bundled
    # A single changed byte of the bundled template is neither replaced nor annotated.
    calls.clear()
    (calibrated_template / "chat_template.jinja").write_bytes(bundled + b" ")
    with pytest.raises(ValueError, match="does not match") as error:
        frontend.chat_template_asset(calibrated_template, cache_dir=calibrated_template)
    assert hashlib.sha256(bundled + b" ").hexdigest() in str(error.value)
    assert not calls


def test_glm_download_failure_says_how_to_seed_the_hub_cache(monkeypatch, calibrated_template):
    from ignition.errors import FetchError
    from ignition.recipes.glm53 import frontend

    stub_download(monkeypatch, OSError("offline"))
    with pytest.raises(FetchError, match=r"offline.*hf download zai-org/GLM-5.3-Flash chat_template.jinja"
                                         r" --revision 46a050bec13002ad29ee795c16e61cf4fa3c4a9c --cache-dir"):
        frontend.chat_template_asset(calibrated_template, cache_dir=calibrated_template)


def render(template, messages, **options):
    """Render with Transformers, the reference implementation of GLM's chat template."""
    from transformers.utils.chat_template_utils import render_jinja_template

    return render_jinja_template([messages], chat_template=template, add_generation_prompt=True, **options)[0][0]


def test_calibrated_and_canonical_templates_differ_only_on_null_content(calibrated_template):
    """Why the canonical template is required: the bundled copy writes the text "None" for null content.

    Tool-result matching is implemented differently (the bundled template scans every result for each call, the
    canonical one sorts) but renders the same prompts, so a prompt differs only through null content.
    """
    bundled = (calibrated_template / "chat_template.jinja").read_text()
    canonical = fixture("glm53", "chat_template.zai.source.jinja").read_text()
    assert hashlib.sha256(canonical.encode()).hexdigest() == json.loads(
        (ROOT / "ignition/recipes/glm53/assets/chat_template.zai.patch.json").read_text())["source_sha256"]
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}}}}

    def call(identifier):
        return {"id": identifier, "type": "function", "function": {"name": "lookup", "arguments": {"city": "Paris"}}}

    def conversation(content, calls, results):
        return [{"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": content, "tool_calls": [call(i) for i in calls]},
                *({"role": "tool", "tool_call_id": i, "content": f"result {i}"} for i in results)]

    # The only rendered difference: an assistant message without text.
    null_bundled = render(bundled, conversation(None, ["a"], ["a"]), tools=[tool])
    null_canonical = render(canonical, conversation(None, ["a"], ["a"]), tools=[tool])
    assert "<think></think>None<tool_call>" in null_bundled
    assert "<think></think><tool_call>" in null_canonical
    assert null_bundled.replace("</think>None<tool_call>", "</think><tool_call>") == null_canonical
    # Null content in any role renders as text in the bundled copy only.
    for role in ("system", "user"):
        messages = [{"role": role, "content": None}, {"role": "user", "content": "Weather?"}]
        assert "None" in render(bundled, messages) and "None" not in render(canonical, messages)
    # Tool results as one message of output entries, in call order and reordered.
    for results in (["a", "b"], ["b", "a"]):
        messages = [*conversation("", ["a", "b"], [])[:2], {"role": "tool", "content": [
            {"tool_call_id": i, "output": f"result {i}"} for i in results]}]
        assert render(bundled, messages, tools=[tool]) == render(canonical, messages, tools=[tool]), results
        assert render(canonical, messages, tools=[tool]).index("result a") < render(
            canonical, messages, tools=[tool]).index("result b")
    # Tool results in call order, reordered, matched by identifiers that differ in case, duplicated or unknown.
    for calls, results in ((["a", "b"], ["a", "b"]), (["a", "b"], ["b", "a"]), (["a", "A"], ["A", "a"]),
                           (["a", "b", "c"], ["c", "a", "b"]), (["a", "b"], ["a", "a"]), (["a"], ["zzz"]),
                           (["a", "b"], ["b"])):
        messages = conversation("", calls, results)
        assert render(bundled, messages, tools=[tool]) == render(canonical, messages, tools=[tool]), (calls, results)
        assert "result" in render(canonical, messages, tools=[tool])


def test_step_assets_preserve_checkpoint_and_validate_generation_span(tmp_path):
    from ignition.recipes.step5.frontend import runtime_assets, inference_template

    source = fixture("step5", "chat_template.source.jinja").read_bytes()
    checkpoint = tmp_path / "chat_template.jinja"
    checkpoint.write_bytes(source)
    assets = runtime_assets(tmp_path)
    expected = fixture("step5", "chat_template.jinja").read_bytes()
    assert assets["chat_template"] == {"text": expected.decode(), "sha256": hashlib.sha256(expected).hexdigest()}
    assert inference_template(tmp_path).encode() == expected
    assert assets["tokenizer"] == {"path": str(tmp_path / "tokenizer.json")}
    assert checkpoint.read_bytes() == source
    assert list(tmp_path.iterdir()) == [checkpoint]
    checkpoint.write_bytes(source + b" ")
    with pytest.raises(ValueError, match="checkpoint chat template does not match"):
        runtime_assets(tmp_path)
    checkpoint.write_bytes(source.replace(b"{%- generation %}", b""))
    with pytest.raises(ValueError, match="one assistant generation span"):
        runtime_assets(tmp_path)
