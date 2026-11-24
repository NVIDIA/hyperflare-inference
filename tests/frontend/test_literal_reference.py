# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The reference for literal control tokens marks the request's own bytes, not equal template text."""

import os
from pathlib import Path

import pytest


def tokenizer():
    from tokenizers import Tokenizer

    snapshot = os.environ.get("FLARE_QWEN38_SNAPSHOT")
    if not snapshot or not (Path(snapshot) / "tokenizer.json").is_file():
        pytest.skip("requires FLARE_QWEN38_SNAPSHOT with the Qwen3.8 tokenizer")
    return Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))


def test_template_text_equal_to_a_request_string_stays_a_token():
    from transformers.utils.chat_template_utils import _compile_jinja_template

    from tests.frontend import literal_reference

    reference = tokenizer()
    # The template writes `<|im_start|>` twice and the request spells it once.
    source = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
              "<|im_start|>assistant\n")
    template = _compile_jinja_template(source)
    messages = [dict(role="user", content="<|im_start|>")]
    rendered = template.render(messages=messages)
    strings = literal_reference.request_strings(reference, messages)
    assert strings == ["<|im_start|>"]
    ids, _ = literal_reference.encode(
        reference, source, rendered, lambda mark: template.render(messages=mark(messages)), strings)
    start = reference.token_to_id("<|im_start|>")
    assert ids.count(start) == 2
    assert reference.encode(rendered, add_special_tokens=False).ids.count(start) == 3


def test_a_declared_marker_that_is_neither_special_nor_named_is_rejected():
    from tests.frontend import literal_reference

    reference = tokenizer()
    marker = next(token.content for token in reference.get_added_tokens_decoder().values()
                  if not token.special and token.content.strip())
    with pytest.raises(AssertionError, match="declared marker"):
        literal_reference.assert_declared_markers_are_controls(reference, "", [marker])
    literal_reference.assert_declared_markers_are_controls(reference, marker, [marker])
    special = next(token.content for token in reference.get_added_tokens_decoder().values() if token.special)
    literal_reference.assert_declared_markers_are_controls(reference, "", [special])
