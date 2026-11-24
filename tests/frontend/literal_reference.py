# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference token IDs for prompts whose request text spells control tokens.

A request string that spells a control is ordinary text. A control is a special added token, a
token that the chat template or the tokenizer defaults name, or a marker that the prepared
frontend contract declares (reasoning delimiters and terminal markers, tool-call delimiters and
argument tags, image tokens). The reference encodes the string with Hugging Face tokenizers and
no other rule: the text between the template's own added tokens, around each request string, is
encoded with `Tokenizer.encode_special_tokens = True`, the setting that transformers applies for
`split_special_tokens=True`, so the tokenizer does not match special tokens inside it. That setting
leaves a non-special added token matched, so the reference holds only while every declared marker
is special or named: `encode` asserts it for the markers passed in `declared`. Template text that
touches the request string is encoded in the same region, so a merge across the boundary (for
example `>` followed by a newline) is the checkpoint tokenizer's own. The rest of the prompt is
encoded normally.

The request's own bytes are found by rendering the prompt a second time with private-use
characters around each request string, so an identical string that the template writes itself
stays template text.
"""

import re

# The tool-call markers of the parser kinds that the fixtures use, as `OutputPipeline::declared_markers`
# (helios/src/output/pipeline.rs) lists them.
QWEN_CODER_MARKERS = ("<tool_call>", "</tool_call>", "<function=", "</function>")
ATEM_MARKERS = ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>")

MARK_START = "\ue000"
MARK_END = "\ue001"


def request_strings(tokenizer, *values):
    """The strings in `values` (nested lists and dicts, keys excluded) that spell an added token."""
    contents = [token.content for token in tokenizer.get_added_tokens_decoder().values()
                if token.content.strip()]
    found = []

    def walk(value):
        if isinstance(value, str):
            if any(content in value for content in contents):
                found.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    for value in values:
        walk(value)
    return found


def _encode(tokenizer, piece, split):
    if not piece:
        return [], []
    tokenizer.encode_special_tokens = split
    try:
        encoding = tokenizer.encode(piece, add_special_tokens=False)
    finally:
        tokenizer.encode_special_tokens = False
    return encoding.ids, encoding.offsets


def _request_spans(text, render, strings):
    """The character ranges of `text` that hold the request strings, found with marked inputs."""

    def mark(value):
        if isinstance(value, str):
            return f"{MARK_START}{value}{MARK_END}" if value in strings else value
        if isinstance(value, dict):
            return {key: mark(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [mark(child) for child in value]
        return value

    assert MARK_START not in text and MARK_END not in text
    clean, spans, start = [], [], 0
    for character in render(mark):
        if character == MARK_START:
            start = len(clean)
        elif character == MARK_END:
            spans.append((start, len(clean)))
        else:
            clean.append(character)
    assert "".join(clean) == text, "the template changed a request string"
    assert spans, strings
    return spans


def declared_markers(tokenizer, frontend, *tool_markers):
    """The marker strings of the prepared frontend contract `frontend` (a dict) that the server
    adds to the controls: the reasoning delimiters and terminal markers, the image tokens, and
    `tool_markers`, the delimiters and argument tags of the tool-call parser, which the contract
    names only by kind."""
    reasoning = frontend.get("reasoning") or {}
    markers = [reasoning.get("open"), reasoning.get("close"), *reasoning.get("terminal_markers", []),
               *tool_markers]
    image = frontend.get("image_tokens") or {}
    span = image.get("span") or {}
    ids = [image.get("feature_token"), span.get("start_token"), span.get("end_token"),
           *span.get("interior_tokens", []), *span.get("unscoped_tokens", [])]
    markers += [tokenizer.id_to_token(token) for token in ids if token is not None]
    return [marker for marker in markers if marker]


def assert_declared_markers_are_controls(tokenizer, named, declared):
    """Every `declared` marker that is an added token is special or in `named`, the text of the
    chat template and the tokenizer defaults."""
    for token in tokenizer.get_added_tokens_decoder().values():
        assert token.special or token.content not in declared or token.content in named, (
            f"{token.content!r} is a declared marker that is neither special nor named: "
            "the server makes it a control but the reference would keep it a token")


def encode(tokenizer, named, text, render, strings, declared=()):
    """Returns the IDs and character offsets of the rendered prompt `text`.

    `tokenizer` is a `tokenizers.Tokenizer`; `named` is the text of the chat template and the
    tokenizer defaults, which makes a non-special added token a control that the reference
    cannot split, so `encode` asserts that no request string spells one and, through
    `assert_declared_markers_are_controls`, that every `declared` marker (see `declared_markers`);
    `strings` are the request strings that spell added tokens (see `request_strings`).
    `render(mark)` renders the same prompt with `mark(inputs)` applied to every input that holds
    a request string.
    """
    assert_declared_markers_are_controls(tokenizer, named, declared)
    added = [token for token in tokenizer.get_added_tokens_decoder().values() if token.content.strip()]
    strings = set(strings)
    if not strings:
        return _encode(tokenizer, text, False)
    spans = sorted(set(_request_spans(text, render, strings)))
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:])), spans
    for string in strings:
        for token in added:
            # `encode_special_tokens` leaves non-special added tokens matched, as for a lexical token.
            assert token.special or token.content not in string or token.content not in named, (
                f"{token.content!r} is a control that is not special: the reference would keep it a token")
    pattern = re.compile("|".join(re.escape(token.content)
                                  for token in sorted(added, key=lambda token: -len(token.content))))
    written = [match.span() for match in pattern.finditer(text)
               if not any(start <= match.start() < end for start, end in spans)]
    regions = []
    for start, end in spans:
        left = max((stop for _, stop in written if stop <= start), default=0)
        right = min((begin for begin, _ in written if begin >= end), default=len(text))
        if regions and left < regions[-1][1]:
            regions[-1] = (regions[-1][0], right)
        else:
            regions.append((left, right))
    ids, offsets, at = [], [], 0
    for start, end in [*regions, (len(text), len(text))]:
        for split, first, last in ((False, at, start), (True, start, end)):
            piece_ids, piece_offsets = _encode(tokenizer, text[first:last], split)
            ids += piece_ids
            offsets += [(begin + first, finish + first) for begin, finish in piece_offsets]
        at = end
    return ids, offsets
