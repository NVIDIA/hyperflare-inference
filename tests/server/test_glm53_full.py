# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 text acceptance against an exported core served by ``flare run``.

Set FLARE_RUN_CUDA_TESTS=1, FLARE_GLM53_FULL_TEST_URL, and
FLARE_GLM53_FULL_REFERENCE, the fixture directory that
``tests/recipes/glm53_full_support.py`` describes. These tests never launch a
diagnostic Python server.

The reference is the Transformers forward in FP32 over fixed token sequences.
Served arithmetic is BF16 with FP8 latent records, and the Transformers forward
in that arithmetic already changes greedy tokens of the FP32 run: 29 of the
1516 rows of the ``long`` fixture whose top-two logit margin is at least 1.0,
and the log-probability of the FP32 run's token by 0.153 on average over the
last 64 rows. Bounds are twice those measurements.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import urllib.request
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("FLARE_GLM53_FULL_TEST_URL")
    or not os.environ.get("FLARE_GLM53_FULL_REFERENCE"),
    reason="requires an exported GLM 5.3 core running through flare run and the reference fixtures",
)

CHANGED_DECISIVE_TOKENS, LOGPROB_FLOOR = 29 / 1516, 0.153
PAGE_TOKENS = 256


def request(path, body=None):
    url = os.environ["FLARE_GLM53_FULL_TEST_URL"].rstrip("/") + path
    payload = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {os.environ.get('FLARE_API_TEST_KEY', 'local')}"}
    with urllib.request.urlopen(urllib.request.Request(url, payload, headers), timeout=900) as response:
        return json.load(response)


@pytest.fixture(scope="module")
def card():
    cards = request("/v1/models")["data"]
    assert len(cards) == 1
    return cards[0]


def fixture(name):
    import torch

    return torch.load(Path(os.environ["FLARE_GLM53_FULL_REFERENCE"]) / f"{name}.pt")


def next_token(card, token_ids):
    """Most likely next token after `token_ids`, the 20 most likely tokens, and the prompt's cached tokens.

    The recipe's declared temperature range excludes greedy sampling, so the most likely token is read from
    the returned alternatives rather than from the sampled token.
    """
    result = request("/v1/completions", {"model": card["id"], "prompt": token_ids, "max_tokens": 1,
                                         "logprobs": 20, "return_tokens_as_token_ids": True})
    choice = result["choices"][0]
    alternatives = {int(token.removeprefix("token_id:")): value
                    for token, value in choice["logprobs"]["top_logprobs"][0].items()}
    return max(alternatives, key=alternatives.get), alternatives, result["usage"]["prompt_tokens_details"]["cached_tokens"]


def test_card_declares_the_recipe(card):
    assert card["id"] == "GLM-5.3"
    flare = card["flare"]
    assert flare["vocab_size"] == 154880
    assert flare["sampling_defaults"]["temperature"] == 1.0 and flare["sampling_defaults"]["top_p"] == pytest.approx(0.95)
    assert flare.get("input_modalities", ["text"]) == ["text"]


def test_greedy_tokens_follow_the_reference(card):
    reference = fixture("long")
    token_ids, expected, margins = (reference[name].tolist() for name in ("token_ids", "next_tokens", "margins"))
    rows = len(token_ids)
    final = range(rows - 64, rows)
    sampled = sorted({*range(31, rows, 8), *final})
    logprobs = reference["logits"].float().log_softmax(-1)
    changed, decisive, differences, cached = [], 0, [], []
    for row in sampled:
        token, alternatives, reused = next_token(card, token_ids[:row + 1])
        cached.append((row + 1, reused))
        if margins[row] >= 1.0:
            decisive += 1
            if token != expected[row]:
                changed.append((row, margins[row]))
        if row in final:
            served = alternatives.get(expected[row])
            exact = float(logprobs[row - (rows - 64), expected[row]])
            # A reference token outside the 20 served alternatives counts as the worst served alternative.
            differences.append(abs((min(alternatives.values()) if served is None else served) - exact))
    assert len(changed) <= math.ceil(2 * CHANGED_DECISIVE_TOKENS * decisive), (changed, decisive)
    assert sum(differences) / len(differences) < 2 * LOGPROB_FLOOR, differences
    # Each prompt extends the previous one. The checkpoint policy chooses how much of it is reused, so the
    # tokens above are compared after reuse without fixing its amount.
    assert all(reused <= length for length, reused in cached), cached
    assert any(reused for _, reused in cached), cached


def test_short_sequence_predicts_the_reference_token(card):
    reference = fixture("short")
    token, _, _ = next_token(card, reference["token_ids"].tolist())
    assert token == int(reference["next_tokens"][-1])


def chat(card, messages, **fields):
    return request("/v1/chat/completions", {"model": card["id"], "messages": messages, **fields})


def test_repeated_and_changed_prompts_reuse_the_prefix(card):
    # A fresh tag keeps prefixes cached by earlier runs against the same service from matching.
    tag = uuid.uuid4().hex[:8]
    document = " ".join(f"Item {index}: sector {index % 17} holds {index * 7 % 113} crates of part {tag}-{index:04d}."
                        for index in range(300))
    system = {"role": "system", "content": "You are a precise inventory assistant.\n" + document}

    def ask(item):
        result = chat(card, [system, {"role": "user", "content": f"How many crates does Item {item} hold? "
                                                                 "Reply with the number only."}], max_tokens=400)
        usage = result["usage"]
        return result["choices"][0]["message"]["content"], usage["prompt_tokens"], \
            usage["prompt_tokens_details"]["cached_tokens"]

    answer, prompt, _ = ask(137)
    assert str(137 * 7 % 113) in answer
    answer, repeated, reused = ask(137)
    assert str(137 * 7 % 113) in answer
    assert repeated == prompt and prompt // PAGE_TOKENS * PAGE_TOKENS - PAGE_TOKENS <= reused <= prompt
    # The changed question shares the system message and the start of the user message.
    answer, changed, reused_changed = ask(201)
    assert str(201 * 7 % 113) in answer
    # The checkpoint policy decides whether a checkpoint precedes the changed tokens, so only the upper
    # bound is fixed: reuse never extends into the question's differing tail.
    assert abs(changed - prompt) <= 2 and reused_changed <= changed - 8
    answer, _, reused_again = ask(201)
    assert str(201 * 7 % 113) in answer
    assert changed // PAGE_TOKENS * PAGE_TOKENS - PAGE_TOKENS <= reused_again <= changed


def test_declared_concurrency_completes(card):
    concurrency = card["flare"]["max_concurrency"]

    def ask(index):
        left, right = 100 + index, 17 + 3 * index
        result = chat(card, [{"role": "user", "content": f"Compute {left} + {right}. Reply with the number only."}],
                      max_tokens=400)
        return str(left + right) in (result["choices"][0]["message"]["content"] or ""), result["choices"][0]

    with ThreadPoolExecutor(concurrency) as executor:
        results = list(executor.map(ask, range(2 * concurrency)))
    assert all(correct for correct, _ in results), [choice for correct, choice in results if not correct]
