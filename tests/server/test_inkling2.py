# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling 2 text, TML tools, images and cached continuations through native TP4 serving.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_INKLING2_TEST_URL to an exported core
running with flare run, concurrency eight, 4096-token prefill and vision.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import urllib.error
import urllib.request

import pytest

from .test_minimax_m3 import answer, image


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1" or not os.environ.get("FLARE_INKLING2_TEST_URL"),
    reason="requires a native Inkling 2 TP4 service",
)


@pytest.fixture(scope="module", params=[None, "none"], ids=["default-effort", "zero-effort"])
def service(request):
    url = os.environ["FLARE_INKLING2_TEST_URL"].rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]
    assert "inkling" in model["id"].lower(), model
    assert model["flare"]["max_concurrency"] == 8
    assert model["flare"]["prefill_chunk_size"] == 4096
    assert "image" in model["flare"]["input_modalities"]

    def ask(content, **options):
        body = dict(model=model["id"], messages=[dict(role="user", content=content)],
                    seed=1234, max_tokens=1024, return_token_ids=True)
        if request.param is not None:
            body["reasoning_effort"] = request.param
        body.update(options)
        http_request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                              {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(http_request, timeout=600) as response:
                if body.get("stream"):
                    events = []
                    for line in response:
                        if line.startswith(b"data: "):
                            payload = line.removeprefix(b"data: ").strip()
                            if payload == b"[DONE]":
                                return events
                            events.append(json.loads(payload))
                    pytest.fail("stream ended without [DONE]")
                result = json.load(response)
        except urllib.error.HTTPError as error:
            pytest.fail(f"HTTP {error.code}: {error.read().decode()}")
        choice = result["choices"][0]
        assert choice["finish_reason"] in ("stop", "tool_calls"), result
        assert result["usage"]["completion_tokens"] == len(choice["token_ids"])
        return result

    return ask


def test_text_and_eight_concurrent_requests(service):
    assert answer(service("What is 17 times 23? Reply with only the integer.")) == "391"
    prompts = [f"Repeat exactly this code and nothing else: FLARE{i}128" for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(service, prompts))
    for index, result in enumerate(results):
        assert answer(result) == f"FLARE{index}128"


def test_streamed_reasoning_and_answer(service):
    events = service("What is 17 times 23? Put only the integer in the final answer.",
                     reasoning_effort="low", stream=True, stream_options=dict(include_usage=True))
    choices = [choice for event in events for choice in event["choices"]]
    reasoning = "".join(choice.get("delta", {}).get("reasoning_content") or "" for choice in choices)
    content = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
    assert reasoning.strip()
    assert content.strip() == "391"
    assert choices[-1]["finish_reason"] == "stop"
    assert any(event.get("usage", {}).get("completion_tokens", 0) > 0
               for event in events if event.get("usage"))


def test_chunked_prefix_reuse_and_branching(service):
    prefix = " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(700))
    history = [dict(role="user", content=prefix), dict(role="assistant", content="I have the item records.")]
    for index in (211, 677, 211):
        messages = history + [dict(role="user",
            content=f"What code does item {index} record? Reply with only the integer.")]
        first, second = service("", messages=messages), service("", messages=messages)
        assert first["usage"]["prompt_tokens"] > 4096
        assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 4096
        assert answer(first) == answer(second) == str(index * 17 % 997)


def test_tml_tool_call_and_result(service):
    tools = [dict(type="function", function=dict(name="lookup_weather", description="Get a city's weather.",
        parameters=dict(type="object", properties=dict(city=dict(type="string")), required=["city"])))]
    prompt = "Use lookup_weather to get the weather in Paris."
    result = service(prompt, tools=tools, tool_choice="required")
    message = result["choices"][0]["message"]
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    call, = message["tool_calls"]
    assert call["function"]["name"] == "lookup_weather"
    assert json.loads(call["function"]["arguments"])["city"] == "Paris"
    result = service("", tools=tools, messages=[dict(role="user", content=prompt), message,
        dict(role="tool", tool_call_id=call["id"], content='{"temperature_c": 21, "condition": "sunny"}')])
    assert "21" in answer(result)


def test_structured_response(service):
    schema = dict(type="object", properties=dict(result=dict(type="integer")), required=["result"],
                  additionalProperties=False)
    result = service("Return the product of 17 and 23 as result.",
        response_format=dict(type="json_schema", json_schema=dict(name="product", strict=True, schema=schema)))
    assert json.loads(answer(result)) == {"result": 391}


def test_images_across_a_prefill_chunk(service):
    content = [dict(type="text", text="filler " * 4100),
        image("red", "FIRST 128"), image("blue", "SECOND 256"),
        dict(type="text", text="Read both image labels in their original order. Only output the labels.")]
    response = service(content)
    assert response["usage"]["prompt_tokens"] > 4096
    result = answer(response)
    assert "FIRST 128" in result and "SECOND 256" in result
    assert result.index("FIRST 128") < result.index("SECOND 256")


def test_four_images_and_four_concurrent_image_requests(service):
    labels = ("FIRST 128", "SECOND 256", "THIRD 512", "FOURTH 1024")
    pictures = [image(color, label) for color, label in
                zip(("red", "blue", "green", "orange"), labels, strict=True)]
    response = service([*pictures, dict(type="text", text="Read all four labels in order. Only output the labels.")])
    text = answer(response)
    assert all(label in text for label in labels), text
    assert [text.index(label) for label in labels] == sorted(text.index(label) for label in labels)
    prompts = [[picture, dict(type="text", text="Read the image label. Only output the label.")]
               for picture in pictures]
    with ThreadPoolExecutor(max_workers=4) as workers:
        responses = list(workers.map(service, prompts))
    for label, response in zip(labels, responses, strict=True):
        assert answer(response) == label
