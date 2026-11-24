# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time one uncached prompt through Flare with a one-token output budget."""

import argparse
import json
from pathlib import Path
import random
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--tokens", type=int, default=32768, help="number of random token IDs (default: 32768)")
    prompts.add_argument("--prompt-file", type=Path, help="send this UTF-8 text through the server tokenizer")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    url = args.url.rstrip("/")
    with urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]
    if args.prompt_file is None:
        if args.tokens + 1 > model["max_model_len"]:
            parser.error(f"prompt plus output exceeds the {model['max_model_len']}-token context")
        token_ids = range(min(8192, model["flare"]["vocab_size"]))
        prompt = random.Random().choices(token_ids[256:] or token_ids, k=args.tokens)
    else:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    body = dict(model=model["id"], prompt=prompt, max_tokens=1)
    request = Request(
        url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=1800) as response:
            result = json.load(response)
    except HTTPError as error:
        raise SystemExit(f"HTTP {error.code}: {error.read().decode()}") from error
    elapsed = time.perf_counter() - started
    usage = result["usage"]
    count = usage["prompt_tokens"]
    if args.prompt_file is None and count != args.tokens:
        raise SystemExit(f"Requested {args.tokens} prompt tokens, got {count}")
    if count < 1 or usage["completion_tokens"] != 1 or usage["prompt_tokens_details"]["cached_tokens"] != 0:
        raise SystemExit(f"Expected an uncached prompt and one output token, got {usage}")
    print(f"{count} prompt tokens, 1 output token, {elapsed:.3f} s, "
          f"{count / elapsed:.1f} prompt tok/s end to end")


if __name__ == "__main__":
    main()
