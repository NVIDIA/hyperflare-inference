# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Collect next-token log probabilities from native Flare or reference vLLM.

The input JSON lists token-ID prefixes. Using the same prefixes on both
servers separates decoder arithmetic from chat rendering and random sampling.
"""

import argparse
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for prefix in json.loads(args.prefixes.read_text()):
        body = dict(model="inkling-2-mini", prompt=prefix, max_tokens=1, seed=1234,
                    logprobs=20, return_tokens_as_token_ids=True, ignore_eos=True)
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions",
            json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        records.append(result)
        top = result["choices"][0]["logprobs"]["top_logprobs"][0]
        print(f"tokens={len(prefix)} top={max(top, key=top.get)}", flush=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
