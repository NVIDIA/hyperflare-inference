# Parallax predictor contract, cost model, and experiment design

This guide defines the external Parallax predictor contract, virtual residency
cost model, authored HTTP workloads, candidate public workloads, and controlled
experiment methodology implemented under [`scripts/parallax/`](../scripts/parallax/).
[External Parallax tooling](parallax-demo.md) covers command invocations,
offline replay, and the `flare parallax` report viewer.

## Qualification status

CPU tests verify offline trace replay, causal observation and forecast timing
boundaries, isolated predictor worker processes (`ShadowPool`), virtual HBM
residency accounting (`VirtualResidency`), idealized and constrained oracle
bounds, capture replay integrity, the authored HTTP workload driver, and the
`flare parallax` terminal viewer.

No live GPU routing capture, physical weight movement, or end-to-end serving
throughput comparison is part of this external tooling or qualified by these
tests.

## Virtual residency cost model and ledger

[`scripts/parallax/simulator.py`](../scripts/parallax/simulator.py) simulates an
HBM cache backed by full immutable expert storage, and
[`scripts/parallax/policy.py`](../scripts/parallax/policy.py) defines
`NoMovement` and greedy `Economic` placement policies.

At each routing frame:

- `VirtualResidency` advances completed copies up to `demand_ns` before
  recording hits and misses against its current resident set.
- Each missed expert demand incurs `Hardware.cold_cost_ns` (explicit modeled
  cold-execution penalty).
- Each promoted expert reserves HBM bytes immediately, evicts victims if
  required, serializes onto the transfer queue for
  `ceil(size / bandwidth_bytes_per_ns) + transfer_latency_ns + synchronization_ns`,
  and becomes resident only when the transfer finishes.
- Promotions arriving after demand are counted as late prefetch demands;
  promotions evicted before any demand uses them are counted as wasted
  prefetches; evictions followed by later misses are counted as eviction reuses.

[`scripts/parallax/replay.py`](../scripts/parallax/replay.py) reports
`estimated_net_route_cost_saved_ns` as the static baseline cold cost minus the
contender's modeled cold, transfer, synchronization, and conservative
producer-delay proxy costs. All copy time is charged in full without modeling
compute overlap or memory contention. A positive modeled net value is an
accounting hypothesis under explicit cost assumptions, not measured wall time
saved or a serving throughput claim.

[`scripts/parallax/oracle.py`](../scripts/parallax/oracle.py) computes two
reference upper bounds on the same trace and hardware profile:

- `idealized`: future-knowing oracle with full lookahead.
- `constrained_heuristic`: causal-budget upper bound restricted to feasible
  transfers within the hardware bandwidth and HBM capacity.

## External predictor contract

The Python predictor protocol is defined in
[`scripts/parallax/contracts.py`](../scripts/parallax/contracts.py) and
implemented by `Static`, `Recent`, and `EWMA` in
[`scripts/parallax/predictors.py`](../scripts/parallax/predictors.py).
[`scripts/parallax/shadow.py`](../scripts/parallax/shadow.py) executes
registered predictors in isolated child processes with per-call timeouts and
busy-observation drop accounting.

| Input or output | Meaning and obligation |
| --- | --- |
| `Context.model` | Immutable `Model` catalog: identity, architecture, sorted canonical `Expert` IDs, and parallel `expert_bytes`. Workers reject snapshots whose catalog differs from registration. |
| `Context.history` | Immutable tuple of eligible prior `Observation` records, oldest first. Each records contiguous sequence index, demand lower bound, availability timestamp, phase, batch size, concurrency, and expert demand counts. |
| `Context.target_offset` | Positive future routing-frame offset from the last eligible observation. This is a frame offset, not a committed-token offset. |
| `Context.horizon_ns` | Configured estimate of time until demand, not knowledge of future scheduling. |
| `Context.available_ns` | Monotonic timestamp when the latest observation became available to predictors. |
| `Forecast.scores` | Unique catalog experts with finite nonnegative scores; at most one entry per catalog expert. Empty output is permitted. |
| `PredictionRecord` | Wraps an immutable `Forecast` with observation availability, worker start, worker completion, compute duration, and explicit `timing_source` (`modeled` for offline replay or `measured` for live worker processes). |

Each predictor worker has at most one in-flight job. Observations arriving
while a worker is busy are dropped and counted in `dropped_busy` rather than
queued indefinitely. Failures, invalid forecasts, and timeouts disable the
worker for the remainder of the session without blocking other contenders or
affecting saved observations.

## Authored workloads

[`scripts/parallax/workloads.py`](../scripts/parallax/workloads.py) is a
stdlib-only HTTP workload generator and traffic client. All prompts,
conversation histories, and tool results are authored synthetic fixtures;
generated code and tool calls are never executed. These workloads test routing
variation across task shapes and do not establish agent quality or
representative production performance.

| Selection | Traffic | Question to investigate |
| --- | --- | --- |
| `code` | Python cache, Rust parser, SQL design | Does routing locality persist across languages and tasks? |
| `agentic` | Source inspection, recorded test failure, patch and review histories | Do repeated repository and tool prefixes change locality and usable lookahead? |
| `prose` | Exposition, fiction, editing | Does routing differ between sustained composition and brief revision? |
| `chat` | Three turns of planning with added constraints | How does growing conversation context affect placement? |
| `long_context` | Fictional service catalog with dispersed retrieval targets | How do prefill and short-answer demand differ? |
| `shift` | Coding -> prose -> chat -> coding blocks | How quickly does a predictor adapt, and does it retain stale experts? |

Agentic and chat histories are fixed before the run. Later requests use
authored assistant messages rather than preceding generated output, preserving
identical inputs across conditions. Shift blocks submit fixed conversation
snapshots concurrently rather than as sequential turns within a block.

From the repository root, export deterministic requests without contacting a
server:

```sh
python3 -m scripts.parallax.workloads --workload agentic \
  --model example-model --copies 4 --concurrency 4 \
  --max-tokens 512 --output /tmp/parallax-agentic-requests
```

When `--base-url` is provided, the client targets a dedicated unauthenticated
server that exposes `/v1/models`, `/metrics`, `/v1/chat/completions`, and
`/v1/parallax/predictions`:

```sh
python3 -m scripts.parallax.workloads --workload shift \
  --base-url http://127.0.0.1:30153 --model MODEL_ID \
  --copies 4 --concurrency 4 --repeats 2 --max-tokens 512 \
  --output /tmp/parallax-shift-results
```

Output directories must not already exist. `--list` prints the workload names.
`--copies` creates distinct synthetic sessions; `--repeats` repeats the case
set with distinct session prefixes. Earlier stages finish before later stages
begin. `--concurrency` bounds client requests rather than guaranteeing a model
batch size. `--records` controls the long-context catalog size rather than a
token count. `--max-tokens` is a generation ceiling, not a fixed completion
length; natural EOS and length stops are retained.

The client saves exact requests, raw responses and usage, errors, stage elapsed
time, model metadata, and before/after drain gauges (`flare_active_requests`,
`flare_pending_requests`, `flare_retiring_request_slots`,
`flare_http_inflight_requests`, `flare_frontend_active_workers`). Failed stages
report `tokens_per_second: null` and exit nonzero. Rates include prefill and
HTTP latency; there is no streaming TTFT/ITL breakdown or output-quality score.

## Public workloads to adapt

| Source | Useful coverage | Integration needed |
| --- | --- | --- |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | Real repository issues and test-based resolution | An agent harness and isolated tools for actual trajectories; replayed histories measure serving, not issue-solving success. |
| [MT-Bench / FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) | Multi-turn chat, writing and reasoning | Pin selected questions and histories. Keep answer-quality judging separate from routing and serving measurements. |
| [LongBench](https://github.com/THUDM/LongBench) | Long-context understanding, retrieval and summarization | Pin a dataset/task version and measure token lengths with the served frontend. |
| [LongWriter](https://github.com/THUDM/LongWriter) | Long-form generation and its LongBench-Write evaluation | Select output-length ranges, retain termination reasons and evaluate writing quality separately. |

These are candidate external sources, not downloaded fixtures or completed
benchmark results. Verify dataset licenses, pin revisions and selected examples,
and preserve the exact prompts used for each condition.

## Controlled experiment plan

1. Compare predictors in shadow on each workload with identical policy,
   capacity, and transfer assumptions. Report precision, timely demand
   coverage, usable lead time, late and expired forecasts, busy drops, and
   predictor latency together; do not rank predictors by raw precision alone or
   count late forecasts as actionable.
2. Separate offline or virtual residency estimates from live serving
   measurements. Reset server process, residency, and prefix-cache state
   between matched runs, record warmup separately, and randomize repeated
   condition order.
3. Estimate uncertainty across repeated workload sessions or run pairs rather
   than treating correlated routing frames as independent samples. Keep
   prediction quality, modeled residency balance, and measured serving
   performance as separate outputs.
