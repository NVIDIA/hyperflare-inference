# External Parallax tooling

Parallax shadow evaluation, predictor benchmarking, counterfactual replay,
authored HTTP workloads, and report viewing live outside Helios. Python tools
under [`scripts/parallax/`](../scripts/parallax/) run CPU-only prediction,
virtual residency simulation, and offline replay. The `flare parallax` CLI
subcommand displays saved or live JSON comparison reports using Flare's
terminal dashboard palette without loading a model or touching a GPU.

Helios serving does not import `scripts/parallax/` and does not run prediction
workers or move expert weights as part of these tools. All placement, hit-rate,
and net route-cost numbers produced by this suite are modeled estimates over
virtual HBM residency, not measured serving speedups.

## Components and data contracts

The shared immutable data types live in
[`scripts/parallax/contracts.py`](../scripts/parallax/contracts.py):

- `Expert(layer, expert)` identifies a canonical routed expert.
- `Model(identity, architecture, experts, expert_bytes)` defines the immutable
  catalog of canonical experts and their parallel byte sizes. Live sessions
  require `identity` to end with `;package=<package_id>`, where `<package_id>`
  comes from `flare inspect --json` (`core.core_id`).
- `Hardware(identity, hbm_bytes, backing_bytes, bandwidth_bytes_per_ns, transfer_latency_ns, synchronization_ns, cold_cost_ns)`
  specifies explicit storage capacity and transfer/cold-execution cost
  assumptions. Uncalibrated cost profiles must be labeled in `identity`.
- `Observation(seq, demand_ns, available_ns, demands, phase, batch_size, concurrency)`
  records one contiguous routing frame in integer nanoseconds within one
  trace-relative monotonic clock domain.
- `Forecast` and `PredictionRecord` preserve predictor scores, cutoff and target
  frame indices, expected demand and expiration timestamps, and separate queue,
  compute, and publication delays.

Compact JSONL trace and prediction serialization lives in
[`scripts/parallax/trace.py`](../scripts/parallax/trace.py). Traces reject
unsupported schema versions, unknown canonical experts, non-contiguous frame
sequences, and non-monotonic timestamps.

## Offline trace replay

[`scripts/parallax/__main__.py`](../scripts/parallax/__main__.py) replays
saved JSONL routing traces against the built-in `Static`, `Recent`, and `EWMA`
predictors, `NoMovement` and `Economic` policies, and idealized/constrained
oracle upper bounds on CPU:

```sh
python3 -m scripts.parallax --synthetic \
  --output /tmp/parallax-synthetic-comparison.json
```

For a recorded JSONL trace, pass the trace path and an explicit hardware cost
JSON file:

```sh
python3 -m scripts.parallax /path/to/routing.jsonl \
  --hardware /path/to/hardware.json \
  --horizon-ns 60000000 --target-offset 8 \
  --output /tmp/parallax-replay-comparison.json
```

All contenders share the same eligible observation stream and hold independent
virtual residency state. Forecast cutoffs prevent future observations from
entering a prediction, and timestamps enforce causal arrival before demand.

## External live collection and demo orchestration

[`scripts/parallax/live.py`](../scripts/parallax/live.py) polls a server
endpoint exposing `GET /v1/parallax/routing?after=<sequence>`, executes Python
predictors in isolated worker processes (`ShadowPool`), evaluates snapshots
asynchronously in a separate worker process (`SnapshotEvaluator`), and writes
`routing.jsonl`, `<predictor>.predictions.jsonl`, `leaderboard.json`, and
`comparison.json`.

[`scripts/parallax/demo.py`](../scripts/parallax/demo.py) wraps live collection
and offline replay:

- `python3 -m scripts.parallax.demo replay /path/to/capture [--hardware /path/to/alt-hardware.json]`
  re-evaluates a saved capture directory without a GPU. Without `--hardware`,
  replay verifies an exact match against `comparison.json` and rejects
  incomplete or inconsistent captures. With `--hardware`, it evaluates
  counterfactual cost assumptions without modifying the saved capture.
- `python3 -m scripts.parallax.demo live --config /path/to/config.json` starts
  an owned single-rank server, checks GPU occupancy before launch, attaches
  `scripts.parallax.live`, sends bounded demonstration requests, drains the
  server, and verifies replay. Live mode requires a server binary and package
  supporting `--parallax-routing` and `GET /v1/parallax/routing`, which are
  separate from this external tooling.

## Inspect or watch reports with `flare parallax`

`flare parallax` reads comparison JSON reports produced by external tooling and
renders them with Flare's terminal UI (`ratatui` / `crossterm`):

```sh
# Inspect a saved comparison report or capture directory without a GPU
flare parallax --report /path/to/capture/comparison.json
flare parallax --report /path/to/capture --plain

# Watch an active external session directory as updates arrive
flare parallax --watch /path/to/session-output
flare parallax --watch /path/to/session-output --plain
```

Tab and arrow keys switch between three panes:

1. **Prediction and modeled residency**: prediction count, precision, timely
   demand coverage, median usable lead time, virtual HBM hit rate, promotion
   count, and modeled balance or legacy net route-cost proxy.
2. **Predictor timing**: median and P95 compute duration, queue delay,
   publication delay, actionable rate, and late/expired forecast counts.
3. **Generated responses**: saved responses from `responses.json` when present
   in the session directory, with up/down scrolling.

While watching an active directory, the viewer polls `capture/live-comparison.json`
first (then `capture/leaderboard.json` and `capture/comparison.json`) so live
updates remain visible during the run, and preserves finalized reports once the
session completes.

## Reference decision vector generator

[`scripts/parallax/lru_reference.py`](../scripts/parallax/lru_reference.py)
extracts synthetic decayed-miss LRU decision vectors from a user-supplied local
checkout of `chronarion/derpseek` (`engine/derpseek_engine/residency.py`
`_pairs`) using NumPy, without importing a GPU engine or embedding DerpSeek
source code in this repository:

```sh
python3 -m scripts.parallax.lru_reference \
  /path/to/derpseek/engine/derpseek_engine/residency.py \
  --output /tmp/parallax_decayed_lru.json
```

## Validation

CPU unit and integration tests cover the external Python contracts, causal
timing boundaries, isolated shadow workers, virtual residency simulator, oracle
bounds, capture replay integrity, authored workloads, and the `flare parallax`
CLI/TUI:

```sh
CUDA_VISIBLE_DEVICES="" pytest \
  tests/test_parallax.py \
  tests/test_parallax_demo.py \
  tests/test_parallax_live.py \
  tests/test_parallax_workloads.py -q

CUDA_VISIBLE_DEVICES="" cargo test --locked \
  --manifest-path flare-cli/Cargo.toml parallax
```

These tests do not execute CUDA graphs or qualify live serving performance. See
[Parallax research contracts and experiment design](parallax-research.md) for
the predictor interface, cost model details, authored workloads, and controlled
experiment methodology.
