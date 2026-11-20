# Parallax predictor contract, cost ledger, and experiment design

Parallax runs compiled expert-demand predictors inside Flare, evaluates their
forecasts independently, and can submit bounded Grace/HBM exchanges through the
native residency backend. This guide defines the observational cost ledger, the
compiled predictor contract, the `uniform_canary` control, authored synthetic
workloads, and the controlled experiment methodology.
[Architecture](parallax.md) describes the subsystem boundaries;
[running Parallax](parallax-demo.md) covers cores, profiles, and serving flags.

## Present qualification status

CPU tests cover offline replay, causal timing boundaries, predictor worker
lifecycles, shadow residency accounting, cost-ledger least-squares arithmetic,
native residency transaction ownership and rollback, and the authored HTTP
workload driver. Component GPU tests cover captured routing counters,
virtual-memory and row-map storage exchanges, and isolated single-layer MoE
graph replay on GB300 SM103.

Full-model GLM-5.3-Flash routing capture, default-off overhead checks,
fixed-MTP3 and adaptive-MTP5 exact-response parity, and controlled
static-versus-active serving throughput comparisons on the current core runtime
are not yet qualified; open items are tracked in [TODO.txt](TODO.txt).

## What the cost ledger means

The [native ledger](../helios/src/server/parallax/ledger.rs) fits graph GPU time
against total distinct expert weight bytes and distinct cold expert weight bytes,
with an intercept. Fits are separate for graph, batch size and target-row count.
Each fit retains up to 256 uncontended samples and needs at least 32 samples plus
sufficient independent variation. A frame is scored using prior samples before
it updates the fit. Missing fits remain unpriced rather than becoming zero cost.

The actual reader generation identifies each frame's placement. The ledger asks
how its executed canonical expert selections would have differed under the
initial static placement. This avoids charging repeated routes as independent
full expert loads, but remains an observational estimate: bytes are representation
extents, not measured memory traffic, and routing can correlate with other work.

The accounting balance subtracts all measured GPU copy time and host publication
time from estimated execution savings. Copy overlap, interference, and predictor
overhead are not fully attributed. Reader-drain gate time is reported separately
because inference can still be doing useful work during it. Startup movement is
excluded. Promotions still awaiting use are censored observations, not proven
waste.

A positive ledger balance is an observational hypothesis about useful placement,
not measured wall time saved or a substitute for controlled serving
measurements. Startup copy calibration alone cannot price direct Grace execution
under inference contention.

## Native predictor contract

The implementation is in
[`prediction.rs`](../helios/src/server/parallax/prediction.rs). A provider is
compiled into Flare and selected by a runtime profile; no Python, dynamically
loaded library, or separate service is needed for the statistical providers.

```rust
pub trait Predictor: Send + 'static {
    fn version(&self) -> &'static str;
    fn predict(&mut self, context: &Context)
        -> Result<Vec<(Expert, f64)>, String>;
}
```

| Input or output | Meaning and obligation |
| --- | --- |
| `Context.model` | Immutable catalog: model identity, architecture, sorted canonical expert IDs and parallel byte sizes. Live profile admission binds identity to the authenticated package ID. Never infer contiguous IDs or physical slots. |
| `Context.history` | Up to eight immutable eligible routing observations, oldest first. Each records sequence, demand and availability times, phase, batch size, concurrency, and expert demand counts. Providers may retain bounded state of their own. |
| `target_offset` | Future routing-frame offset from the last eligible observation. This is not a token offset; an MTP verification frame contains multiple candidates. |
| `horizon_ns` | Configured estimate of time until demand, not knowledge of future scheduling. |
| `available_ns` | Observation consumer availability in the native monotonic time domain. Host dispatch is a demand lower bound, not the precise GPU expert-access timestamp. Native readback and observation-consumer delay are separate from predictor compute. |
| Returned scores | Unique catalog experts with finite nonnegative scores; at most one entry per catalog expert. Empty output is permitted. Scores are not automatically probabilities. |
| Provider identity | Registry name, provider version and validated JSON configuration are retained with forecasts/profile. Change the version when forecast semantics change. |

The native recent/EWMA/frequency providers return demand-count estimates. An
economic policy needs meaningful score magnitudes; a ranking policy only needs
an ordering. Document units and suitability before substituting a ranking,
probability or neural output. The Rust contract does not yet carry a typed score
kind, per-score uncertainty or learned-artifact compatibility metadata. A provider
must reject an incompatible catalog explicitly; package binding alone does not
validate a learned artifact.

The observatory wraps scores in an immutable `Forecast`: provider identity,
observation cutoff, target frame, availability, worker start, completion,
computation duration, estimated demand time and expiration. Expected demand is
consumer availability plus `horizon_ns`; expiration is availability plus twice
that horizon. Providers do not assign timestamps or choose the scorer. Placement
policies and residency execution consume forecasts separately.

Settings version 1 allows one through four unique provider names and one shared
execution budget. Each provider has one outstanding job. Busy providers skip
observations and accumulate `dropped_busy`; this difference in feature exposure
must be included in comparisons. A panic, invalid forecast, error or timeout
disables the provider. A hung native thread cannot be safely killed; it is not
joined or replaced. Compiled plugins are trusted code, not a resource sandbox.
Keep their allocations and work bounded, and never perform synchronous GPU work
on an inference thread.

The same eligible observation is offered to each available provider. Actual
selections are scored independently against each contender's virtual residency,
pending transfers and movement history. Sequence cutoffs prevent future routing
from entering a forecast; completion timing determines whether it was actionable.
Observation loss invalidates comparison and suppresses active proposals. A failed
active contender stops further proposals; already valid placement remains usable.

The native source interface is not a stable binary plugin ABI. Native settings
and capture version 1, evaluation version 2, and the Python research trace schemas
are distinct formats. Live hot swapping, per-provider frequency/budget settings,
continuous archival capture and a neural-service client are not implemented.

## Canary example and baseline choices

[`canary.rs`](../helios/src/server/parallax/canary.rs) is a complete minimal
provider. It assigns one unit of demand to every catalog expert, regardless of
routing history. Its factory accepts only empty configuration; `Registry::builtins`
registers it as `uniform_canary`, version `1`. It exercises factory admission,
worker execution, forecast validation and common scoring without learning anything.

The canary is an uninformed control, not a contender expected to improve serving.
Equal scores imply no strict preference for the active `predictor_rank` policy,
so it should propose no rank-driven exchanges. Top-K evaluation resolves ties
deterministically; any coverage comes from that tie ordering and workload, not
predictive intelligence. Its full-catalog output consumes more retention space
than a sparse predictor and can shorten the replayable prefix.

To compare all four built-ins, use this `predictors` list in a valid profile from
the [serving guide](parallax-demo.md#observe-serving-traffic):

```json
[
  {"name": "uniform_canary", "config": {}},
  {"name": "recent", "config": {}},
  {"name": "ewma", "config": {"alpha": 0.5}},
  {"name": "frequency", "config": {"warmup_frames": 128}}
]
```

Omit `active` for a shadow comparison and choose a fresh profile output directory.
`recent` repeats the most recent demand. `ewma` folds the bounded observation
history with decay; it is not an unlimited lifetime average. `frequency` accumulates
distinct eligible observations until its warmup count and then freezes. Separate
its fitting period from held-out evaluation. Static physical placement is another
control, not another name for the frequency provider.

To add a provider, place its implementation beside `canary.rs`, declare the module
in `server/parallax.rs`, and register a validated factory in `Registry::builtins`.
Test sparse canonical IDs, score units, empty/invalid inputs, bounded state and
the observer/replay path. Rebuild the release Flare binary; adding a provider does
not require a model re-export when the existing package supplies its observations.
The canary tests cover invalid configuration, independence from observed popularity,
simultaneous evaluation with `recent`, and exact retained-capture replay.

## Authored workloads

[`scripts/parallax/workloads.py`](../scripts/parallax/workloads.py) supplies
stdlib-only HTTP traffic to an already running Flare server. It is a client, not
a serving dependency or prediction process. All prompts, histories and tool
results are authored synthetic fixtures. Generated code and tool calls are never
executed. These cases validate experiments and expose workload differences; they
do not establish agent capability or representative production performance.

| Selection | Traffic | Question to investigate |
| --- | --- | --- |
| `code` | Python cache, Rust parser, SQL design | Does the coding result persist across languages and tasks? |
| `agentic` | Source inspection, recorded test failure, patch and review histories | Do repeated repository/tool prefixes change locality and usable lookahead? |
| `prose` | Exposition, fiction, editing | Does routing differ between sustained composition and brief revision? |
| `chat` | Three turns of planning with added constraints | How does growing conversation context affect placement? |
| `long_context` | Fictional service catalog with dispersed retrieval targets | How do prefill and short-answer demand differ? |
| `shift` | Coding → prose → chat → coding blocks | How quickly does a predictor adapt, and does it retain stale experts? |

Agentic and chat histories are fixed before the run. Later requests use authored
assistant messages, not a preceding generated response. This preserves identical
inputs across policies but does not emulate a feedback-driven agent. Shift blocks
submit fixed conversation snapshots concurrently; they are not sequential turns
within that block. Identical session prefixes intentionally permit cache reuse.

From the repository root, export requests without contacting a server:

```sh
python3 -m scripts.parallax.workloads --workload agentic \
  --model example-model --copies 4 --concurrency 4 \
  --max-tokens 512 --output /tmp/parallax-agentic-requests
```

For live traffic, first launch a dedicated server following the serving guide.
Use its exact model ID from `GET /v1/models`; the loopback example below assumes
authentication is disabled for that dedicated local server. The client has no
authentication option. Replace the example URL and model ID as needed:

```sh
python3 -m scripts.parallax.workloads --workload shift \
  --base-url http://127.0.0.1:30153 --model MODEL_ID \
  --copies 4 --concurrency 4 --repeats 2 --max-tokens 512 \
  --output /tmp/parallax-shift-results
```

Output directories must not already exist. `--list` prints the workload names.
`--copies` creates distinct synthetic sessions; `--repeats` repeats the case set
with distinct session prefixes. Earlier stages finish before later ones begin.
`--concurrency` bounds client requests, not guaranteed model batch size.
`--records` controls the long-context catalog size; it is not a token count.
`--max-tokens` is a generation ceiling, not a fixed completion length. Read actual
usage before labeling a case “32K”; natural EOS and length stops are retained.

The client saves exact requests, raw responses/usage, errors, stage elapsed time,
model metadata, and before/after Parallax snapshots and drain gauges. It waits
for all five request/worker gauges to reach zero before and after traffic. This
does not exclude unrelated clients during the measurement; reserve the server.
The result's `passed` field means requests returned choices and positive completion
usage; inspect the saved Parallax errors and worker counters separately.
Failed stages have no successful throughput value and produce a nonzero exit.
Results already written survive a final drain failure. Rates include prefill and
HTTP; there is no streaming TTFT/ITL measurement or output-quality score.

These workload cases have CPU tests and have not been qualified against a GLM
core on a GPU; tool-history template acceptance and actual token-length checks
are tracked in [TODO.txt](TODO.txt).

## Public workloads to adapt

| Source | Useful coverage | Integration needed |
| --- | --- | --- |
| [SWE-bench](https://github.com/SWE-bench/SWE-bench) | Real repository issues and test-based resolution | An agent harness and isolated tools for actual trajectories; replayed histories measure serving, not issue-solving success. |
| [MT-Bench / FastChat](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) | Multi-turn chat, writing and reasoning | Pin selected questions and histories. Keep answer-quality judging separate from routing and serving measurements. |
| [LongBench](https://github.com/THUDM/LongBench) | Long-context understanding, retrieval and summarization | Pin a dataset/task version and measure token lengths with the served frontend. |
| [LongWriter](https://github.com/THUDM/LongWriter) | Long-form generation and its LongBench-Write evaluation | Select output-length ranges, retain termination reasons and evaluate writing quality separately. |

These are candidate sources, not downloaded fixtures or completed benchmark
results. Check dataset licenses, pin revisions and selected examples, and preserve
the prompts used for each condition. Start with authored traffic to qualify the
runner, then use held-out public cases rather than tuning and reporting on the
same requests.

## Controlled experiment plan

First compare canary, recent, EWMA and frozen frequency in shadow on each workload.
Report coverage, demand-weighted misses, completion lead, late forecasts, skipped
observations and predictor cost. Hold policy, capacity and transfer model constant.
Do not rank a provider by accuracy alone or score a late forecast as usable.

Then compare static and one active policy with identical initial placement and
request bodies. Begin with C1 and C4, fixed MTP3 and ordinary decode as separate
conditions. Reset process, residency and cache state between matched runs, record
warmup separately, and randomize repeated condition order. Placement and prefix
state persist, so switching labels in one live process is not an independent
comparison. Increase concurrency and adaptive MTP only after their correctness
and resource gates pass.

Report uncertainty across repeated run pairs or workload sessions, rather than
treating correlated routing frames as independent samples. Evaluate the cost fit
on later frames using its recorded prequential error, inspect residuals by shape
and phase, and retain unpriced frames in the denominator. The observational fit
does not provide causal confidence intervals. Any richer statistical model must
earn its complexity on held-out traffic and account for prediction cost.

Keep three outputs separate: prediction quality, modeled residency balance, and
measured serving performance. A useful result improves the last while preserving
the correctness gates, and explains where the first two did or did not predict it.
