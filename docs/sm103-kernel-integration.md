# Integrating model execution on SM103

This guide describes how to adapt a model's execution path to SM103 while
preserving Flare's preparation and serving boundaries. Apply it when bringing a
model such as GLM or Qwen to GB300, or when replacing an existing path that is
incorrect or slow there. It specifies a development process; it does not claim
that every supported recipe is qualified on SM103.

A clean kernel path has compatible producers and consumers, preserves the
model's arithmetic and state, and performs only the work required by that
contract. A fast GEMM behind repeated layout conversions can be a slow path.
Selecting an SM103 implementation in Python does not establish which kernels
an exported core executes.

Functional SM103 execution is a correctness baseline, not a performance
qualification. Reassess choices inherited from SM120/SM121, including kernel
dispatch, activation encoding, weight layouts, tuning candidates, launch
geometry, scratch storage, graph dependencies and memory placement. Existing
fusions and fallback paths need the same complete-path measurements as proposed
replacements. Preserve the declared arithmetic and state contract while testing
whether each implementation suits SM103.

Read [model development](model-development.md),
[preparation development](preparation-development.md), and
[GPU performance work](performance-work.md) before making changes. Their
architecture, precision, capture, and measurement requirements apply throughout.

## Trace the complete operation

Start at the recipe's forward and follow an operation from its producer to its
consumer. Include checkpoint loading, activation preparation, output processing,
and state updates. Inspect the installed dependency implementation and the
selected preparation plan, including dispatch conditions at each relevant
shape. An architecture label or an available API is not evidence that a call
uses a suitable implementation.

Check that hardware dispatch describes the device owning the operands. CUDA
device masking renumbers visible devices; an NVML query of physical GPU zero
does not follow that renumbering. On a host with different GPU architectures,
derive CUDA execution choices from the selected CUDA device and verify the
captured kernel path on that device.

For each affected operation, establish:

| Boundary | What to inspect |
| --- | --- |
| Checkpoint to resident weights | Packed values, scale encoding and grouping, calibration, TP slicing, padding, permutations, and compressed-scale expansion |
| Producer to kernel | Activation precision, strides, alignment, quantization, masks, routing IDs, and index translation |
| Kernel to consumer | Output layout and precision, scale layout, residual or gating order, and repeated conversions |
| Invocation to persistent state | Cache writers/readers, page geometry, recurrent or compressor history, inactive rows, and speculative commit/rollback |
| Preparation to native replay | Selected programs, scratch bindings, captured arguments, retained resource owners, and export relocation |

Map target decode, prefill, speculative verification, and draft execution
separately. They can share weights while selecting different operators or
requiring different state transitions. Batch size and physical verification rows
can change dispatch even when the model and GPU are unchanged.

Compare the dependency's available implementations with the candidates the Flare
adapter actually admits. Autotuning cannot select an omitted implementation.
Check shape restrictions, candidate enumeration, selection-cache identity, and
the selected plan separately. Include concurrency one, two, and four with their
actual verification widths; a larger-batch comparison can miss a kernel intended
for only a few rows. Verify the selected implementation in captured launches.

When comparing another runtime, inspect its actual enabled path. Determine
which operations it fuses, which work it overlaps, what precision it uses, and
what storage it reads. A kernel with the same name may have different arguments,
layouts, or surrounding work. Account for those differences before attributing
a performance gap to scheduling or residency.

## Prepare a complete reference forward

When a model author provides an optimized inference implementation, evaluate its
complete forward as the preparation input before rebuilding its computation from
individual operators. Preserve compatible fusion, activation layouts and branch
dependencies across layers. Ignition owns construction, weight binding, warmup,
capture and export; Helios owns request scheduling and native replay.

The DeepSeek V4.1 `"execution": "reference"` path in `--sm103-preparation`
integrates the model computation in
[DeepInfer](https://github.com/deepseek-ai/DeepInfer), including its attention,
mHC, expert composition, DFlash draft model, vision encoder and vision-language
projector, as an experimental TP1 GB300 path. DeepInfer's CUDA graph runner
captures only hidden-state forwards; Ignition supplies the complete generation,
sampling, speculative-commit, prefix-snapshot, and export bindings.

The integration targets one GB300 with local expert and activation buffers.
It must not initialize DeepEP or NCCL. A later two-GPU composition would use
Flare's RoCEnante collectives, with communication prepared by Ignition and
captured for native replay. The reference computation does not select Flare's
communication stack.

Adapt these boundaries while retaining the reference computation:

| Boundary | Required Flare binding |
| --- | --- |
| Checkpoint and expert storage | Load the CSF checkpoint without requantizing weights. Preserve canonical expert identities and bind physical HBM/Grace rows through Parallax's existing storage and publication contracts. |
| Inputs and cache state | Construct fixed-capacity metadata from declared token, position, page-table and request-state inputs. Include sliding-window rollback history, compressor and indexer state, shared cache aliases and inactive-row destinations. |
| Engram delivery | Bind the reference's embedding consumers to declared native delivery buffers and readiness events, preserving hash geometry and scale layout. |
| Vision | Prepare the reference encoder and projector with the checkpoint's image processing, patch ordering and rotary geometry. Bind image features through Flare's existing vision graph and frontend contracts. |
| Generation | Compose target and draft computation with request-local sampling, accepted-state commit and completion publication inside the existing captured operation families. |
| Export | Retain and relocate all referenced allocations, tensor maps, synchronization buffers and kernel resources. Preserve graph forks and joins in a fresh native process. |

CSF storage and expert residency are independent requirements. The source scale
encoding does not determine the kernel's scale layout. Any lossless expansion
must have a bounded lifetime and an explicit memory charge in its destination
tier. Do not qualify an integration by silently materializing every expert and
scale in HBM. DeepInfer's native loader expects native scale tensors and rejects
the `mxfp4_csf` configuration; the integration must provide a real loading and
storage adaptation rather than relabeling the compressed checkpoint as native.

The experimental parameter loader materializes the complete text, draft and
vision parameter tree while leaving routed experts and Engram tables as storage
declarations. On SM103, all 1,219 ordinary parameter entries and their source
scales match the checkpoint exactly, and tied parameters share storage. This
qualifies parameter loading, not complete forward execution or serving.

The experimental model assembly constructs the original target, DFlash and
vision modules with explicit execution settings, Flare-owned expert planes and
native Engram delivery. Its SM103 construction check loads all 40 target layers,
three draft layers and the 32-block vision encoder, placing half of each layer's
routed experts in HBM and half in Grace. Target expert operations are prepared
at 1, 8, 192 and 8192 rows and draft operations at 1 and 7 rows. Parameter
construction is consuming: reference projection fusion transfers storage
ownership, so a failed assembly requires a fresh parameter tree.

On one GB300, the complete target transformer compiles and executes through
Helios's native graph executor with full checkpoint Engram tables in host RAM.
Synthetic prompts at prefill capacities 1, 192 and 8192, followed by four sampled
outputs per request, preserve finite hidden outputs, FP32 vocabulary logits and
complete request/cache state exactly against direct execution using the same
delivered table rows. Chunk sequences are 1; 129, 128, 3; and 8192, 129, 3 actual
tokens. They exercise padded rows, compression and page boundaries, continued
prefill and the transition into decode. Intermediate chunks consume tokens
without sampling; prompt histogram entries are counted once at admission.

The compiled graphs include the reference BF16 vocabulary projection,
target-layer feature storage, DFlash main-cache writes, Engram history commit
and shared FlashInfer sampling. Request RNG counters advance once per output;
the completion cursor reports draft context through the consumed target token.
Replay allocates no GPU storage and leaves the unassigned continuation row
unchanged. A queued decode after the output limit preserves request and cache
state. These ordinary-generation checks do not establish
long-context accuracy, cross-capacity numerical equivalence or fresh-process
core loading.

With seven proposed tokens on the same GB300 and split expert placement,
complete native speculative rounds preserve proposal IDs, draft and target
logits, filtered probabilities, request counters and model/cache bytes exactly
against direct execution until a twelve-output limit and a queued finished
round. The shared sampler's FP32 survival estimates satisfy its existing
relative bound of 1e-6 against direct execution and an independent FP64
calculation; token decisions and cache state retain exact comparisons. An
actual-model rollback check compares every accepted input count from zero
through eight with target-only execution of that prefix at the same physical
row capacity. These checks cover one request and synthetic prompt tokens;
they do not qualify concurrent speculative serving or fresh-process core loading.

The complete 32-layer reference vision encoder, spatial downsample and projector
compile and replay on GB300 with checkpoint weights. Normalized synthetic image
inputs with 24-by-24 and 25-by-31 patch grids match direct outputs exactly across
three changing inputs per shape, without GPU allocation during replay. Compiler
declarations retain TileKernels rotary/SwiGLU, FlashAttention and the reference
ATen projections; unbiased BF16 projections select DeepGEMM from the operand's
CUDA device. Native feature publication also passes at patch capacities 576,
1152 and 9216, including partial spatial groups, append offsets, stale request
generations and invalid rotary positions. Published features match the original
encoder, spatial merge and projector exactly; rejected writes preserve existing
features. Replay allocates no GPU storage. Native image preprocessing matches
DeepInfer's processor and GPU normalization exactly for tiny, landscape,
extreme-aspect-ratio and multiple-image inputs. The comparison covers token
types, feature indices, atomic spans, patch geometry, positions and BF16 pixels.

With the complete checkpoint and half of each expert layer in HBM, a native
vision graph publishes features from a synthetic 24-by-24 patch image for
chunked text execution. Prefill chunks of 129 and 3 tokens, four ordinary sampled
outputs and subsequent seven-token speculative rounds through a twelve-output
limit preserve outputs and complete state against direct execution. The image
span ends at the first chunk boundary; Engram history retains image sentinels
and excludes them from subsequent text n-grams. Learned image-marker and feature
injection matches the original reference merge operation exactly. These checks
cover one request and one image; they do not establish image understanding,
concurrent image serving or fresh-process core loading.

The reference cached-prefix sampler uses the original BF16 vocabulary projection
and shared request validation/completion. Native replay at one and four rows
with synthetic saved hidden states produces the expected greedy tokens without
consuming input, changing cache bytes or touching inactive requests. Complete
continuation snapshots and forks preserve all reference cache regions and partial
pages at endpoints 1, 127, 128, 129, 255 and 256. Both checks allocate no GPU
storage during replay. Full-model generation after native prefix restoration
remains unqualified.

The target-feature storage binding preserves strided destinations and adjacent
feature planes during compiled replay at 1, 8, 129 and 8192 rows, for one- and
four-way residual inputs. These synthetic component checks compare three
changing inputs against the original native operation with no replay allocation.

Reference gather, BF16 projection and mHC operators expose native launches through
Torch operation declarations. On SM103, complete-forward compilation and captured
replay preserve exact output bytes for synthetic 5120-wide inputs at 1, 8 and 129
rows, including shifted and unshifted hyperconnections. Checks cover
FP8 normalization output, residual mixing, changing inputs and replay allocation
stability. Component checks qualify the operator bindings; complete-model
prefill checks at 8,192 rows verify numerical repeatability across direct and
captured execution (`tests/components/test_reference_model_cuda.py`).

Compiled reference attention transforms preserve cache bytes and BF16 outputs
at capacities 1, 8, 129 and 8192 for the 528-byte sliding-window, 288-byte
compressed-attention and 68-byte index-cache formats. Checks cover normalization,
rotary or Hadamard transforms, quantization and masked stores through the original
fused operator. CUDA memory checking reports zero invalid accesses for these
bindings and the mHC component cases; it does not establish numerical
repeatability or whole-model correctness.

The experimental reference expert binding supplies DeepInfer's normalization
producer with views into Flare's prepared MegaMoE workspace. DeepGEMM's router
writes canonical selections through a physical-row mapping and preserves the
separate image routing bias. The expert consumer rejects encoded tensors that
do not alias its prepared storage. Shared-first host accumulation matches the
reference's FP32 reduction order; the existing host binding retains routed-first
accumulation by default. These are preparation-time arithmetic choices, not
runtime residency policies. This binding does not establish complete-model or
native-serving qualification.
Canonical expert identities remain separate from physical storage rows. On
SM103, the binding matches the reference with all-HBM storage, disjoint HBM/Grace
tiers and HBM copies of Grace-backed experts at capacities 1, 8, 129 and 8192.
These tests use eight checkpoint experts and the shared expert, changing text
and image routing masks. Duplicate physical copies do not add router choices.

The experimental reference Engram binding publishes graph-produced hashes to
Helios's mapped request mailboxes. Its projection consumes delivered FP8 row
planes after their external CUDA readiness event and losslessly packs E8M0
scale bytes for DeepGEMM. The reference hash and gate kernels remain intact;
Torch operation declarations expose their outputs and mutations to preparation.
No BF16 lookup materialization or activation recasting is required. Inactive
tokens publish negative row IDs and contribute no Engram residual update.

On SM103, native execution of both checkpoint Engram projections and gates
matches the reference exactly at graph capacities 1, 192 and 8192 with shared
8192-token delivery buffers. Tests use checkpoint table samples and a synthetic
hash space, changing histories, image bypass and fully inactive batches. The
full checkpoint's hash geometry is checked separately against the PyTorch hash
reference. These checks do not qualify full-table delivery, fresh-process core
export or complete-model serving.

The reference Engram history binding reads raw tokens from native continuation
rows and constructs compressed-vocabulary n-grams inside captured computation.
Image positions terminate lookback, including across chunk boundaries. Commit
appends only the consumed input prefix; rejected drafts and inactive requests
leave history unchanged. On SM103, compiled graph replay matches DeepInfer's
native n-gram builder at request widths 1, 6 and 8 with four packed requests and
at single-request prefill capacities 129 and 8192. Checks cover image boundaries,
full/partial/zero acceptance and restored history, with exact state agreement,
no replay allocations and zero CUDA memory errors. Complete speculative serving
and native continuation transfer remain separate qualification requirements.

The experimental reference cache binding exposes shared compressed pages and
per-request sliding-window and compressor rings through Flare's continuation
storage. Target and draft layers retain separate windows; shared cache consumers
alias their source pages. Every history ring and partial page belongs to the
complete continuation row, including state omitted by the reference engine's
disk-cache format. Ring lengths follow the prepared verification capacity.
Attention and draft index construction consume the actual cache ring geometry.
Speculative transactions snapshot cache rows before draft execution and restore
rejected writes after verification. Main-cache records contain values and scales
together; index caches store them in separate page sections. Rollback addresses
both index sections independently. On SM103, compiled transactions using the
reference cache-store operation preserve every backing byte against accepted-prefix
stores at widths one and eight, for every accepted count, across compression and
page boundaries and inactive requests. This component check does not establish
complete speculative generation or native prefix restoration.
Sliding-window cache records occupy 528 bytes. FlashMLA's tensor map requires
the stride between request slots to contain whole records, so continuation-row
padding preserves both that divisibility and the state allocator's alignment.

All 40 target and three draft attention modules construct from checkpoint
parameters on SM103 with explicit attention and indexer preparation settings.
The first target layer's complete reference attention compiles and replays with exact output
and cache agreement at 129- and 8192-token prefill capacities and one- and
eight-token decode capacities, without GPU allocations during replay. The
reference attention chain from target layers 2 through 39 also compiles and
preserves exact active outputs and complete cache state for short decode sequences at widths
one and eight and prefill capacities of 129 and 8192. Prefill checks continue
across odd chunk boundaries, shorten chunks and replay fully inactive graphs.
These checks use checkpoint attention weights and synthetic activations, with
compressed-cache sharing, index projection and candidate-block selection at a
prepared context capacity of 32768 tokens. They do not establish complete
decoder execution, long-context model accuracy or native core export.
CUDA memory checking reports zero errors for direct and compiled attention
and metadata checks, including fully inactive graphs that preserve all cache storage.

Sparse-indexer compiler bindings retain DeepGEMM scoring and scheduling,
DeepSelect selection, and the reference candidate-index transformations.
Pointer alignment is checked inside the native scoring calls. Scheduling
outputs match the declared compiler shape, dtype and stride for contiguous
and paged FP4 metadata at 1, 8 and 129 rows, with 4, 128 and 4096 candidate
blocks. These allocation checks do not establish long-context selection quality.

The experimental metadata bindings consume Flare's validated packed requests
inside captured computation. They supply reference token positions, page tables,
C1/C2 compression destinations and sliding-window indices. Padded tokens receive
invalid state-write destinations. Decode output matches DeepInfer's native C++
input builder at verification widths 1, 6 and 8 across page and ring boundaries.
Prefill follows Flare's single-request graph contract. It uses fixed-capacity
cache gathers, zeroes unused output rows and excludes them from attention.
At capacities 129 and 8192, metadata and gathered cache values match the native
builder, while compressed outputs and state match the reference kernel with
NaNs in padded input rows. Engram history and image routing masks remain explicit
inputs; these bindings do not commit request history.

On SM103, captured reference compression matches an uninterrupted reference
prefill across a 129-token chunk followed by 8192 tokens, including restoration
at the odd chunk boundary. Verification widths 1, 6 and 8 preserve exact outputs
and ring state after full, partial and zero draft acceptance. Decode indexing
and five- and seven-token draft metadata match the reference index formulas.
Compressor kernels skip inactive rows before accessing state, using an explicit
negative state index; verification bounds come from prepared input geometry.
Captured tests preserve active outputs and all inactive state with interleaved
padded rows, empty prefill requests and fully inactive batches. CUDA memory
checking reports zero errors on those cases.

Exported full-model serving is qualified on one GB300 across both the
complete-chunk core
`63bf1d05b2011d9a4ffeda37a4f3fb9f98144733b6eb0ddd08439a477ecde79a` (`207`
compiled forwards, `237` graphs, `--speculative-graph-budget 64`) and the CED +
Encoder-staged core
`13e47a848e903037c8e30b0c8082544d9d4f3b79844429349e9b462fdab9c736` (`TP1`,
262,144-token context, concurrency 16, 8,192-token prefill chunks, adaptive
draft depths through seven, 128-token packed verification capacity,
`--speculative-graph-budget 80`, image concurrency one, `--parallax-routing`,
`"prefill_staging": {"min_rows": 1024, "min_cold_experts": 8}`, 11,680 active
target HBM experts plus 40 spare exchange rows and 3,680 Grace experts). The
CED + staged core compiles 283 forwards and exports 313 graphs (`35,409` total
CUDA graph kernel nodes, fewer than the 237-graph core's `37,300` nodes after
vectorizing `ReferenceCacheTransaction` ring snapshots/commits across the
contiguous ring slice) using a `9.13 GiB` shared CUDA graph pool; fresh-process
`flare run` verifies all shards, loads without Python, and allocates
`18,029,568` logical KV tokens and `1,568` snapshot slots (`15.01 GiB` cache
budget). Because `dspark_target_layer_ids` (`[37, 38, 39]`) lies entirely in the
consumer half (`layers 20..39`), `"reference"` enables CED consumer compaction
(`yoco_prefill_truncation="compact"`, `keep_tokens = 144` >= `window_tokens = 135`,
`terminal_prefill_min_tokens = 144`), skips consumer `layers 20..39` and LM-head
projection on intermediate `prefill` chunks (`terminal=False`), raises the
sparse-indexer scoring workspace limit to `256 MiB` (`512` query rows), and
stages cold Context Encoder experts (`layers 0..19`) two layers ahead on two
auxiliary streams. Vision encoder varlen attention zero-initializes its output
buffer before `_flash_attn_fwd` so non-128-aligned live patch counts cannot
propagate `NaN` values left in the shared graph pool.

Under the equal 11,680-expert HBM budget on core `13e47a84...`, sixteen
completed 2,048-token responses per cell with distinct request seeds deliver
`243.1`, `414.2`, `667.0`, `993.1`, and `1,478.9 tokens/s` at concurrencies 1,
2, 4, 8, and 16 with `decayed_miss_lru` active movement at adaptive `MTP-3`
(mean verification graph durations `8.91`, `10.21`, `11.79`, `14.45`, and
`18.02 ms`), compared with `166.3`, `528.7`, and `730.9 tokens/s` under static
placement at concurrencies 1, 8, and 16. Across runtime draft depths 0, 3, 5,
and 7 (`alpha = 1.0`) with active movement, concurrency-one decode delivers
`146.6`, `243.1`, `246.9`, and `241.3 tokens/s` (`249.6 tokens/s` at `MTP-5`
with `--mtp-adaptive-verification-alpha 0.5`), and concurrency-sixteen decode
delivers `1,016.7`, `1,478.9`, `1,496.1`, and `1,479.0 tokens/s`. Median
uncached prefill throughput (`N = 10` timed samples per length) at 8,192,
32,768, and 131,072 tokens reaches `58.6K`, `72.4K`, and `79.3K tokens/s`
(`139.7`, `452.7`, and `1,653.0 ms` TTFT) with active `decayed_miss_lru`
(`41.2K`, `52.1K`, and `55.8K tokens/s` / `198.6`, `629.5`, and `2,351.0 ms`
static), surpassing both the `"hybrid"` equal-residency core
`9078905c99a9473fe553c8ede38dc7e1ef8fd71271d0ec425aed2cacd7f93b50` (`150.1`,
`575.1`, and `2,399.7 ms` active) and local lossless DerpSeek exact 1.9 (`175.2`,
`512.5`, and `2,024.0 ms`). Prompt-heavy mixed waves (`agentic-c8` and held-out
`public-holdout/agentic-c8`) rise from `493.6` and `354.7 tokens/s` (complete-chunk
core `63bf1d05...`) to `574.6` (`+16.4%`) and `463.6 tokens/s` (`+30.7%`).
Unseeded multi-stream decode benchmarks that omit `seed` share
`defaults.seed = 0`; because the reference kernels are batch-invariant across
lanes, identical unseeded prompts follow identical token trajectories when using
fixed widths (`2,590.5–3,012.1 tokens/s` sustained `C16` under adaptive MTP), so
multi-stream MoE comparisons require distinct per-request seeds. Multi-GPU TP2+
composition and prompt log-probabilities remain unimplemented on `"reference"`.

Carry forward measured techniques from
[DerpSeek](https://github.com/chronarion/derpseek) where they fit the reference
forward: expert staging, transfer/compute overlap, SM reservation, direct
producer-to-consumer encoding and asynchronous readiness. Compare each change
against the adapted reference with identical expert capacity and workload.
Retain the existing race protections and rollback contracts; a reference's
different allocator or scheduler does not establish those properties for Flare.
Document external code and inspiration under the repository's attribution rules.

Preparation settings must be explicit recipe or configuration values. Importing
a reference implementation does not authorize environment-selected model behavior,
its serving scheduler, Python callbacks, runtime compilation or recapture.
Complete all kernel specialization and allocation before capture. Replace
distributed setup with explicit local geometry and buffers for single-device
preparation. Account for every resource that the exported computation retains.

Keep each native launcher's template signatures and workspace layouts paired
with its matching headers. A private kernel fork must use its own compilation
runtime rather than replacing the installed library's global include root.
Check accumulation order as well as input encoding when comparing forks:
identical expert contributions can round differently when the shared expert
is summed before or after routed experts.

Reference launchers that read device pointers need an opaque Torch operation
with explicit reads, writes and output metadata before complete-forward
compilation. Keep the underlying kernel unchanged when adding that boundary.
Pass producer outputs as explicit consumer inputs when native tensor views
have separate Torch storage objects over the same CUDA allocation; the compiler
cannot infer those dependencies from addresses. Warm every capture stream
that owns library synchronization storage, and retain that storage through
export and native replay.

Qualify ordinary generation and the complete speculative operation with changed
inputs, inactive rows, page and window boundaries, prefix restore, cancellation
and repeated graph transitions. Compare persistent state as well as outputs.
Then measure complete decode rounds and exported serving at concurrency one,
two and four, with prefill chunks up to 8,192 tokens and equal residency budgets.
Report source-reference parity, Flare integration correctness and comparative
serving performance separately. Preserving the reference's operator sequence
does not itself establish its arithmetic or performance on SM103.

## Adapt the boundaries that need changing

Keep model composition and checkpoint interpretation in the recipe. Put reusable
execution mechanisms in `ignition/components/`. Helios continues to replay
prepared graphs and manage declared storage; it must not implement model-specific
computation or invoke Python. Do not add a model identity branch to native policy
to compensate for a missing preparation contract.

The usual adaptation points are:

- **Weight loading.** Perform lossless repacking, permutation, and scale-layout
  conversion once during preparation. Move values and their associated scales
  together. Verify inverse reconstruction against the checkpoint, including TP
  partitions and tails. Do not requantize weights to fit an operator. Distinguish
  compressed storage from the representation the kernel consumes, and budget
  temporary plus final representations during loading.
- **Activation interfaces.** Let the producer emit the layout and precision the
  consumer accepts where semantics permit. Track every cast, quantization,
  transpose, concatenation, and contiguous copy across the seam. Removing one
  requires proving that its rounding, permutation, or masking effect is already
  supplied elsewhere.
- **Attention and state storage.** Match writers, readers, gather operations,
  and snapshot/restore together. Record byte order, page stride, alignment,
  scales, logical-to-physical indices, and padding. Equal tensor shapes or byte
  counts do not establish layout compatibility. Check both full pages and partial
  windows, including speculative capacity beyond the ordinary decode window.
  Check singleton slices of broadcast tables: PyTorch can classify them as
  contiguous while retaining a zero row stride. Kernels that use that stride
  as a physical width or address bound need an explicitly materialized row.
- **Composition and fusion.** Identify exactly which normalization, rotation,
  gating, shared-expert, reduction, or output-quantization work the kernel includes.
  Remove duplicate work only when the replacement preserves the complete
  operation. Keep routing and all selected experts intact.
- **Workspace and dependencies.** Expose temporary tensors, reads, writes,
  aliases, and auxiliary-stream dependencies to compilation. Keep cache history,
  counters that persist between launches, and other cross-graph state in explicit
  storage. Determine whether scratch must be initialized once or written on every
  replay; zeroing everything can conceal a dependency error or add avoidable work.

Prefer a shared component change when other recipes have the same requirement.
Inspect its consumers before choosing the boundary, and report untested recipes
separately. A mechanism that works for DeepSeek's sparse MLA does not establish
support for GLM's attention or Qwen's recurrent state.

## Make preparation own the implementation

Declare operators and the full shape catalog before tuning. Use the existing
preparation session for compilation, candidate selection, workspace sizing, and
binding finalizers. Constructors declare work; compilation and materialization
finish before capture. The [native operator helper](../ignition/components/native_operator.py)
and [SM103 projections](../ignition/components/sm103_projection.py) illustrate
existing preparation integration points; choose one only when its contract fits.

Custom operators must describe mutations and aliases accurately and expose their
workspaces as arguments. Compile complete forwards after operator preparation,
warm every selected variant, and capture through the shared preparation machinery.
Retain libraries, compiled programs, tensors, and graph resources until native
work drains. Exported serving must need no Python callback, runtime compilation,
graph recapture, or steady-state allocation.

Exercise the compiler-worker handoff used by export. Worker payloads carry tensor
layouts and selected-plan metadata; preparation-built native bindings remain in
the owning rank. In-process component compilation does not test this serialization
boundary. Check worker compilation and captured replay of a composed producer and
consumer before attempting full-model export.

Introducing an external implementation also requires its license and attribution,
a reproducible preparation build, and tests against its reference behavior.
Qualify the implementation before rewriting it. This does not authorize new
captured-graph operation families; those still require the approval specified in
[repository guidance](../AGENTS.md#architecture-and-ownership).

## Prove the path is clean

Inspect a complete captured forward, then confirm the corresponding native replay.
Check emitted kernel names and launch counts alongside the selected plans; include
copies, fills, casts, scale expansion, gathers, reductions, and synchronization.
Use dataflow to explain each launch. Small copies or conversion kernels repeated
per layer can dominate low-batch decode even when individual GEMMs are fast.

Look for these integration defects:

- A shape, stride, dtype, or alignment mismatch selects an unintended implementation.
- Repacking or transposing weights occurs inside the forward instead of loading.
- A fused operator is followed by normalization, RoPE, shared-expert computation,
  or quantization that it already performs.
- A producer writes one layout, an adapter converts it, and the consumer converts
  it back. Prefer a compatible producer/consumer interface when arithmetic allows.
- Padding expands actual work unnecessarily, or an inactive lane still mutates state.
- Opaque allocations or implicit dependencies prevent scratch reuse or allow early reuse.
- Stream joins serialize independent work, or overlap races on shared scratch.
- A warmed preparation process works, but export omits initialized storage or
  a resource owner needed by a fresh native process.

A conversion or synchronization is not automatically waste. Keep it when required
by precision, state, or lifetime contracts. Measure the complete producer,
conversion, kernel, and consumer sequence before deciding to remove or fuse it.
Compare decode and prefill separately; an integration may improve one and hurt the
other. Use complete graph duration to assess overlap, not a sum of isolated kernel
speedups.

Use the following procedure to establish which path actually runs:

1. **Exercise dispatch boundaries.** Derive cases from the implementation's
   selection conditions, including row counts immediately below, at, and above
   each threshold. Cover supported decode, prefill, and speculative shapes,
   partial pages, and relevant strides. Record the selected preparation plan for
   each case. A batch-one result cannot establish the path used by verification.
2. **Inspect the emitted graph.** Examine the captured graph's nodes and edges
   after complete-forward compilation. Associate each kernel, copy, fill, and
   dependency with the operation it implements. Check that loading-time weight
   transforms stay outside replay and fused work appears only once. A graph dump
   establishes structure; it does not measure latency or prove safe overlap.
3. **Trace native execution.** Collect a GPU kernel timeline from the exported
   core, with layer and operator attribution, using the
   [performance measurement guidance](performance-work.md#measurement). Compare
   actual launches with the captured structure and selected plans. Include
   transfers and synchronization outside the graph; a host graph-launch event
   alone cannot identify the executed kernels. Measure unprofiled serving
   separately so profiler overhead does not become the performance result.
4. **Explain remaining cost.** Attribute the complete graph interval to useful
   computation, required conversions, memory traffic, and dependencies. Investigate
   unexplained launches, idle intervals, repeated transformations, and differences
   from the reference runtime. Recheck affected graph variants after changing a
   producer, consumer, or layout, even when the central kernel is unchanged.

Call a path clean only for the exercised configurations: dispatch is understood,
surrounding work has a contract-based reason, fresh native replay preserves
outputs and state without allocations, and measured execution accounts for the
claimed benefit. Few launches do not prove correctness or optimal performance.
Keep structural checks, numerical checks, and timing results distinct.

For example, a fused attention operator may return FP8 activations with packed
scales for a grouped output projection. Check the producer's output coordinates
against the projection's weight columns and scale groups. If they differ only
by a fixed permutation, permute the weights and their scales during loading and
verify exact inverse reconstruction. Then test the composed attention and
projection against the reference, including masked rows and dispatch boundaries.
Inspect the compiled graph to confirm that no activation unpack/repack was
introduced between them. Preserve any BF16 intermediate rounding required by
the reference before the following dense projection. Direct FP8 consumption
alone does not establish numerical equivalence or lower complete-forward time.

When comparing activation encoders, distinguish logical scale entries from
padding that the consumer does not read. Different finite scales for an all-zero
group can represent the same values. Check the reconstructed activations and
the actual consuming projection before accepting such a difference. Keep exact
scale checks for nonzero groups. A fused preparation stage must also produce
every intermediate used by another branch; a quantized query cannot silently
replace a BF16 query consumed by an indexer.

Document the resulting execution path in the model's preparation guidance so
another developer can distinguish an integrated path from an available component.
Identify the recipe entry point, shape-dependent selection, resident weight and
cache layouts, and tests that exercise the composed forward. State which paths
are qualified for prefill, decode, verification, and draft execution. A remaining
reference-runtime fusion or precision difference belongs in the comparison's
limitations; unresolved integration work belongs in [TODO.txt](TODO.txt).

## Qualify in increasing scope

1. **Storage and arithmetic.** Verify packed values, scales, permutations, and
   reference outputs. Cover tails, noncontiguous views, page boundaries, and
   invalid-contract rejection. Derive numerical bounds from required arithmetic
   and measured repeatability; do not widen them to accommodate a failing kernel.
2. **Captured component execution.** Change inputs, routes, valid lengths, and
   masks between replays. Poison disposable outputs and scratch where their
   contract requires complete writes. Check inactive rows, allocation stability,
   and resource lifetimes. Compare exact bytes for lossless transformations.
3. **Compiled forward and graph transitions.** Validate producer/consumer
   composition, ordinary and speculative capacities, prefill-to-decode, partial
   pages, accepted/rejected drafts, and cache reuse/restore. A passing isolated
   attention or expert kernel does not prove these transitions.
4. **Fresh-process native replay.** Export and load through `flare run CORE`.
   Check exported storage, pointer relocation, graph dependencies, initialized
   state, and shutdown/drain behavior. Preparation-process replay alone cannot
   establish this boundary.
5. **Serving comparison.** Match checkpoint precision, workload, concurrency,
   context, speculation, memory placement, sampling, and completed work. Measure
   throughput and latency alongside complete GPU rounds, transfer volume, and
   scheduler overhead. Preserve regressions as well as improvements.

Keep reusable checks in ordinary tests and reproduction instructions in the
model preparation guide. Keep unresolved work in [TODO.txt](TODO.txt). Retain raw
measurements and diagnostic logs outside the repository; do not substitute a run
journal for tests or the documented integration contract. State qualification
by the configurations actually exercised, and carry unrun gates forward explicitly.
