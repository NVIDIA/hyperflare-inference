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

### DeepSeek V4.1 `"reference"` preparation boundaries

The `"execution": "reference"` path in `ignition/recipes/deepseek41/` binds the
DeepInfer target, DFlash draft, and vision modules into Flare's preparation and
export contracts with the following architectural invariants:

- **Parameter loading and consuming fusion (`reference_parameters.py`,
  `reference_model.py`).** Ordinary text, draft, and vision parameters
  materialize once with tied storage preserved while routed experts and Engram
  tables stay as explicit storage declarations. Reference projection fusion
  transfers storage ownership, so a failed assembly requires a fresh parameter
  tree.
- **Expert residency and accumulation order (`reference_experts.py`,
  `megamoe_kernel/`).** Canonical expert identities remain separate from
  physical HBM and Grace row indices; duplicate physical copies never expand
  router choices. Normalization producers write into prepared MegaMoE workspace
  views, and expert consumers reject encoded tensors that do not alias prepared
  storage. Shared-first FP32 accumulation matches DeepInfer's reduction order;
  routed-first remains the default for the standalone host binding.
- **Cache storage and speculative rollback (`reference_cache.py`,
  `reference_metadata.py`).** Shared compressed pages (`288`-byte records),
  index pages (`68`-byte records with separated payload and scale sections),
  and per-request sliding-window (`528`-byte records) and compressor rings live
  in Flare's continuation storage with alignment preserving both allocator
  granularity and FlashMLA's tensor-map row stride (`lcm(256, 528)`). Padded
  tokens receive invalid write destinations and leave cache state unchanged.
  Speculative transactions snapshot and restore both main-cache and separated
  index-cache regions independently across accepted prefix lengths.
- **Engram delivery and history (`reference_engram.py`).** Captured graphs
  publish token hashes to Helios's mapped request mailboxes and consume
  delivered FP8 rows after external CUDA readiness events, with negative row
  IDs bypassing inactive tokens. Compressed-vocabulary n-gram history lives in
  continuation rows, terminates lookback at image sentinels, and commits only
  accepted prefixes.
- **Vision and multimodal prefill (`reference_vision.py`).** The 32-block
  vision encoder, spatial downsample, and projector bind into Flare's vision
  graph family (`PATCH_CAPACITIES`) and image feature arena. Varlen attention
  zero-initializes its output buffer before `_flash_attn_fwd` so non-128-aligned
  patch counts cannot propagate `NaN` values from the shared CUDA graph pool.
- **Context-Encoder-Decoder (CED) prefill and Encoder staging (`bootstrap.py`,
  `megamoe_staging.py`).** Because `dspark_target_layer_ids` (`[37, 38, 39]`)
  lies entirely inside the consumer half (`layers 20..39`), CED prefill
  compacts the consumer suffix to `yoco_keep_tokens = 144` (`>= 135` window
  tokens) and omits consumer layers on intermediate chunks (`terminal=False`),
  while staging cold Context Encoder experts (`layers 0..19`) two layers ahead
  on auxiliary streams.
- **Scope and open work.** Present single-GPU GB300 (`SM103`, `TP1`) serving
  workflows and commands live in [model preparation](model-preparation.md) and
  [Parallax](parallax.md). Multi-GPU (`TP2+`) composition, prompt
  log-probabilities, pinned external preparation packaging for DeepInfer, and
  remaining long-context and concurrency qualification items are tracked in
  [TODO.txt](TODO.txt).

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
