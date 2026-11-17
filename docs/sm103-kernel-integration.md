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

When comparing another runtime, inspect its actual enabled path. Determine
which operations it fuses, which work it overlaps, what precision it uses, and
what storage it reads. A kernel with the same name may have different arguments,
layouts, or surrounding work. Account for those differences before attributing
a performance gap to scheduling or residency.

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
