# Deriving a serving contract from model code

A serving contract records the facts the engine needs to bind and schedule
prepared computation: inputs, state, capacities, dependencies, progress, and
results. Weight values are not required to draft these facts. Model source and
checkpoint configuration supply the initial evidence; compilation, capture, and
native replay establish whether the implementation satisfies the declaration.

The [iterative schema proposal](proposals/iterative_contract.proto) and
[SDXL-shaped example](proposals/sdxl_conditioned_example.textproto) are designed,
not implemented serving support. The example names hypothetical captured graphs.
Its four iterations are a short teaching sequence, not a recommended sampler
preset. It accepts prepared conditioning and noise tensors; it does not implement
prompt tokenization, text encoding, or checkpoint loading.

## Read the execution path before the layers

Start at the public pipeline entry point and find the generation loop. Follow
the calls that produce its inputs, the denoiser call inside the loop, the solver
call after it, and the decoder after the loop. Then inspect called modules and
configuration wherever an argument's shape or lifetime remains unclear.

For SDXL, use the [pipeline source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py).
Useful search terms are `encode_prompt`, `prepare_latents`,
`_get_add_time_ids`, `scale_model_input`, `self.unet`, `scheduler.step`, and
`vae.decode`. These identify conditioning, sample initialization, the repeated
update, and decoding. Treat Python callbacks as separate API behavior; they
cannot become production Python callbacks in Helios.

Draw the dataflow before drafting fields. This is conceptual pseudocode, not an
executable implementation or a replacement for the upstream arithmetic:

```text
conditioning = produce_conditioning(request)
sample = initialize_sample(request, prepared_constants)
for each prepared iteration:
    sample, history = complete_update(sample, conditioning, history, constants)
result = produce_result(sample)
```

This exposes a choice: capturing only the denoiser leaves guidance and solver
updates outside that graph. For the proposed contract, capture the complete
update together so the Rust engine does not need a solver implementation.

## Classify each value by its lifetime

For every tensor or mutable scalar, record its producer, consumers, shape, dtype,
and last use. A local variable name does not establish its lifetime. Inspect
returned values, aliases, in-place writes, and persistent `self` attributes in
every called component.

| Value class | How to recognize it | Contract consequence |
|---|---|---|
| Prepared constant | Independent of the admitted request; retained weights or a resolved fixed schedule | Stored compiled resource or immutable buffer; no request upload required |
| Request input | Supplied or derived for this request | Typed input, or output of a separately specified native frontend/prepared encoder |
| Persistent request state | Produced by one captured operation and read by another | Declared storage view and read/write access |
| Temporary | Produced and consumed entirely within one captured operation | Compiler-visible local storage; no shared semantic field solely for this intermediate |
| Delivered result | Read outside captured computation after completion | Typed result view and ownership through transfer |
| Control value | Used to validate identity, iteration, or submission | Fixed shared-engine source and completion checks |

At the proposed complete-update boundary, the evolving sample is persistent.
Conditioning reused by later operations must remain available. An intermediate
denoiser prediction can remain temporary if guidance and the solver consume it
inside the same captured update.

Inspect the solver rather than assuming it has no history. In
[Wan's UniPC source](https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/fm_solvers_unipc.py),
search `model_outputs`, `timestep_list`, `last_sample`, and `lower_order_nums`.
These identify values used across updates. A different solver can require a
different set of persistent tensors without requiring a model-specific Rust
algorithm. Random-stream counters also belong to persistent state when later
updates draw randomness.

## Derive shapes from configuration and transformations

Configuration gives dimensions; source code explains how those dimensions
combine. Do not infer a shape from the checkpoint's marketing name. Trace
concatenation axes, reshapes, padding, masks, and layout transforms explicitly.

For the SDXL base checkpoint:

| Evidence | Derived fact for a 1024 x 1024 single-image request |
|---|---|
| [U-Net configuration](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/unet/config.json): four input channels, conditioning width 2048 | Spatial sample has four channels; token conditioning width is 2048 |
| [VAE configuration](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/vae/config.json): four latent channels and four encoder block levels, interpreted by pipeline compression | Spatial compression is eight; sample shape is `[4,128,128]` per request |
| [Both tokenizers](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer/tokenizer_config.json) ([second configuration](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer_2/tokenizer_config.json)) set length 77; the [second encoder](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/text_encoder_2/config.json) has projection width 1280 | With pipeline padding, prepared conditioning can contain sequence `[77,2048]` and pooled vector `[1280]` |
| Pipeline's size/crop construction combines original dimensions, crop coordinates, and target dimensions | Six size/crop values per request |

The example chooses FP16 conditioning/sample storage and an FP32 decoded result.
Those are explicit proposal choices, not dtype guarantees inferred from shape.
Actual preparation must check checkpoint precision, casts, accumulation, and
decoder requirements. The VAE configuration's `force_upcast` setting is one
reason decoder precision needs separate inspection.

The [U-Net forward interface](https://huggingface.co/docs/diffusers/api/models/unet2d-cond)
separates sample, timestep, encoder features, and additional conditioning.
Read the caller too: it determines which optional inputs are actually used.
Those forward arguments are internal model inputs. The proposed serving boundary
accepts admission tensors and invokes complete updates, so forward arguments do
not map one-for-one to serving fields.

For classifier-free guidance, SDXL expands internal denoiser work to two lanes
per external image request, then combines the predictions. External batch one
still produces one result. The proposal's batch capacity therefore counts
requests, while captured graph buffers account for the internal work. This is
a dataflow inference from the pipeline, not a native guidance policy.

## Convert the evidence into schema declarations

Declare the smallest set of facts the engine consumes. For each proposed field,
identify its native consumer and validation rule. If captured computation alone
uses a numerical value, retain it in prepared storage rather than adding a Rust
algorithm selector.

| Observed requirement | Declaration in the proposal |
|---|---|
| Fixed input names, dtypes, and dimensions | `TensorProfile.inputs` / `TensorInput` |
| Distinct operations with a complete numerical update | `IterativeProgram.operation` and physical graph reference |
| Several batch capacities for the same external geometry | `TensorProfile.variants` |
| Fixed resolved update sequence, possibly reusing a program | `BatchVariant.iterations` |
| Sample, conditioning, solver history, or RNG state crossing operations | `IterativeProgram.state`, using existing `StateAccess` and `StateView` |
| Mapping host tensors and native controls to device inputs | `TensorBinding` with exactly one source |
| Request identity and committed progress | Contract generation/iteration views and `IterativeCompletion` |
| One or several fixed tensor results | `TensorProfile.results` / `TensorResult` |

For example, sample storage `[4,128,128]` at two bytes per element needs
131072 bytes per request. A separate slot-row buffer can declare a `StateView`
with offset zero and stride/extent 131072. The complete-update program declares
read/write access. Finalization reads that view and writes a result view. These
are storage facts; no field instructs Rust how to denoise the sample.

The example has seven admission tensors: initial noise, positive/negative
sequence features, positive/negative pooled features, shared size/crop values,
and guidance strength. Their exact dimensions live in the profile. Choosing to
capture text encoders would change the external tensors to encoder inputs;
it would not require a `number_of_text_encoders` scheduling flag.

The example assumes identical size/crop geometry for the positive and negative
branches; the full upstream API permits different negative geometry. It also
assumes a deterministic, history-free update after admission. A multistep or
stochastic solver would require additional state views. `initial_noise` means
supplied unit Gaussian noise, scaled by initialization using the scheduler's
prepared `init_noise_sigma`. This scale need not equal the first schedule sigma:
Euler scheduling with leading timestep spacing uses `sqrt(max(sigmas)^2 + 1)`.
`decoded_image` means the raw, channel-first VAE output before image
postprocessing, clamping, or watermarking. Its shape and dtype alone do not define
pixel value range or a media encoding.

The profile's fixed four-iteration sequence references the same complete-update
program four times. Immutable per-iteration coefficients are captured resources.
The committed cursor selects the appropriate prepared row inside computation.
No host solver formulas or arbitrary graph branch language are introduced.

## Review semantics before loading weights

Check these properties against the proposed declarations and actual code:

1. Every value crossing an operation boundary has an owner and retained storage.
2. Every state read has a producer before its first use, including after export
   zeros disposable preparation storage.
3. Tensor shape, dtype, row stride, and byte extent agree. Shared profile buffers
   cannot make concurrent request rows overlap.
4. Initialization establishes identity and cursor zero. A complete iteration
   validates the expected cursor and commits exactly one advance. Finalization
   requires the terminal cursor.
5. Inactive rows and rejected requests do not mutate persistent state. Failures
   prevent publication and retain resources until execution drains.
6. Result storage remains owned until GPU execution and transfer both finish.
7. Each declared source has a bounded native packing rule. Names and enums do
   not select per-model numerical or scheduling policy.

The proposed packing rules make the cohort explicit: active submission rows are
the prefix below `LIVE_ROWS`, with distinct in-range slot IDs; inactive tail slot
IDs are -1. Slot, generation, and expected-iteration controls use submission-row
indexing. Live count, frame index, and submission sequence are scalar controls.
Tensor inputs use a batch-leading or request-capacity-leading registered buffer
whose contiguous inner shape matches the declared input.

Initialization reports submitted and committed cursors zero and requires exclusive
ownership of a drained slot. Completion frames remain owned through terminal rank
agreement and host transfer/consumption. Native code observes only the submitted
live rows and verifies identity before accepting their records. Capacities fit
the int32 controls, and identity/cursor values fit nonnegative int64 without
increment overflow. These are proposed lifecycle requirements, not runtime
guarantees established by compiling the schema.

The proposal uses final, fixed-shape tensor results. Dynamic output extents,
media encoding, and token-canvas streaming need their own demonstrated contracts.

## What model loading and execution prove

| Stage | Evidence it can establish |
|---|---|
| Source and configuration inspection | Intended signatures, dimensions, branches, numerical conventions, and candidate state lifetimes |
| Meta/fake tensor inspection where supported | Additional shape and dtype checks without weight values; data-dependent behavior can remain unresolved |
| Real checkpoint loading | Weight mapping, actual parameter layouts and precision, and concrete tensor bindings |
| Compilation and capture | Whether the prepared computation can be compiled and captured with explicit storage and dependencies |
| Export and Rust replay | Whether recorded resources execute independently of Python, including with disposable scratch zeroed |
| Reference and lifecycle tests | Numerical agreement, complete state, seed behavior, cancellation, inactive lanes, result transfer, and allocation stability |

Loading weights does not by itself prove capture or native replay correctness.
Conversely, schema compilation and an example's Protobuf round trip establish
syntax and representation, not an executable model. Preserve that distinction
when reporting progress or claiming model support.
