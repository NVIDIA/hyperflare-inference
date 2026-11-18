# Prepared diffusion execution

Flare prepares model computation in Ignition and serves captured CUDA graphs in
Helios. Diffusion requires explicit iterative state and tensor results alongside
the existing token-generation contract. Denoiser architecture, conditioning,
guidance, numerical integration, and codec transforms remain recipe computation.

The geometry and schedule helpers and the fixed SDXL and PixArt tensor
preparation paths described below are implemented. The concurrent serving
contract in this document is **designed, not implemented**.
The [isolated Protobuf proposal](proposals/iterative_contract.proto) describes
fixed tensor inputs, iteration programs, persistent state, and results. It is
outside the production schema imports and build, and its field numbers belong
only to the proposal. The tensor command executes one explicitly described
request; it does not implement the proposed scheduler or a diffusion HTTP API.

[Deriving contracts from model code](diffusion-contract-walkthrough.md) explains
how to identify those declarations from upstream execution without loading weights.

## Model requirements

These representative published model families establish different storage and
execution requirements. The table describes upstream implementations, not Flare
model support or a ranking of model quality.

| Family | Computation and conditioning | Requirement on prepared execution |
|---|---|---|
| [SDXL base](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py) | U-Net; two CLIP encoders, pooled text features, and size/crop conditioning. The [published VAE](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/vae/config.json) has four latent channels and spatial compression by eight. | Several conditioning tensors; spatial denoiser input; classifier-free guidance capacity. |
| [SD3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | Multimodal diffusion transformer with two CLIP encoders and T5; flow matching. | Token and pooled conditioning; model-specific prediction and timestep interpretation inside the prepared forward. |
| [FLUX.1](https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py) and [FLUX.2](https://huggingface.co/docs/diffusers/main/api/pipelines/flux2) | FLUX.1 uses T5 sequence features, CLIP pooled features, and packed latent patches. FLUX.2 uses Mistral3 conditioning. Guidance embeddings and external classifier-free guidance are different computations. | Concrete encoder outputs and positional tensors; no fixed encoder count or universal guidance scalar interpretation. |
| [Qwen-Image](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py) and [Qwen-Image 2.1](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21) | Masked multimodal conditioning; the original pipeline uses per-channel latent normalization and norm-corrected guidance. Version 2.1 uses block-causal attention and persistent condition-prefix KV state. | Explicit masks, normalization in captured computation, and persistent conditioning storage, including KV where used. |
| [Z-Image-Turbo](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py) | Qwen3 conditioning; reversed normalized model time and negated prediction before the flow update; FP32 latent updates with lower-precision denoiser input. | Independent storage and compute dtypes; no native universal prediction conversion or time formula. |
| [Wan2.2](https://github.com/Wan-Video/Wan2.2) | Causal video VAE; temporal/spatial compression differs between TI2V-5B and A14B. A14B uses separate high-noise and low-noise denoisers. | Causal temporal geometry, conditioning masks, and concrete prepared variants for stages with different computation. |
| [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/modules/models.py) | Video transformer with three-dimensional positions, token and pooled conditioning, and embedded guidance. | Several independent conditioning tensors and explicit temporal extents. |
| [LTX-2](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/README.md) | Coupled audio/video denoising with separate modality tensors, coordinates, decoders, and a vocoder. | Multiple evolving state tensors and multiple typed results with independent shapes and time bases. |
| [LLaDA](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) and [Dream](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/blob/main/generation_utils.py) | Masked token canvases; selected positions are revealed out of text order. Their default samplers preserve already revealed tokens. | Position-addressed working state; reveal order is distinct from a streamable contiguous text prefix. Continuous sigma schedules are not a requirement for discrete diffusion. |

The [multistep solver interface](https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver)
also permits prediction conventions and stochastic variants with different
state. [Wan's UniPC implementation](https://github.com/Wan-Video/Wan2.2/blob/main/wan/utils/fm_solvers_unipc.py)
retains earlier predictions, timesteps, a previous sample, and order counters.
The evolving latent and an initial seed alone do not describe complete
continuation state.

## Preparation modules

[`LatentGeometry`](../ignition/components/diffusion_geometry.py) computes exact
compression and patch extents for one sample. Axes are frames, height, width;
latent shapes include their channel dimension. Each axis obeys:

```text
output_extent = (latent_extent - 1) * stride + offset
```

Dense spatial compression uses an offset equal to its stride. Causal temporal
compression uses offset one: 81 output frames at temporal stride four require
21 latent frames. Images have frame stride and offset one. Patch alignment is a
separate validation after compression. The helper rejects unsupported extents
instead of rounding, cropping, or padding. Packed shapes count tokens and
elements per token; the recipe owns tensor permutation and within-patch order.

Examples grounded in the [SDXL VAE](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/vae/config.json),
[FLUX sampler](https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py),
and [Wan configurations](https://github.com/Wan-Video/Wan2.2/tree/main/wan/configs):

| Geometry | Requested extent | Latent shape | Packed shape |
|---|---|---|---|
| Four channels, spatial stride 8, patch 1 | 1 x 1024 x 1024 | 4 x 1 x 128 x 128 | 16384 x 4 |
| Sixteen channels, spatial stride 8, spatial patch 2 | 1 x 768 x 1024 | 16 x 1 x 96 x 128 | 3072 x 64 |
| Sixteen channels, causal time stride 4, spatial stride 8, spatial patch 2 | 81 x 480 x 832 | 16 x 21 x 60 x 104 | 32760 x 64 |
| Forty-eight channels, causal time stride 4, spatial stride 16, spatial patch 2 | 121 x 704 x 1280 | 48 x 31 x 44 x 80 | 27280 x 192 |

[`PreparedNoiseSchedule`](../ignition/preparation/noise_schedule.py) snapshots
already resolved model timesteps and adjacent sigma values for sigma-based
solvers. Recipes construct schedules using their authoritative implementation
during preparation. The helper preserves host scalar types and values, including
repeated timesteps, increasing intervals, and nonzero terminal noise. It performs
no prediction conversion or solver update. The bound tensor declarations and
recipe materialization determine GPU dtype and precision.

These helpers perform preparation metadata calculations. They do not register
recipes, capture graphs, extend `GraphSignature`, or export a serving contract.

## Fixed SDXL tensor execution

[`ignition.recipes.sdxl.prepare`](../ignition/recipes/sdxl/prepare.py) prepares
SDXL Base 1.0 revision `462165984030d82259a11f4367a4eed129e94a7b`, using the
checkpoint's FP16 weights and one 1024-square image. Its `--steps` argument
accepts a positive integer and defaults to four Euler updates. Preparation fixes
the schedule, creates one graph and persistent latent output for each update,
and binds the decoder to the final latent. The step count is recorded in the
compilation cache identity, reference metadata, and exported profile; changing
it requires preparing another core. It is not a runtime iteration override.
The VAE decoder uses the checkpoint's required FP32 upcast. Four steps are a
short numerical qualification workload, not an image-quality preset. The entry
point accepts an existing checkpoint snapshot; it is separate from the
token-model recipe registry.

The request supplies both tokenizers' token IDs, initial noise, guidance, and
size/crop conditioning. Prepared forwards execute the canonical text encoders,
the U-Net with classifier-free guidance and Euler arithmetic, and the VAE.
Initialization, each updated latent, text conditioning and the raw decoded
image have explicit persistent output storage. Heliopause exports the graph
resources and weights as a signed core. The preparation process exits before
[`flare run --tensor-request`](cli-design.md#fixed-tensor-execution) loads and
executes the core in Rust. Rust uploads and downloads declared byte spans and
launches the ordered graphs; it contains no denoising or solver arithmetic.

The preparation command also runs the canonical Diffusers pipeline for two
different prompts and noise tensors. It checks the composed eager forwards
against those observations before compilation. The
[`sdxl_compare`](../qualify/sdxl_compare.py) command compares the saved reference
with independently loaded native outputs, including repeated executions.
Saved outputs from captured graphs distinguish compiler arithmetic differences
from export or native replay differences.
Default numerical acceptance is exact equality; any nonzero tolerance must be
explicit. Reports retain per-tensor errors and raw timing samples.

On NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), the four-step and
30-step profiles have been checked with the recipe's two prompts and noise
seeds 42 and 43. Independently loaded native executions replay all seven or
33 graphs respectively. Their eight or 34 saved outputs and final repetitions
match the captured compiled outputs byte for byte. The native processes load
neither Python nor PyTorch. The composed eager forwards match the canonical
Diffusers observations exactly.
The compiled text conditioning, denoising updates, and decoded images differ
from the canonical reference; initialization remains exact. Replaying the
first update and decoder with canonical intermediate inputs also produces
differences in both profiles. Export and native replay are verified for these
cases, while numerical equivalence to canonical Diffusers remains unqualified.
Exact reference comparisons fail and performance ratios are withheld.

Timing begins with token IDs and noise already on the GPU and ends with the raw
decoded tensor. It excludes loading, tokenization, transfers and image-file
encoding. The eager pipeline performs its canonical VAE precision conversions
per inference; the prepared core retains the converted weights. A timing ratio
therefore compares whole execution paths and does not isolate graph-launch
overhead or an individual kernel optimization.

The request sidecar declares physical tensor views and graph order. It is not
the proposed signed semantic serving contract and does not provide concurrent
admission, cancellation, continuation, native prompt tokenization or an HTTP
image endpoint. Other shapes, solvers and diffusion architectures are outside
this preparation path; each step count requires separate numerical
qualification. Outstanding qualification is tracked in
[`TODO.txt`](TODO.txt).

## Fixed PixArt tensor execution

[`ignition.recipes.pixart.prepare`](../ignition/recipes/pixart/prepare.py) prepares
`PixArt-alpha/PixArt-XL-2-512x512` revision
`50f702106901db6d0f8b67eb88e814c56ded2692` for one 512-square image. The fixed
profile uses 120 text positions, classifier-free guidance 4.5, and the
checkpoint's second-order midpoint DPM-Solver++ schedule with epsilon
prediction. The default schedule contains 20 updates; changing `--steps`
requires preparing another core. The recipe preserves canonical caption
cleaning and resolution binning. Caption cleaning and tokenization precede
the tensor request and its inference timing.

The request supplies positive and negative token IDs and attention masks,
initial noise, and guidance. Prepared forwards execute T5 text encoding, the
diffusion transformer, classifier-free guidance, DPM-Solver++ updates, and the
VAE decoder. T5 retains its canonical mixture of FP16 and FP32 parameters;
the transformer and decoder use FP16. The
[`PreparedDPMStep`](../ignition/recipes/pixart/scheduler.py) preparation helper
resolves scalar coefficients and captures solver arithmetic into each update.
Previous converted predictions occupy explicit persistent buffers. The first
and final updates use the canonical first-order formula; intervening updates
consume the previous prediction for the second-order formula.

The 20-step profile contains 23 graphs and exposes 43 intermediate and final
outputs: text conditioning, initialized latent, each updated latent, each
converted solver prediction, and the decoded image. It uses the same signed
physical core format, tensor request, and native graph executor as SDXL.
Neither Helios nor the Flare CLI contains PixArt-specific computation or
dispatch. The preparation entry point is separate from the token-model recipe
registry, and the concurrent serving limitations of the SDXL tensor command
also apply here.

On NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120), both fixed prompts and
seeds 42 and 43 pass all 43 exact comparisons between the composed eager recipe
and canonical Diffusers. Independently loaded native executions and their
repetitions match all captured compiled outputs byte for byte. The native
executor implementation is unchanged from the SDXL path. Compiled results
differ from canonical eager results, so exact reference comparisons fail and
performance ratios are withheld. This qualifies export and native replay for
the stated profile; canonical numerical equivalence remains unqualified.

The solver helper's FP16 arithmetic is qualified on CUDA at four and 20 steps
against the canonical scheduler. CPU FP32 trajectories also match, but CPU
FP16 is outside its contract because host scalar multiplication rounds
differently. These scheduler checks do not establish complete-model
qualification for other step counts, shapes, precision modes, or checkpoints.
Timing starts with token IDs, masks, and noise on the GPU and ends with the raw
decoded tensor; it excludes caption processing, loading, transfers, and image
encoding. Outstanding numerical qualification is tracked in
[`TODO.txt`](TODO.txt).

## Prepare and compare fixed profiles

Use the [preparation environment](model-preparation.md#environment-setup),
including the CUDA toolkit, CUPTI, and a built Heliopause extension and Flare
CLI. The complete profiles above are qualified for native replay on one SM120
GPU. Their preparation was exercised with 96 GiB of host memory and two compiler
workers. Keep checkpoint downloads, compilation caches, and exported cores
outside the checkout, with enough disk space for several copies of the weights.

Diffusion recipes require supplemental packages beyond the `prepare` extra.
Install the tested versions while constraining already installed preparation
dependencies, then invoke the environment's Python directly. A subsequent
`uv sync` can remove these supplemental packages.

```sh
diffusion_work=$(mktemp -d)
uv pip freeze --python .venv/bin/python --exclude-editable > "$diffusion_work/constraints.txt"
uv pip install --python .venv/bin/python --constraint "$diffusion_work/constraints.txt" \
  'diffusers==0.43.0' 'beautifulsoup4==4.14.3' 'ftfy==6.3.1' \
  'sentencepiece==0.2.1' 'soupsieve==4.1.1' 'wcwidth==0.9.4'
uv pip check --python .venv/bin/python

sdxl_snapshot=$(.venv/bin/hf download stabilityai/stable-diffusion-xl-base-1.0 \
  --revision 462165984030d82259a11f4367a4eed129e94a7b \
  --include '*.json' '*.txt' '*.fp16.safetensors')
pixart_snapshot=$(.venv/bin/hf download PixArt-alpha/PixArt-XL-2-512x512 \
  --revision 50f702106901db6d0f8b67eb88e814c56ded2692 \
  --include '*.json' '*.txt' '*.model' '*.safetensors')

.venv/bin/python -m ignition.recipes.sdxl.prepare --snapshot "$sdxl_snapshot" \
  --output "$diffusion_work/sdxl-30" --steps 30 --compile-workers 2
.venv/bin/python -m ignition.recipes.pixart.prepare --snapshot "$pixart_snapshot" \
  --output "$diffusion_work/pixart-20" --steps 20 --compile-workers 2
```

Each command runs the two canonical references, enforces exact eager composition,
captures compiled outputs, and publishes a core with both request files. For
the four-step SDXL profile, use `--steps 4` and a separate output directory.
The recipes generate a development publisher; the following native commands
accept the fingerprint recorded by that local preparation. Existing output
directories are rejected, so use fresh native destinations when repeating runs.

```sh
: > "$diffusion_work/flare.toml"
comparison_status=0
for prepared in "$diffusion_work/sdxl-30" "$diffusion_work/pixart-20"; do
  fingerprint=$(.venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["fingerprint"])' \
    "$prepared/publisher-info.json")
  for case in case_a case_b; do
    flare-cli/target/release/flare --config "$diffusion_work/flare.toml" \
      run "$prepared/core" --gpus 0 --offline --verify-weights \
      --publisher-keys "$prepared/publisher.json" \
      --trust-store "$diffusion_work/trust.json" \
      --dangerously-approve-key "$fingerprint" \
      --dangerously-accept-publisher local-development \
      --tensor-request "$prepared/$case/request.json" \
      --tensor-output "$prepared/$case/native" \
      --tensor-warmup 3 --tensor-repetitions 5 || exit $?
    .venv/bin/python -m qualify.sdxl_compare \
      --reference "$prepared/$case/reference.json" \
      --captured "$prepared/$case/captured.json" \
      --native "$prepared/$case/native/report.json" \
      --output "$prepared/$case/comparison.json" --images || comparison_status=1
  done
done
test "$comparison_status" -eq 0
```

The comparison writes per-tensor errors, repeatability, raw timing samples,
and `pytorch.png` and `flare.png`. Native/captured acceptance requires identical
bytes as well as finite values, including identical signed zeros. Canonical
acceptance uses the explicitly recorded numerical bounds, which default to
zero. A failed canonical comparison returns nonzero and withholds timing
ratios even when native replay is byte-identical to the captured graphs.

## Proposed serving contract

This section is designed, not implemented. It describes a concrete iterative
operation boundary; it is not a general-purpose graph interpreter. Introducing
these operations requires the approval specified in
[repository guidance](../AGENTS.md#architecture-and-ownership).

The existing contract remains the token-generation declaration. An additive
iterative contract beside it would describe the same core's tensor-generation
workload. Exactly one serving declaration is selected for a deployment; neither
model identity nor a `supports_diffusion` flag selects native policy. A core with
an iterative declaration must be rejected explicitly by engines that cannot
interpret it. It must not fall through to the token path.

| Declaration | Concrete meaning |
|---|---|
| Prepared profile | One supported combination of request batch, input and output tensor extents, conditioning capacities, and prepared schedule extent. Profiles reference already captured programs. |
| Input bindings | Named sources for request slot, generation, selected profile, input tensors, and expected iteration cursor. Each destination references a typed physical buffer with validated shape, stride, extent, and host staging bounds. |
| Persistent state | Read/write storage views for evolving samples, conditioning features or caches, solver history, noise state, and iteration cursor. Views describe storage; native code does not interpret their numerical contents. |
| Schedule storage | Recipe-resolved timesteps and coefficients with explicit dtype and extent. Shared native code does not choose a sampler, rescale time, or derive coefficients. |
| Operation catalog | Prepared initialization, complete iteration, and finalization programs with exact profile and state dependencies. State access and ownership reuse the existing storage-view mechanisms. |
| Completion | Slot, generation, submission identity, submitted cursor, completed cursor, and status. Iteration progress is independent of emitted token count. |
| Result bindings | One or more named typed tensor views, valid extents within prepared capacity, and presentation metadata such as dimensions, frame rate, or sample rate. Completion status never substitutes for a result payload. |

Prepared tensor signatures must distinguish external request count, packed token
capacity, and effective denoiser work. Classifier-free guidance may use a doubled
batch or separate evaluations. A complete iteration may contain more than one
model evaluation. Requested sampling steps, completed iterations, and denoiser
evaluations are distinct quantities.

Native request preparation validates the supplied tensors or invokes explicitly
supported native tokenization/image preparation. A prepared graph may contain
text encoders and codec computation. Arbitrary Python preprocessing, raw
checkpoint configuration interpretation, and additional frontend asset kinds
are not implied by this proposal. Each frontend algorithm and asset extension
needs a concrete contract and validation.

### Initialization

Initialization validates request identity and initializes every state value read
by subsequent operations. Conditioning computation, initial samples, and required
noise initialization belong to the captured program or explicitly bound inputs.
Preparation-time warmup contents cannot serve as request state.

Randomness has explicit storage and semantics. Raw native replay bypasses PyTorch
generator bookkeeping, as required by
[`PreparedGraphBundle`](../ignition/preparation/graphs.py). An initial seed is not
an implementation of reproducible Gaussian noise. A recipe must capture a
validated random-number mechanism with request-local counter state, or bind
explicit noise tensors whose production and lifetime are specified. Every
random draw affecting later state is covered, including stochastic solver steps.

### Complete iteration

One submitted iteration performs its required denoiser evaluations, prediction
conversion, guidance, solver arithmetic, randomness, and state commit inside
prepared computation. Helios selects a prepared program, binds its declared
inputs, and validates progress. It performs no model-specific numerical update.

The captured program validates the expected request generation and cursor before
mutating state. A successful completion advances the declared iteration cursor;
inactive lanes make no persistent writes. Native code retires the operation only
after matching submission identity, validating progress, and obtaining the
required rank agreement. Pending work retains every input, state, and scratch
owner through its terminal fence.

For fixed schedules with different denoisers, preparation may enumerate a
bounded, ordered catalog of concrete iteration programs. Each entry is an
instance of the same complete-iteration contract. Shared execution advances
through that prepared order; it never tests model identity, noise thresholds,
or a named expert. This is one ordered sequence, without arbitrary branches,
callbacks, jumps, or executable metadata. Data-dependent stage choices must stay
inside captured computation or receive a separate design decision.

Latents, previous model predictions, conditioning caches, and any other values
crossing this boundary reside in declared persistent storage. Compiler-planned
temporaries remain graph-local. Request continuation requires all persistent
state, including random-stream position; checkpointing cannot save only the
latent tensor.

### Finalization and delivery

Finalization verifies the terminal cursor and writes every declared result.
Decoding, latent normalization, image reconstruction, and audio synthesis stay
in captured computation. If temporal decoding spans several operations, its
convolution caches are persistent state with the same lifetime requirements.

Result buffers remain owned until device execution and result transfer finish.
Host delivery has a bounded allocation budget separate from active GPU slots.
Cancellation suppresses publication but cannot recycle in-flight state or result
storage. Step-boundary cancellation does not imply that CUDA work already
submitted is interrupted.

Media results are tensors with concrete formats. Token-canvas results additionally
need a definition of finalized positions and a contiguous streamable prefix.
Final-only token delivery is coherent; arbitrary canvas edits are not equivalent
to existing token SSE output. This proposal does not claim a shared media and
token completion ABI.

## Integration and compatibility

The [physical execution schema](../helios/proto/runtime_core.proto) and
[payload core schema](../helios/proto/core.proto) already describe compiled
resources, allocations, typed buffers, and CUDA graph nodes. They remain the
resource carrier. A serving extension must also update semantic validation,
graph signatures, request packing, scheduling, completion processing, frontend
admission, and result transfer. It cannot be implemented by adding an enum alone.

Existing `ModelContract`, token completions, vocabulary and cache validation,
frontend defaults, signatures, and supported core readers retain their semantics.
An absent iterative declaration retains existing behavior. Existing cores load
without re-export or re-signing. Iterative cores must not manufacture vocabulary,
page tables, pending tokens, or fake text completions to pass token validation.

Production wire fields, operation identifiers, and semantic versions are assigned
only with the approved operation design. The isolated proposal does not add fields
to `ServingContract` or operations to `GraphOperation`. Reserved fields and enum
values remain reserved.
Execution ABI labels do not become loading gates; validation checks actual
operations, buffer geometry, dependencies, and capacities.

## Qualification requirements

Each recipe compares the authoritative model computation, compiled graph replay,
and exported-core native replay on the same conditioning and initial noise.
Floating-point tolerances follow required arithmetic and measured repeatability.
Checks cover every prepared shape, inactive lanes, changed seeds, solver history,
graph transitions, and disposable scratch zeroing. Geometry validation alone
does not establish model correctness.

Lifecycle checks exercise cancellation, stale completions, slot reuse, bounded
result transfer, and shutdown with in-flight work. GPU allocations remain stable
through steady-state serving. If exact continuation is offered, interrupted and
uninterrupted runs compare all relevant persistent state. Compatibility checks
load and serve previously supported cores without re-export. Performance claims
require exported cores running through `flare run` on the specified hardware.
