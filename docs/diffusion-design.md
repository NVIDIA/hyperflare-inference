# Prepared diffusion execution

Ignition prepares complete SDXL and PixArt pipelines as signed CUDA graph cores.
`flare run --tensor-request` loads a core in a separate Rust process, uploads the
request tensors, replays its ordered graphs and downloads persistent outputs.
Both models use the same native executor; text encoding, denoising, guidance,
solver arithmetic and image decoding remain prepared model computation.

This path executes one fixed tensor request on one GPU. Shapes and schedules
are fixed during preparation; changing the step count requires another core.
It does not provide concurrent admission, cancellation, continuation, native
prompt tokenization or an HTTP image endpoint. The request JSON describes
physical allocation views and graph order, not a signed semantic serving
contract, and must match the preparation that exported the core.

See the [tensor CLI](cli-design.md#fixed-tensor-execution) for request fields,
validation and output files; the [runtime contract](runtime-core-design.md#fixed-tensor-requests)
for resource ownership; and the [core format](core-format-design.md) for signing
and compatibility with token-serving cores.

## Prepared profiles

Both preparation commands accept a local checkpoint snapshot and run two prompts
with noise seeds 42 and 43. They are separate from the token-model recipe registry.
The pinned checkpoint revisions are included in the commands below.

| Recipe | Fixed profile | Prepared computation |
|---|---|---|
| [SDXL Base 1.0](../ignition/recipes/sdxl/prepare.py) | One 1024-square image; Euler updates; default 4 steps, qualified at 4 and 30 | Two CLIP encoders, U-Net, classifier-free guidance, Euler updates and VAE decode |
| [PixArt-Alpha XL 512](../ignition/recipes/pixart/prepare.py) | One 512-square image; 120 text positions; guidance 4.5; default 20 DPM-Solver++ updates | T5, diffusion transformer, classifier-free guidance, solver updates and VAE decode |

SDXL requests contain both tokenizers' IDs, initial noise, guidance and size/crop
conditioning. The model uses FP16 checkpoint weights with the required FP32 VAE
upcast. Four steps provide a short correctness check, not an image-quality preset.

PixArt requests contain positive/negative token IDs and masks, initial noise and
guidance. Caption cleaning and resolution binning follow Diffusers. T5 retains
its FP16/FP32 parameter mix; the transformer and decoder use FP16. The
[prepared solver](../ignition/recipes/pixart/scheduler.py) uses second-order
midpoint DPM-Solver++ with epsilon prediction and first-order initial/final
updates. Earlier converted predictions have persistent storage.

## Numerical qualification

Preparation first checks composed eager forwards against the canonical Diffusers
pipeline, then saves captured outputs. The [comparison tool](../qualify/sdxl_compare.py)
checks independently loaded native outputs against both references, including
conditioning, initialized latents, every update, solver history and decoded images.
Default acceptance is exact; tolerances must be explicit. Raw tensors and saved
repeat outputs establish agreement before image conversion.

On NVIDIA RTX 6000D (SM120), with PyTorch 2.14.1+cu132 and Diffusers 0.43.0,
all saved outputs and saved repeats match byte for byte for both prompts:

| Profile | Graphs | Saved outputs per prompt | Decoded-image MAE, seeds 42 / 43 |
|---|---:|---:|---:|
| SDXL, 4 steps | 7 | 8 | 0 / 0 |
| SDXL, 30 steps | 33 | 34 | 0 / 0 |
| PixArt, 20 steps | 23 | 43 | 0 / 0 |

This qualifies these fixed profiles and inputs. Other shapes, schedules,
checkpoints and performance remain unqualified. Native processes load neither
Python nor PyTorch.

Replay timing starts with input tensors on the GPU and ends with the raw decoded
tensor. It excludes loading, tokenization, transfers and image encoding. The eager
SDXL pipeline converts VAE precision per inference, while the prepared core retains
converted weights; timing ratios compare whole paths, not isolated launch overhead.

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
