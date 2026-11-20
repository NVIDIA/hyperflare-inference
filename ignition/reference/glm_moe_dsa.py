# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Layer-streamed GLM 5.3 text reference from the Transformers `glm_moe_dsa` forwards.

The reference runs a `GlmMoeDsaForCausalLM` checkpoint one decoder layer at a
time, so the model is never resident at once. Attention, the sparse-attention
indexer, rotary positions, normalization, routing, and dense MLPs execute the
installed Transformers modules with checkpoint weights. Routed experts read one
checkpoint expert at a time; NVFP4 experts decode their checkpoint weights and
apply either BF16 activations or the checkpoint's calibrated NVFP4 activation
quantization. No prepared Flare operator executes in this path.

The Transformers model class does not load the checkpoint's draft layer
(`model.layers.78`), so this reference covers the 78 target layers only.

Run ``python -m ignition.reference.glm_moe_dsa SNAPSHOT INPUT_JSON OUTPUT_PT``.
The JSON input contains ``token_ids``. The output stores those IDs, the hidden
states after the requested layers, each indexer layer's selected positions, the
final normalized hidden states, the greedy next token and top-two logit margin
of every row, and the logits of the trailing rows. Logit row j predicts the
token after input positions zero through j.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.reference.qwen38_quant import Nvfp4Projection, nvfp4_expert_reference, nvfp4_linear_reference

# Transformers keeps these weights in FP32 (`_keep_in_fp32_modules*` of `GlmMoeDsaPreTrainedModel`).
_FP32_SUFFIXES = ("indexer.weights_proj.weight", "e_score_correction_bias")


class CheckpointReader:
    """Read whole tensors or leading-axis rows from an indexed checkpoint onto one device."""

    def __init__(self, manifest, device, dtype):
        self.manifest, self.device, self.dtype = manifest, device, dtype
        self._files = {}

    def raw(self, name, row=None):
        metadata = self.manifest.tensors[name]
        if metadata.shard not in self._files:
            self._files[metadata.shard] = safe_open(self.manifest.snapshot / metadata.shard, framework="pt", device="cpu")
        file = self._files[metadata.shard]
        value = file.get_tensor(name) if row is None else file.get_slice(name)[row:row + 1][0]
        # Complete the transfer before the mapped checkpoint storage can be released.
        return value.to(self.device, non_blocking=False)

    def floating(self, name, row=None):
        value = self.raw(name, row)
        return value.float() if name.endswith(_FP32_SUFFIXES) else value.to(self.dtype)

    def close(self):
        self._files.clear()

    def install(self, module, prefix):
        """Replace every parameter and buffer of `module`, except routed experts, with checkpoint values."""
        for collection in (module.named_parameters(), module.named_buffers()):
            for name, _ in tuple(collection):
                if name.startswith("mlp.experts."):
                    continue
                path, _, attribute = name.rpartition(".")
                target = module.get_submodule(path) if path else module
                value = self.floating(prefix + "." + name)
                if attribute in target._parameters:
                    value = torch.nn.Parameter(value, requires_grad=False)
                setattr(target, attribute, value)


class StreamingExperts(torch.nn.Module):
    """Routed experts that read each selected expert from the checkpoint when it is used.

    `activation` is "a16" for BF16 activations or "a4" for the checkpoint's
    calibrated NVFP4 activation quantization. BF16 experts ignore it.
    """

    def __init__(self, canonical, reader, prefix, *, activation):
        super().__init__()
        if activation not in ("a16", "a4"):
            raise ValueError("expert activation must be a16 or a4")
        self.reader, self.prefix, self.activation = reader, prefix, activation
        self.act = canonical.act_fn
        self.quantized = f"{prefix}.0.gate_proj.weight_scale" in reader.manifest.tensors

    def projection(self, expert, name):
        prefix = f"{self.prefix}.{expert}.{name}"
        return Nvfp4Projection(self.reader.raw(prefix + ".weight"), self.reader.raw(prefix + ".weight_scale"),
            self.reader.raw(prefix + ".weight_scale_2"),
            self.reader.raw(prefix + ".input_scale") if self.activation == "a4" else None)

    def expert(self, current, expert):
        if not self.quantized:
            gate, up, down = (self.reader.floating(f"{self.prefix}.{expert}.{name}.weight")
                              for name in ("gate_proj", "up_proj", "down_proj"))
            linear = torch.nn.functional.linear
            return linear(self.act(linear(current, gate)) * linear(current, up), down)
        gate, up, down = (self.projection(expert, name) for name in ("gate_proj", "up_proj", "down_proj"))
        if self.activation == "a4":
            return nvfp4_expert_reference(current, gate, up, down).down
        intermediate = (self.act(nvfp4_linear_reference(current, gate))
                        * nvfp4_linear_reference(current, up)).to(torch.bfloat16)
        return nvfp4_linear_reference(intermediate, down)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        output = torch.zeros_like(hidden_states)
        for expert in top_k_index.unique(sorted=True).tolist():
            rows, slots = torch.where(top_k_index == expert)
            contribution = self.expert(hidden_states[rows], expert) * top_k_weights[rows, slots, None]
            output.index_add_(0, rows, contribution.to(output.dtype))
        return output


@torch.no_grad()
def text_reference(snapshot, token_ids, *, device="cuda:0", dtype=None, expert_activation="a16", layers=None,
                   progress=None, layer_hook=None, position_hook=None, teacher=None):
    """Run the canonical target decoder and return its hidden states and logits on the CPU.

    `layers` stops after that many decoder layers and omits the final norm and
    logits. `progress(index, hidden, selected)` observes each layer's output and
    its sparse-attention selection. `layer_hook(index, layer)` and
    `position_hook(cos, sin)` let a test alter one canonical layer or the rotary
    embeddings to produce a deliberately wrong model. `teacher`, a fixture of
    the same token sequence, isolates layers: every layer after the first
    receives the fixture's output of the preceding layer, and a layer without
    an indexer receives the fixture's selection, so each recorded output shows
    one layer's own error without accumulation.
    """
    from transformers.masking_utils import create_causal_mask
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
        GlmMoeDsaDecoderLayer, GlmMoeDsaRMSNorm, GlmMoeDsaRotaryEmbedding,
    )

    if not isinstance(token_ids, (list, tuple)) or not token_ids or any(type(token) is not int for token in token_ids):
        raise ValueError("reference token IDs must be a nonempty integer sequence")
    snapshot, device = Path(snapshot), torch.device(device)
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    configuration = json.loads((snapshot / "config.json").read_text())
    if configuration.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        raise ValueError("text reference requires GlmMoeDsaForCausalLM")
    config = GlmMoeDsaConfig.from_dict(configuration)
    config._attn_implementation = "eager"
    count = config.num_hidden_layers if layers is None else layers
    if type(count) is not int or not 0 < count <= config.num_hidden_layers:
        raise ValueError("reference layer count must be within the target decoder")
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=device).reshape(1, -1)
    if bool(((ids < 0) | (ids >= config.vocab_size)).any()):
        raise ValueError("reference token IDs must be inside the checkpoint vocabulary")
    reader = CheckpointReader(inspect_indexed_checkpoint(snapshot), device, dtype)
    unique, inverse = ids.flatten().unique(sorted=True, return_inverse=True)
    embeddings = torch.stack([reader.floating("model.embed_tokens.weight", int(token)) for token in unique.tolist()])
    hidden = embeddings[inverse].reshape(1, -1, config.hidden_size)
    position_ids = torch.arange(ids.shape[1], device=device).reshape(1, -1)
    position_embeddings = GlmMoeDsaRotaryEmbedding(config).to(device)(hidden, position_ids)
    if position_hook is not None:
        position_embeddings = position_hook(*position_embeddings)
    mask = create_causal_mask(config=config, inputs_embeds=hidden, attention_mask=None, past_key_values=None,
                              position_ids=position_ids, allow_is_causal_skip=False)
    selected = None
    if teacher is not None and not torch.equal(teacher["token_ids"], ids.flatten().cpu()):
        raise ValueError("the teacher fixture holds a different token sequence")
    for index in range(count):
        if teacher is not None and index:
            hidden = teacher["layers"][index - 1].to(device=device, dtype=dtype).unsqueeze(0)
            source = max(layer for layer in teacher["selections"] if layer < index)
            selected = teacher["selections"][source].to(device).unsqueeze(0)
        with torch.device("meta"):
            layer = GlmMoeDsaDecoderLayer(config, index)
        prefix = f"model.layers.{index}"
        reader.install(layer, prefix)
        if config.mlp_layer_types[index] == "sparse":
            layer.mlp.experts = StreamingExperts(layer.mlp.experts, reader, prefix + ".mlp.experts",
                                                 activation=expert_activation)
        if layer_hook is not None:
            layer_hook(index, layer)
        hidden, selected = layer.eval()(hidden, attention_mask=mask, position_embeddings=position_embeddings,
            position_ids=position_ids, prev_topk_indices=selected)
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError(f"GLM reference layer {index} produced nonfinite hidden states")
        if progress is not None:
            progress(index, hidden.squeeze(0), selected.squeeze(0) if layer.self_attn.indexer is not None else None)
        del layer
    if layers is not None:
        reader.close()
        return hidden.squeeze(0).float().cpu(), None
    final_norm = GlmMoeDsaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    final_norm.weight = torch.nn.Parameter(reader.floating("model.norm.weight"), requires_grad=False)
    hidden = final_norm(hidden)
    logits = torch.nn.functional.linear(hidden, reader.floating("lm_head.weight"))
    reader.close()
    hidden, logits = hidden.squeeze(0).float().cpu(), logits.squeeze(0).float().cpu()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("GLM reference produced nonfinite logits")
    return hidden, logits


def compare_fixtures(actual, expected, *, decisive_margin=1.0):
    """Measure one saved forward against a reference fixture of the same token sequence.

    Returns, per decoder layer present in both, the relative L2 error of the
    layer output; per indexer layer present in both, the mean fraction of each
    row's reference selection that the measured selection contains, counting
    only positions at or before the row; the relative L2 error of the final
    normalized hidden states; and the rows whose greedy token differs where
    the reference top-two logit margin is at least `decisive_margin`. The
    function applies no tolerance.
    """
    if not torch.equal(actual["token_ids"], expected["token_ids"]):
        raise ValueError("fixtures hold different token sequences")

    def relative(value, reference):
        return float((value.double() - reference.double()).norm() / reference.double().norm())

    result = {"layers": {index: relative(actual["layers"][index], reference)
                         for index, reference in sorted(expected["layers"].items()) if index in actual["layers"]},
              "selections": {}}
    rows = expected["token_ids"].numel()
    causal = torch.arange(rows)[:, None]
    for index, reference in sorted(expected["selections"].items()):
        if index not in actual["selections"]:
            continue
        chosen = torch.zeros((rows, rows), dtype=torch.bool).scatter_(1, actual["selections"][index].long(), True)
        valid = reference.long() <= causal
        result["selections"][index] = float(((chosen.gather(1, reference.long()) & valid).sum(1) / valid.sum(1)).mean())
    if actual["hidden"].shape == expected["hidden"].shape:
        result["hidden"] = relative(actual["hidden"], expected["hidden"])
    if "next_tokens" in actual and "next_tokens" in expected:
        decisive = expected["margins"] >= decisive_margin
        differing = decisive & (actual["next_tokens"] != expected["next_tokens"])
        result.update(decisive_rows=int(decisive.sum()), greedy_mismatches=int(differing.sum()),
                      logits=relative(actual["logits"], expected["logits"]))
    return result


def compare_layer_updates(actual, expected):
    """Measure each layer's residual update of a layer-isolated forward against its teacher fixture.

    `actual` must come from `text_reference(..., teacher=expected)` or from an
    implementation given the same layer inputs. For layer i > 0 the update is
    the layer output minus the teacher's output of layer i - 1. Returns per
    layer the relative L2 error of the update over all rows and the median,
    99th percentile, and maximum of the per-row relative error. The function
    applies no tolerance.
    """
    if not torch.equal(actual["token_ids"], expected["token_ids"]):
        raise ValueError("fixtures hold different token sequences")
    result = {}
    for index in sorted(expected["layers"]):
        if not index or index not in actual["layers"] or index - 1 not in expected["layers"]:
            continue
        source = expected["layers"][index - 1].double()
        update = expected["layers"][index].double() - source
        error = (actual["layers"][index].double() - source - update).norm(dim=-1) / update.norm(dim=-1)
        result[index] = dict(relative_l2=float((actual["layers"][index].double() - source - update).norm() / update.norm()),
                             median=float(error.median()), p99=float(error.quantile(0.99)), maximum=float(error.max()))
    return result


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("input", type=Path, help="JSON containing token_ids")
    parser.add_argument("output", type=Path, help="PyTorch fixture of hidden states, selections and logits")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expert-activation", choices=("a16", "a4"), default="a16")
    parser.add_argument("--layers", type=int, help="stop after this many decoder layers, without logits")
    parser.add_argument("--record-layers", type=int, nargs="*",
                        help="decoder layers whose output is stored; every executed layer when omitted")
    parser.add_argument("--logit-rows", type=int, default=64, help="trailing rows whose full logits are stored")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    ids = json.loads(args.input.read_text())["token_ids"]
    started = time.monotonic()
    recorded, selections = {}, {}

    def progress(index, hidden, selected):
        if args.record_layers is None or index in args.record_layers:
            recorded[index] = hidden.float().cpu().clone()
        if selected is not None:
            selections[index] = selected.cpu().clone()
        print(f"layer={index} tokens={hidden.shape[0]} elapsed_seconds={time.monotonic() - started:.3f}", flush=True)

    hidden, logits = text_reference(args.snapshot, ids, device=args.device, expert_activation=args.expert_activation,
                                    layers=args.layers, progress=progress)
    result = dict(token_ids=torch.tensor(ids), layers=recorded, selections=selections, hidden=hidden,
                  expert_activation=args.expert_activation, device=args.device)
    if logits is not None:
        top = logits.topk(2, dim=-1)
        result.update(next_tokens=top.indices[:, 0].clone(), margins=(top.values[:, 0] - top.values[:, 1]).clone(),
                      logits=logits[-args.logit_rows:].clone())
        print(dict(next_token=int(top.indices[-1, 0]), margin=float(top.values[-1, 0] - top.values[-1, 1])), flush=True)
    torch.save(result, args.output)
    print(dict(output=str(args.output), elapsed_seconds=time.monotonic() - started), flush=True)


if __name__ == "__main__":
    main()
