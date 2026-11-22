# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-preserving tensor partitions for Inkling2's dense DSpark decoder."""

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path

import torch

from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header
from ignition.preparation.persistent import persistent_copy, persistent_empty
from ignition.preparation.rebuild_manifest import weight_container


def mtp_quantization(configuration, names):
    groups = configuration.get("mtp", {}).get("parameter_groups", ())
    declared, quantized = set(), {}
    for group in groups:
        tensors = group.get("tensors", ())
        if not tensors or declared.intersection(tensors):
            raise ValueError("Inkling2 MTP precision groups must contain distinct tensor names")
        declared.update(tensors)
        if group.get("compute_dtype") == "bfloat16":
            if (group.get("output_dtype") != "bfloat16" or group.get("weight_transform") != "cast"
                    or group.get("required_activation_quantization") is not None
                    or group.get("weight_block_shape") != []):
                raise ValueError("Inkling2 MTP BF16 weights require an unquantized precision declaration")
        elif group.get("compute_dtype") == "mxfp8":
            if (group.get("output_dtype") != "float8_e4m3fn"
                    or group.get("weight_transform") != "mxfp8_nearest"
                    or group.get("weight_block_shape") != [1, 32]
                    or group.get("required_activation_quantization") != {
                        "block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}):
                raise ValueError("Inkling2 MTP MXFP8 weights require nearest E4M3 activations and UE8M0 K32 scales")
            for name in tensors:
                if not name.endswith(".weight") or ".mlp." not in name:
                    raise ValueError("Inkling2 MTP MXFP8 storage is defined for dense MLP matrices")
                quantized[name.removesuffix(".weight")] = {"quant_algo": "MXFP8", "group_size": 32}
        else:
            raise ValueError("Inkling2 MTP weights require BF16 or MXFP8 precision groups")
    expected = declared | {prefix + ".weight_scale" for prefix in quantized}
    if expected != set(names):
        raise ValueError("Inkling2 MTP precision groups must cover exactly the sidecar tensors")
    return quantized


class DraftWeightLoader(DirectWeightLoader):
    @cached_property
    def quantization(self):
        configuration = json.loads((self.manifest.snapshot / "hf_quant_config.json").read_text())
        return mtp_quantization(configuration, self.manifest.tensors)


class DraftCheckpoint(IndexedCheckpoint):
    @property
    def index_path(self):
        return None


@weight_container(('qkv', 'context_kv', 'output', 'query_norm', 'key_norm', 'context_key_norm', 'context_norm', 'input_norm', 'post_attention_norm', 'mlp'))
@dataclass(frozen=True)
class DraftLayerWeights:
    qkv: torch.Tensor
    context_kv: torch.Tensor
    output: torch.Tensor
    query_norm: torch.Tensor
    key_norm: torch.Tensor
    context_key_norm: torch.Tensor
    context_norm: torch.Tensor
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@weight_container(('embedding', 'state_weight', 'context_weight', 'bias', 'head', 'head_bias', 'context_norm'))
@dataclass(frozen=True)
class RecurrentWeights:
    embedding: torch.Tensor
    state_weight: torch.Tensor
    context_weight: torch.Tensor
    bias: torch.Tensor
    head: torch.Tensor
    head_bias: torch.Tensor
    context_norm: torch.Tensor
    softcap: float = 30.


def fold_recurrent_head(head, hidden_out, hidden_bias, *, chunk=4096, temporary=False):
    """Fold the BF16 output map through the vocabulary head with FP64 products."""
    if (head.ndim != 2 or hidden_out.ndim != 2 or hidden_out.shape[0] != head.shape[1]
            or hidden_bias.shape != (head.shape[1],)
            or any(t.dtype != torch.bfloat16 or t.device != head.device for t in (head, hidden_out, hidden_bias))):
        raise ValueError("recurrent output folding requires compatible BF16 output and vocabulary matrices")
    shape = (head.shape[0], hidden_out.shape[1])
    weight = (torch.empty(shape, dtype=torch.bfloat16, device=head.device) if temporary
              else persistent_empty(shape, torch.bfloat16, head.device))
    bias = persistent_empty((head.shape[0],), torch.bfloat16, head.device)
    projection, offset = hidden_out.double(), hidden_bias.double()
    for start in range(0, head.shape[0], chunk):
        rows = slice(start, start + chunk)
        source = head[rows].double()
        weight[rows].copy_(source @ projection)
        bias[rows].copy_(source @ offset)
    return weight, bias


@weight_container(('alpha', 'mask_embedding', 'norm', 'layers', 'recurrent', 'draft_head'))
@dataclass(frozen=True)
class DraftWeights:
    alpha: torch.Tensor
    mask_embedding: torch.Tensor
    norm: torch.Tensor
    layers: tuple[DraftLayerWeights, ...]
    recurrent: RecurrentWeights
    draft_head: object | None = None

    @classmethod
    def load(cls, snapshot, config, device, shard, *, target_head, progress=None,
             draft_head_quantization=None, markov_head_quantization=None):
        if draft_head_quantization not in (None, "nvfp4", "mxfp8"):
            raise ValueError(f"unsupported DSpark draft head quantization: {draft_head_quantization!r}")
        if markov_head_quantization not in (None, "nvfp4"):
            raise ValueError(f"unsupported DSpark Markov head quantization: {markov_head_quantization!r}")
        snapshot = Path(snapshot)
        sidecar, tensors = _read_header(snapshot / "mtp.safetensors", "mtp.safetensors")
        manifest = DraftCheckpoint(snapshot, tensors, (sidecar,))
        d = config.draft
        hidden = config.text["hidden_size"]
        q_rows = shard.partition(d.heads * d.head_dim, alignment=d.head_dim)
        kv_rows = shard.partition(d.kv_heads * d.head_dim, alignment=d.head_dim)
        intermediate = shard.partition(d.text["intermediate_size"], alignment=32)
        with DraftWeightLoader(manifest, device, progress=progress, progress_label="Load DSpark weights") as loader:
            loader.quantization

            def vector(name, width):
                value = loader.load_tensor(name)
                if value.dtype != torch.bfloat16 or value.shape != (width,):
                    raise ValueError(f"{name} requires BF16 shape [{width}]")
                return value

            layers = []
            for index in range(d.layers):
                prefix = f"layers.{index}"
                attention = prefix + ".self_attn."
                qkv = loader.load_linear_group(tuple(attention + role + "_proj" for role in ("q", "k", "v")),
                    row_ranges=(q_rows, kv_rows, kv_rows))
                context = loader.load_linear_group(tuple(attention + role + "_proj_ctx" for role in ("k", "v")),
                    row_ranges=(kv_rows, kv_rows))
                output = loader.load_linear(attention + "o_proj", columns=q_rows)
                mlp = MLPWeights(loader.load_linear_group(tuple(prefix + ".mlp." + role + "_proj" for role in ("gate", "up")),
                    row_ranges=(intermediate, intermediate)),
                    loader.load_linear(prefix + ".mlp.down_proj", columns=intermediate))
                norms = [vector(attention + name + ".weight", d.head_dim)
                         for name in ("q_norm", "k_norm", "k_norm_ctx")]
                norms += [vector(prefix + "." + name + ".weight", hidden)
                          for name in ("dflare_norm", "input_layernorm", "post_attention_layernorm")]
                layers.append(DraftLayerWeights(qkv, context, output, *norms, mlp))
            fuse = loader.load_tensor("fuse_weight", temporary=True)
            if fuse.shape != (d.layers + 1, len(d.auxiliary_layers)) or fuse.dtype != torch.bfloat16:
                raise ValueError("Inkling2 fuse_weight must provide one BF16 target-tap mixture per draft layer and recurrence")
            alpha = persistent_copy(fuse.float().softmax(-1).bfloat16())
            state = loader.load_tensor("markov_head.rnn_proj.weight", temporary=True)
            rank = d.markov_rank
            if state.dtype != torch.bfloat16 or state.shape != (3 * rank, 2 * rank + 2 * hidden):
                raise ValueError("Inkling2 recurrent projection must consume state, predecessor embedding, draft hidden, and target context")
            embedding = loader.load_tensor("markov_head.markov_w1.weight")
            if embedding.dtype != torch.bfloat16 or embedding.shape != (config.vocabulary, rank):
                raise ValueError("Inkling2 predecessor embedding must be a BF16 vocabulary by recurrent-rank matrix")
            hidden_out = loader.load_tensor("markov_head.hidden_out.weight", temporary=True)
            hidden_bias = vector("markov_head.hidden_out.bias", hidden)
            head, head_bias = fold_recurrent_head(
                target_head, hidden_out, hidden_bias, temporary=markov_head_quantization is not None
            )
            if markov_head_quantization == "nvfp4":
                from ignition.components.vocabulary import nvfp4_draft_head
                head = nvfp4_draft_head(head)
            recurrent = RecurrentWeights(embedding, persistent_copy(state[:, :2 * rank].contiguous()),
                persistent_copy(state[:, 2 * rank:].contiguous()), persistent_copy(vector("markov_head.rnn_proj.bias", 3 * rank).float()),
                head, head_bias, vector("markov_head.dflare_norm.weight", hidden))
            draft_head = None
            if draft_head_quantization is not None:
                from ignition.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
                draft_head = {"nvfp4": nvfp4_draft_head, "mxfp8": mxfp8_draft_head}[draft_head_quantization](target_head)
            return cls(alpha, vector("mask_embed", hidden), vector("norm.weight", hidden), tuple(layers), recurrent, draft_head)
