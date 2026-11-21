# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM MoE DSA checkpoint weights partitioned by attention head and expert channel."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader, MLPWeights, nvfp4_expert_scale_vectors
from ignition.preparation.checkpoint import IndexedCheckpoint
from ignition.preparation.rebuild_manifest import weight_container

from .config import INDEX_DIM, INDEX_HEADS, LATENT_RANK, ROTARY_DIM, ModelConfig

if TYPE_CHECKING:
    import torch
    from b12x.moe.fused_moe import PreparedExperts

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@weight_container(('router', 'correction', 'experts', 'shared'))
@dataclass(frozen=True)
class MoEWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: PreparedExperts
    shared: MLPWeights
    # A16 experts prepared with their calibrated activation scales: a prefill graph may bind A4 launches.
    experts_a4_prefill: bool = False


@weight_container(('attention', 'mlp', 'norms'))
@dataclass(frozen=True)
class LayerWeights:
    attention: dict[str, object]
    mlp: MLPWeights | MoEWeights
    norms: dict[str, torch.Tensor]


@weight_container(('layer', 'tensors', 'draft_head'))
@dataclass(frozen=True)
class MTPWeights:
    layer: LayerWeights
    tensors: dict[str, torch.Tensor]
    # Proposal head. None uses the target vocabulary projection at checkpoint precision.
    draft_head: object | None = None


@dataclass(frozen=True)
class ModelWeights:
    """One rank's canonical modules and packed decoder weights.

    Embeddings, indexer projections, low-rank attention inputs, normalization, and
    routers are replicated. Query and value expansion, attention output, dense
    MLPs, and experts partition by head or intermediate channel. Vocabulary
    weights partition their output rows. Routed layers either share one
    scale-expansion workspace and execute serially, or hold their CSF block
    scales expanded once at load.
    """

    config: ModelConfig
    tree: torch.nn.Module
    shard: TensorParallelShard
    layers: tuple[LayerWeights, ...]
    expert_scale_scratch: tuple[torch.Tensor, torch.Tensor] | None
    allocated_bytes: int
    head: torch.Tensor
    mtp: MTPWeights | None = None

    @property
    def draft(self):
        """The draft model shared generation components ask for: the multi-token prediction layer, if loaded."""
        return self.mtp

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, mtp=False, expert_activation="a16",
             expand_expert_scales=False):
        import torch

        from ignition.preparation.persistent import allocated_tensor_bytes

        from .model import create_model_tree

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GLM MoE DSA model weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
            raise ValueError("GLM MoE DSA prepared execution requires SM120 or SM121")
        before = allocated_tensor_bytes(device)
        with GlmDsaWeightLoader(manifest, device, shard=shard, mtp=mtp,
                                progress=progress if shard.rank == 0 else None,
                                expert_activation=expert_activation) as loader:
            config = loader.config
            tree = create_model_tree(config, draft=mtp)
            for name in ("model.embed_tokens.weight", "model.norm.weight"):
                path, attribute = name.rsplit(".", 1)
                setattr(tree.get_submodule(path), attribute,
                        torch.nn.Parameter(loader.load_tensor(name), requires_grad=False))
            head = loader.load_head()
            scales = None if expand_expert_scales else loader.allocate_expert_scale_scratch()
            layers = tuple(loader.load_layer(index, scale_scratch=scales) for index in range(config.layers))
            draft = loader.load_mtp(scales) if mtp else None
            if draft is not None:
                from ignition.components.vocabulary import nvfp4_draft_head

                draft = replace(draft, draft_head=nvfp4_draft_head(head))
        torch.cuda.synchronize(device)
        return cls(config, tree, shard, layers, scales, allocated_tensor_bytes(device) - before, head, draft)


def checkpoint_quantization(config: ModelConfig, tensors, *, layers):
    """Derive routed-expert storage declarations from the checkpoint's stored tensors.

    The result maps each packed expert group, `model.layers.<n>.mlp.experts`, to
    the ModelOpt entry shared loaders read. A group is packed when its
    projections are stored as NVFP4 nibbles, and its `weight_scale_encoding` is
    `csf` when their block scales are stored as CSF planes. Every projection of a
    group must use one storage. A stored format that contradicts the
    configuration's NVFP4 exclusions is an error, except that the draft layer,
    which the exclusions do not name, may keep BF16 experts. Decoder layers
    require packed experts, whether or not the exclusions name them.
    """
    declarations = {}
    for layer in layers:
        group = f"model.layers.{layer}.mlp.experts"
        formats = set()
        for expert in range(config.experts):
            for projection in _PROJECTIONS:
                module = f"{group}.{expert}.{projection}"
                weight = tensors.get(f"{module}.weight")
                if weight is None:
                    raise ValueError(f"{module}: missing expert projection weight")
                rows, columns = ((config.hidden, config.expert_intermediate) if projection == "down_proj"
                                 else (config.expert_intermediate, config.hidden))
                compressed = f"{module}.weight_scale.nvfp4_csf_fixed" in tensors
                if weight.dtype == "BF16" and weight.shape == (rows, columns):
                    fmt = "bf16"
                elif weight.dtype == "U8" and weight.shape == (rows, columns // 2):
                    scale = tensors.get(f"{module}.weight_scale")
                    if not compressed and (scale is None or scale.dtype != "F8_E4M3"
                                           or scale.shape != (rows, columns // 16)):
                        raise ValueError(f"{module}: packed NVFP4 weight has no E4M3 or CSF block scales")
                    fmt = "nvfp4_csf" if compressed else "nvfp4"
                else:
                    raise ValueError(f"{module}: unsupported expert storage {weight.dtype}{list(weight.shape)}")
                if (fmt != "bf16") != config.quantized(module) and (fmt != "bf16" or layer < config.layers):
                    raise ValueError(f"{module}: stored {fmt} weights contradict the quantization declaration")
                # The loader converts BF16 experts to NVFP4, which only draft proposals tolerate.
                if fmt == "bf16" and layer < config.layers:
                    raise ValueError(f"{module}: BF16 routed experts are prepared only in the draft layer")
                formats.add(fmt)
        if len(formats) != 1:
            raise ValueError(f"{group}: routed experts require one stored weight format")
        fmt = formats.pop()
        if fmt != "bf16":
            declarations[group] = {"quant_algo": "NVFP4", "group_size": 16,
                                   **({"weight_scale_encoding": "csf"} if fmt == "nvfp4_csf" else {})}
    return declarations


class GlmDsaWeightLoader(DirectWeightLoader):
    """Head and intermediate sharding with replicated low-rank attention inputs and indexers.

    Query expansion, key/value expansion, and attention output partition the
    heads. Attention and MLP outputs are partial hidden vectors that require an
    all-reduce. Indexer projections, low-rank input projections, normalization,
    and routers stay replicated. Routed experts partition their intermediate
    dimension, preserving every expert on every rank.
    """

    def __init__(self, manifest: IndexedCheckpoint, device, *, shard: TensorParallelShard,
                 mtp=False, progress=None, expert_activation="a16"):
        super().__init__(manifest, device, progress=progress)
        self.shard, self.mtp = shard, mtp
        if expert_activation not in ("a4", "a16"):
            raise ValueError("GLM MoE DSA NVFP4 experts execute with A4 or A16 activations")
        # A16 experts keep their calibrated activation scales, so a prefill graph can bind A4 launches.
        self.expert_activation = expert_activation
        self.config = ModelConfig.load(manifest.metadata_path)
        self.config.validate_shard(shard)

    @cached_property
    def quantization(self):
        layers = range(self.config.dense_layers, self.config.layers + (self.config.draft_layers if self.mtp else 0))
        return checkpoint_quantization(self.config, self.manifest.tensors, layers=layers)

    def _selected_names(self):
        loaded = self.config.layers + (self.config.draft_layers if self.mtp else 0)

        def selected(name):
            if name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
                return True
            parts = name.split(".")
            return parts[:2] == ["model", "layers"] and parts[2].isdecimal() and int(parts[2]) < loaded

        return {name for name in super()._selected_names() if selected(name)}

    def _layer(self, layer):
        if type(layer) is not int or not 0 <= layer < self.config.layers + self.config.draft_layers:
            raise ValueError("GLM MoE DSA layer index is outside the decoder and its draft layer")
        return f"model.layers.{layer}"

    def load_head(self):
        """This rank's vocabulary rows. The last rank's rows past the vocabulary are zero."""
        import torch

        source = self._sources["lm_head.weight"]
        if source.dtype != torch.bfloat16 or source.shape != (self.config.vocabulary, self.config.hidden):
            raise ValueError("lm_head.weight: expected the checkpoint's BF16 vocabulary projection")
        return self.load_padded_tensor("lm_head.weight", rows=self.config.vocabulary_partition(self.shard))

    def load_attention(self, layer):
        c = self.config
        prefix = self._layer(layer) + ".self_attn"
        query = self.shard.partition(c.heads * (c.content_dim + ROTARY_DIM), alignment=c.content_dim + ROTARY_DIM)
        latent = self.shard.partition(c.heads * (c.content_dim + c.value_dim), alignment=c.content_dim + c.value_dim)
        value = self.shard.partition(c.heads * c.value_dim, alignment=c.value_dim)
        # Adjacent projections of the layer input share a GEMM, and the indexer's key and head weights follow.
        inputs = ["q_a_proj", "kv_a_proj_with_mqa"] + (["indexer.wk", "indexer.weights_proj"] if c.indexed(layer) else [])
        result = {
            "input_proj": self.load_linear_group(tuple(f"{prefix}.{name}" for name in inputs)),
            "q_b_proj": self.load_linear(f"{prefix}.q_b_proj", rows=query),
            "kv_b_proj": self.load_linear(f"{prefix}.kv_b_proj", rows=latent),
            "o_proj": self.load_linear(f"{prefix}.o_proj", columns=value),
        }
        names = ["q_a_layernorm.weight", "kv_a_layernorm.weight"]
        if c.indexed(layer):
            result["indexer.wq_b"] = self.load_linear(f"{prefix}.indexer.wq_b")
            names += ["indexer.k_norm.weight", "indexer.k_norm.bias"]
        for name in names:
            result[name] = self.load_tensor(f"{prefix}.{name}")
        expected = {"input_proj": (c.query_rank + LATENT_RANK + ROTARY_DIM
                                   + (INDEX_DIM + INDEX_HEADS if c.indexed(layer) else 0), c.hidden),
                    "q_b_proj": (query[1] - query[0], c.query_rank),
                    "kv_b_proj": (latent[1] - latent[0], LATENT_RANK),
                    "o_proj": (c.hidden, value[1] - value[0]),
                    "indexer.wq_b": (INDEX_HEADS * INDEX_DIM, c.query_rank)}
        for name, shape in expected.items():
            if name in result and tuple(result[name].shape) != shape:
                raise ValueError(f"{prefix}.{name}: expected BF16 weights of shape {shape}")
        return result

    def load_mlp(self, layer, *, scale_scratch=None):
        c = self.config
        prefix = self._layer(layer) + ".mlp"
        if not c.routed(layer):
            extent = self.shard.partition(c.intermediate, alignment=32)
            return MLPWeights(
                self.load_linear_group((f"{prefix}.gate_proj", f"{prefix}.up_proj"), row_ranges=(extent, extent)),
                self.load_linear(f"{prefix}.down_proj", columns=extent))
        extent = self.shard.partition(c.expert_intermediate, alignment=32)
        shared = MLPWeights(
            self.load_linear_group((f"{prefix}.shared_experts.gate_proj", f"{prefix}.shared_experts.up_proj"),
                                   row_ranges=(extent, extent)),
            self.load_linear(f"{prefix}.shared_experts.down_proj", columns=extent))
        experts, a4_prefill = self.load_experts(layer, scale_scratch=scale_scratch)
        return MoEWeights(self.load_tensor(f"{prefix}.gate.weight"),
                          self.load_tensor(f"{prefix}.gate.e_score_correction_bias"),
                          experts, shared, experts_a4_prefill=a4_prefill)

    def load_experts(self, layer, *, scale_scratch=None):
        """Prepare one layer's routed experts and report whether prefill may bind A4 activations."""
        from ignition.preparation.quantization import expert_format

        c = self.config
        if not c.routed(layer):
            raise ValueError(f"GLM MoE DSA routed experts begin at layer {c.dense_layers}")
        group = self._layer(layer) + ".mlp.experts"
        extent = self.shard.partition(c.expert_intermediate, alignment=32)
        geometry = dict(hidden_size=c.hidden, intermediate_size=c.expert_intermediate, intermediate_range=extent)
        if group not in self.quantization:
            # BF16 draft experts are quantized once at load. Proposals need no checkpoint equivalence.
            return self.load_bf16_experts_as_nvfp4(group, num_experts=c.experts, **geometry), False
        prefixes = tuple(f"{group}.{index}" for index in range(c.experts))
        fmt, mode = expert_format(self.quantization, group)
        scales = nvfp4_expert_scale_vectors(self._sources, prefixes)
        if fmt == "nvfp4_csf":
            return self.load_nvfp4_experts(prefixes, **geometry, scale_vectors=scales, scale_scratch=scale_scratch,
                                           activation_mode=self.expert_activation), self.expert_activation == "a16"
        return self.load_fp4_experts(prefixes, **geometry, format=fmt, activation_mode=mode,
                                     scale_vectors=scales), False

    def allocate_expert_scale_scratch(self):
        """Expansion buffers that serially executed routed layers share for their CSF block scales."""
        import torch

        c = self.config
        local = c.expert_intermediate // self.shard.world_size
        return tuple(torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device) for shape in (
            (c.experts, (2 * local + 127) // 128 * 128, c.hidden // 16),
            (c.experts, c.hidden, (local // 16 + 3) // 4 * 4)))

    def load_norms(self, layer):
        prefix = self._layer(layer)
        return {name: self.load_tensor(f"{prefix}.{name}")
                for name in ("input_layernorm.weight", "post_attention_layernorm.weight")}

    def load_layer(self, layer, *, scale_scratch=None):
        return LayerWeights(self.load_attention(layer), self.load_mlp(layer, scale_scratch=scale_scratch),
                            self.load_norms(layer))

    def load_mtp(self, scale_scratch):
        """The draft layer: a decoder layer after the stack with its embedding and hidden-state mixers."""
        if not self.config.draft_layers:
            raise ValueError("this GLM MoE DSA checkpoint has no multi-token prediction layer")
        layer = self.config.layers
        prefix = self._layer(layer)
        return MTPWeights(self.load_layer(layer, scale_scratch=scale_scratch),
            {name: self.load_tensor(f"{prefix}.{name}.weight")
             for name in ("enorm", "hnorm", "eh_proj", "shared_head.norm")})
