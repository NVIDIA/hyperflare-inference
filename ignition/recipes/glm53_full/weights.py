# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM 5.3 checkpoint weights at checkpoint precision, partitioned over tensor-parallel ranks.

`tensor_partitions` is the single statement of how each checkpoint tensor
divides over ranks. The loader takes every row and column range from it, so
a test of the partition table on checkpoint headers covers the ranges the
loader reads.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import re
from typing import TYPE_CHECKING

from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.planning import SUPPORTED_ARCHITECTURES, SUPPORTED_CAPABILITIES
from ignition.preparation.checkpoint import IndexedCheckpoint
from ignition.preparation.persistent import allocated_tensor_bytes
from ignition.preparation.persistent import persistent_empty
from ignition.preparation.rebuild_manifest import weight_container

from .config import ModelConfig

if TYPE_CHECKING:
    import torch
    from b12x.moe.fused_moe import PreparedExperts

_LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")
_EXPERT = re.compile(r"mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2|input_scale)")
_REPLICATED = frozenset((
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
    "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_a_layernorm.weight",
    "self_attn.indexer.wq_b.weight", "self_attn.indexer.wk.weight", "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
    "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
))
_DRAFT_REPLICATED = frozenset(("eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight"))


@dataclass(frozen=True)
class Partition:
    """One rank's part of a checkpoint tensor: all of it, or a range of stored elements on one axis."""

    axis: int | None = None
    start: int = 0
    stop: int = 0

    def extent(self):
        """Index expression selecting this part of the stored tensor."""
        if self.axis is None:
            return ...
        return (slice(None),) * self.axis + (slice(self.start, self.stop),)


def tensor_partitions(config: ModelConfig, names, shard: TensorParallelShard, *, draft):
    """Map every checkpoint tensor name to this rank's `Partition`, or to None when it is unused.

    Attention heads, MLP intermediate channels, and vocabulary rows partition
    over ranks; a projection that produces a partitioned axis partitions its
    rows, and one that consumes it partitions its columns. Low-rank inputs,
    normalization, routers, and the indexer are replicated. Packed NVFP4
    weights hold two values per byte and one block scale per sixteen values
    along their input axis. The draft layer is unused unless `draft` is set.
    Unknown names raise: a tensor is never dropped silently.
    """
    raw = config.raw
    heads = raw["num_attention_heads"]
    query = raw["qk_nope_head_dim"] + raw["qk_rope_head_dim"]
    key_value = raw["qk_nope_head_dim"] + raw["v_head_dim"]

    def split(axis, extent, alignment, divisor=1):
        first, last = shard.partition(extent, alignment=alignment)
        return Partition(axis, first // divisor, last // divisor)

    result = {}
    for name in names:
        if name in ("model.embed_tokens.weight", "model.norm.weight"):
            result[name] = Partition()
            continue
        if name == "lm_head.weight":
            # Ranks hold equal 64-row-aligned widths, the column gather's unit; the stored range ends at the vocabulary.
            first, last = shard.padded_partition(config.vocabulary, alignment=64)
            result[name] = Partition(0, min(first, config.vocabulary), min(last, config.vocabulary))
            continue
        match = _LAYER.fullmatch(name)
        if match is None or int(match[1]) > config.draft_layer:
            raise ValueError(f"unrecognized GLM 5.3 checkpoint tensor: {name}")
        layer, field = int(match[1]), match[2]
        if layer == config.draft_layer and not draft:
            result[name] = None
            continue
        expert = _EXPERT.fullmatch(field)
        indexer = field.startswith("self_attn.indexer.")
        if field in _REPLICATED and (not indexer or config.selects(layer)) and (
                not field.startswith("mlp.gate.") or config.routed(layer)):
            result[name] = Partition()
        elif layer == config.draft_layer and field in _DRAFT_REPLICATED:
            result[name] = Partition()
        elif field == "self_attn.q_b_proj.weight":
            result[name] = split(0, heads * query, query)
        elif field == "self_attn.kv_b_proj.weight":
            result[name] = split(0, heads * key_value, key_value)
        elif field == "self_attn.o_proj.weight":
            result[name] = split(1, heads * raw["v_head_dim"], raw["v_head_dim"])
        elif not config.routed(layer) and field in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            result[name] = split(0, raw["intermediate_size"], 128)
        elif not config.routed(layer) and field == "mlp.down_proj.weight":
            result[name] = split(1, raw["intermediate_size"], 128)
        elif config.routed(layer) and field in ("mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight"):
            result[name] = split(0, raw["moe_intermediate_size"], 128)
        elif config.routed(layer) and field == "mlp.shared_experts.down_proj.weight":
            result[name] = split(1, raw["moe_intermediate_size"], 128)
        elif config.routed(layer) and expert is not None and int(expert[1]) < raw["n_routed_experts"]:
            projection, tensor = expert[2], expert[3]
            packed = config.projection_format(name.rsplit(".", 1)[0]) == "nvfp4"
            if tensor in ("weight_scale", "weight_scale_2", "input_scale") and not packed:
                raise ValueError(f"{name}: scale tensor of a projection declared unquantized")
            if tensor in ("weight_scale_2", "input_scale"):
                result[name] = Partition()
            elif projection != "down_proj":
                result[name] = split(0, raw["moe_intermediate_size"], 128)
            else:
                divisor = 1 if not packed else 2 if tensor == "weight" else 16
                result[name] = split(1, raw["moe_intermediate_size"], 128, divisor)
        else:
            raise ValueError(f"unrecognized GLM 5.3 checkpoint tensor: {name}")
    return result


def expected_tensor_names(config: ModelConfig):
    """Every tensor name a complete checkpoint holds, from the configuration alone."""
    raw = config.raw
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(config.draft_layer + 1):
        prefix = f"model.layers.{layer}."
        fields = {field for field in _REPLICATED
                  if (config.selects(layer) or not field.startswith("self_attn.indexer."))
                  and (config.routed(layer) or not field.startswith("mlp.gate."))}
        fields |= {f"self_attn.{name}.weight" for name in ("q_b_proj", "kv_b_proj", "o_proj")}
        if layer == config.draft_layer:
            fields |= _DRAFT_REPLICATED
        projections = ("gate_proj", "up_proj", "down_proj")
        if not config.routed(layer):
            fields |= {f"mlp.{name}.weight" for name in projections}
        else:
            fields |= {f"mlp.shared_experts.{name}.weight" for name in projections}
            for expert in range(raw["n_routed_experts"]):
                for name in projections:
                    stem = f"mlp.experts.{expert}.{name}"
                    packed = config.projection_format(prefix + stem) == "nvfp4"
                    fields |= {f"{stem}.{tensor}" for tensor in (
                        ("weight", "weight_scale", "weight_scale_2", "input_scale") if packed else ("weight",))}
        names |= {prefix + field for field in fields}
    return names


@weight_container(("router", "correction", "experts", "shared"))
@dataclass(frozen=True)
class MoEWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: PreparedExperts
    shared: MLPWeights


@weight_container(("gate_up", "down"))
@dataclass(frozen=True)
class BF16Experts:
    """Routed experts stored unquantized: fused gate and up rows, and down columns, of this rank's channels."""

    gate_up: torch.Tensor
    down: torch.Tensor


@weight_container(("attention", "mlp", "norms"))
@dataclass(frozen=True)
class LayerWeights:
    attention: dict[str, object]
    mlp: MLPWeights | MoEWeights
    norms: dict[str, torch.Tensor]


@weight_container(("layer", "embedding_norm", "hidden_norm", "projection", "head_norm"))
@dataclass(frozen=True)
class DraftWeights:
    """The draft layer: one decoder layer behind a projection of the next token's embedding and a hidden state.

    `embedding_norm` and `hidden_norm` normalize the two inputs that
    `projection` joins; `head_norm` precedes the vocabulary projection, which
    the draft shares with the target.
    """

    layer: LayerWeights
    embedding_norm: torch.Tensor
    hidden_norm: torch.Tensor
    projection: torch.Tensor
    head_norm: torch.Tensor


@dataclass(frozen=True)
class ModelWeights:
    """One rank's canonical tree and prepared decoder weights.

    Embeddings and normalization are replicated; vocabulary rows partition
    over ranks. `allocated_bytes` is the device memory the weights retain.
    """

    tree: torch.nn.Module
    config: ModelConfig
    shard: TensorParallelShard
    embedding: torch.Tensor
    norm: torch.Tensor
    layers: tuple[LayerWeights, ...]
    head: object
    allocated_bytes: int
    # The draft layer's weights when preparation speculates; shared generation tests this field.
    draft: object | None = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, expert_activation="a16", layers=None, draft=False):
        """Load one rank. `layers` limits loading to the leading decoder layers, for tests.

        `draft` also loads the draft layer, which speculative preparation uses.
        """
        import torch

        from .model import create_model_tree

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("GLM 5.3 model weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
            raise ValueError(f"GLM 5.3 prepared execution requires {SUPPORTED_ARCHITECTURES}")
        tree = create_model_tree(manifest.snapshot)
        before = allocated_tensor_bytes(device)
        with GlmWeightLoader(manifest, device, shard=shard, draft=draft,
                             progress=progress if shard.rank == 0 else None,
                             expert_activation=expert_activation) as loader:
            count = loader.config.layers if layers is None else layers
            if type(count) is not int or not 0 < count <= loader.config.layers:
                raise ValueError("GLM 5.3 layer count must be within the target decoder")
            embedding = loader._tensor("model.embed_tokens.weight")
            norm = loader._tensor("model.norm.weight")
            head = loader.load_head()
            loaded = tuple(loader.load_layer(index) for index in range(count))
            draft_weights = loader.load_draft() if draft else None
            config = loader.config
        torch.cuda.synchronize(device)
        return cls(tree, config, shard, embedding, norm, loaded, head, allocated_tensor_bytes(device) - before,
                   draft_weights)


class GlmWeightLoader(DirectWeightLoader):
    """Load one rank's tensors with the ranges `tensor_partitions` assigns.

    Output projections of attention and of each MLP produce partial hidden
    vectors that the decoder sums over ranks. Every rank holds all 256 routed
    experts and partitions their intermediate channels.
    """

    def __init__(self, manifest: IndexedCheckpoint, device, *, shard: TensorParallelShard, draft=False,
                 prefixes=None, progress=None, expert_activation="a16"):
        super().__init__(manifest, device, prefixes=prefixes, progress=progress)
        if expert_activation not in ("a4", "a16"):
            raise ValueError("GLM 5.3 NVFP4 experts execute with A4 or A16 activations")
        self.shard, self.draft, self.expert_activation = shard, draft, expert_activation
        self.config = ModelConfig.load(manifest.snapshot)
        self.config.validate_shard(shard)
        expected = expected_tensor_names(self.config)
        if manifest.tensors.keys() != expected:
            difference = sorted(manifest.tensors.keys() ^ expected)
            raise ValueError(f"GLM 5.3 checkpoint tensors differ from the configuration: {difference[0]}")
        self.partitions = tensor_partitions(self.config, manifest.tensors, shard, draft=draft)
        self.consumed = set()

    def __exit__(self, *exc):
        result = super().__exit__(*exc)
        if exc[0] is None:
            def group(name):
                match = _LAYER.fullmatch(name)
                return "" if match is None else match[1]

            # A load limited to leading layers leaves whole later layers unread; a layer
            # that was started must be read completely.
            started = {group(name) for name in self.consumed}
            selected = self._selected_names()
            unread = [name for name in selected if name not in self.consumed and group(name) in started]
            if unread:
                raise ValueError(f"GLM 5.3 loading left checkpoint tensors unread: {min(unread)}")
        return result

    def _selected_names(self):
        return {name for name in super()._selected_names() if self.partitions[name] is not None}

    @functools.cached_property
    def quantization(self):
        """NVFP4 projections in the shared loader's declaration form, derived once from the configuration."""
        return {name.removesuffix(".weight"): {"quant_algo": "NVFP4", "group_size": 16}
                for name in self.partitions if name.endswith(".weight")
                and self.config.projection_format(name.removesuffix(".weight")) == "nvfp4"}

    def _use(self, name):
        if name in self.consumed:
            raise ValueError(f"{name}: checkpoint tensor consumed twice")
        part = self.partitions[name]
        if part is None:
            raise ValueError(f"{name}: checkpoint tensor is unused in this preparation")
        self.consumed.add(name)
        return part

    def _range(self, name, axis):
        part = self._use(name)
        if part.axis is None:
            return None
        if part.axis != axis:
            raise ValueError(f"{name}: partitioned on axis {part.axis}, not {axis}")
        return part.start, part.stop

    def _tensor(self, name):
        if self._use(name).axis is not None:
            raise ValueError(f"{name}: expected a replicated tensor")
        return self.load_tensor(name)

    def _rows(self, prefix):
        return self._range(prefix + ".weight", 0)

    def _columns(self, prefix):
        return self._range(prefix + ".weight", 1)

    def load_head(self):
        """Vocabulary rows of this rank, zero-filled past the vocabulary to the width every rank holds."""
        self._use("lm_head.weight")
        return self.load_padded_tensor("lm_head.weight",
                                       rows=self.shard.padded_partition(self.config.vocabulary, alignment=64))

    def load_layer(self, layer):
        if type(layer) is not int or not 0 <= layer <= self.config.draft_layer:
            raise ValueError("GLM 5.3 decoder layer is out of range")
        prefix = f"model.layers.{layer}"
        return LayerWeights(self.load_attention(layer), self.load_mlp(layer),
            {name: self._tensor(f"{prefix}.{name}.weight") for name in ("input_layernorm", "post_attention_layernorm")})

    def load_draft(self):
        prefix = f"model.layers.{self.config.draft_layer}"
        return DraftWeights(self.load_layer(self.config.draft_layer), self._tensor(f"{prefix}.enorm.weight"),
                            self._tensor(f"{prefix}.hnorm.weight"), self._tensor(f"{prefix}.eh_proj.weight"),
                            self._tensor(f"{prefix}.shared_head.norm.weight"))

    def load_attention(self, layer):
        prefix = f"model.layers.{layer}.self_attn"
        inputs = ["q_a_proj", "kv_a_proj_with_mqa"]
        result = {}
        if self.config.selects(layer):
            inputs += ["indexer.wk", "indexer.weights_proj"]
            result["indexer.wq_b"] = self.load_linear(f"{prefix}.indexer.wq_b", rows=self._rows(f"{prefix}.indexer.wq_b"))
            for name in ("indexer.k_norm.weight", "indexer.k_norm.bias"):
                result[name] = self._tensor(f"{prefix}.{name}")
        # Projections of the layer input share one GEMM; all are replicated.
        result["input_proj"] = self.load_mixed_linear_group(tuple(f"{prefix}.{name}" for name in inputs),
            row_ranges=tuple(self._rows(f"{prefix}.{name}") for name in inputs))
        result["q_b_proj"] = self.load_linear(f"{prefix}.q_b_proj", rows=self._rows(f"{prefix}.q_b_proj"))
        if self.linear_format(f"{prefix}.kv_b_proj") != "bfloat16":
            raise ValueError(f"{prefix}.kv_b_proj: MLA absorption requires BF16 weights")
        # Each head's key rows absorb into the query, and its value rows expand the attended latent. They
        # are stored as two contiguous matrix stacks, the value stack transposed, for batched products.
        raw, heads = self.config.raw, self.config.raw["num_attention_heads"] // self.shard.world_size
        key, value, latent = raw["qk_nope_head_dim"], raw["v_head_dim"], raw["kv_lora_rank"]
        joint = self.load_linear(f"{prefix}.kv_b_proj", rows=self._rows(f"{prefix}.kv_b_proj"),
                                 temporary=True).view(heads, key + value, latent)
        result["key"] = persistent_empty((heads, key, latent), dtype=joint.dtype,
                                         device=joint.device).copy_(joint[:, :key])
        result["value"] = persistent_empty((heads, latent, value), dtype=joint.dtype,
                                           device=joint.device).copy_(joint[:, key:].transpose(1, 2))
        result["o_proj"] = self.load_linear(f"{prefix}.o_proj", columns=self._columns(f"{prefix}.o_proj"))
        for name in ("q_a_layernorm.weight", "kv_a_layernorm.weight"):
            result[name] = self._tensor(f"{prefix}.{name}")
        return result

    def _mlp(self, prefix):
        return MLPWeights(
            self.load_linear_group(tuple(f"{prefix}.{name}" for name in ("gate_proj", "up_proj")),
                row_ranges=tuple(self._rows(f"{prefix}.{name}") for name in ("gate_proj", "up_proj"))),
            self.load_linear(f"{prefix}.down_proj", columns=self._columns(f"{prefix}.down_proj")))

    def load_mlp(self, layer):
        prefix = f"model.layers.{layer}.mlp"
        if not self.config.routed(layer):
            return self._mlp(prefix)
        return MoEWeights(self._tensor(f"{prefix}.gate.weight"), self._tensor(f"{prefix}.gate.e_score_correction_bias"),
                          self.load_experts(layer), self._mlp(prefix + ".shared_experts"))

    def load_experts(self, layer):
        """Routed experts of one layer in the checkpoint's packed NVFP4 storage."""
        raw = self.config.raw
        prefixes = tuple(f"model.layers.{layer}.mlp.experts.{index}" for index in range(raw["n_routed_experts"]))
        projections = ("gate_proj", "up_proj", "down_proj")
        formats = {self.config.projection_format(f"{prefix}.{name}") for prefix in prefixes for name in projections}
        if formats == {"bf16"}:
            return self._load_bf16_experts(prefixes)
        if formats != {"nvfp4"}:
            raise ValueError(f"model.layers.{layer}: routed experts require NVFP4 storage, found {sorted(formats)}")
        ranges = set()
        for prefix in prefixes:
            for name in projections:
                for tensor in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                    part = self._use(f"{prefix}.{name}.{tensor}")
                    if tensor == "weight" and name != "down_proj":
                        ranges.add((part.start, part.stop))
        if len(ranges) != 1:
            raise ValueError(f"model.layers.{layer}: routed experts require one intermediate range")
        return self.load_fp4_experts(prefixes, hidden_size=raw["hidden_size"],
            intermediate_size=raw["moe_intermediate_size"], format="nvfp4", activation_mode=self.expert_activation,
            intermediate_range=ranges.pop(),
            scale_vectors=_expert_scales(self._sources, prefixes) if self.expert_activation == "a4" else None)


    def _load_bf16_experts(self, prefixes):
        """Stack unquantized experts: `[expert, gate rows then up rows, hidden]` and `[expert, hidden, columns]`."""
        import torch

        hidden = self.config.raw["hidden_size"]
        ranges = {name: {self._range(f"{prefix}.{name}.weight", int(name == "down_proj")) for prefix in prefixes}
                  for name in ("gate_proj", "up_proj", "down_proj")}
        if len(set().union(*ranges.values())) != 1:
            raise ValueError("GLM 5.3 unquantized routed experts require one intermediate range")
        first, last = next(iter(ranges["gate_proj"]))
        local = last - first
        gate_up = persistent_empty((len(prefixes), 2 * local, hidden), dtype=torch.bfloat16, device=self.device)
        down = persistent_empty((len(prefixes), hidden, local), dtype=torch.bfloat16, device=self.device)
        for index, prefix in enumerate(prefixes):
            sources = {name: self._sources[f"{prefix}.{name}.weight"] for name in ranges}
            if any(source.dtype != torch.bfloat16 for source in sources.values()):
                raise ValueError(f"{prefix}: declared unquantized experts disagree with checkpoint storage")
            self._copy(gate_up[index, :local], sources["gate_proj"][first:last])
            self._copy(gate_up[index, local:], sources["up_proj"][first:last])
            self._copy(down[index], sources["down_proj"][:, first:last])
        self._session.flush()
        return BF16Experts(gate_up, down)


def _expert_scales(sources, prefixes):
    """Per-expert global weight scales and reciprocal calibrated activation scales for A4 execution."""
    import torch

    w13, w2, a1, a2 = [], [], [], []
    for prefix in prefixes:
        values = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("weight_scale_2", "input_scale"):
                name = f"{prefix}.{projection}.{suffix}"
                value = sources[name]
                if (value.device.type != "cpu" or value.dtype != torch.float32 or value.numel() != 1
                        or not bool((torch.isfinite(value) & (value > 0)).all())):
                    raise ValueError(f"{name}: expected a finite positive FP32 scalar")
                values[projection, suffix] = value.item()
        for suffix in ("weight_scale_2", "input_scale"):
            if values["gate_proj", suffix] != values["up_proj", suffix]:
                raise ValueError(f"{prefix}: gate and up {suffix} differ; their fused projection has one scale")
        w13.append(values["up_proj", "weight_scale_2"])
        w2.append(values["down_proj", "weight_scale_2"])
        a1.append(values["up_proj", "input_scale"])
        a2.append(values["down_proj", "input_scale"])
    return (torch.tensor(w13, dtype=torch.float32), torch.tensor(w2, dtype=torch.float32),
            torch.tensor(a1, dtype=torch.float32).reciprocal(), torch.tensor(a2, dtype=torch.float32).reciprocal())
