# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native FP8/MXFP4 checkpoint transfers and tensor-parallel partitions."""

from dataclasses import dataclass, replace

import torch

from ignition.components.weights import MLPWeights
from ignition.components.megamoe_tiers import MegaMoeLayerLayout
from ignition.components.fused_mla import FusedMLAPreparation
from ignition.components.mega_attention_weights import FusedMLAOutputWeights, query_planes, output_planes, pack_output_planes
from ignition.components.block_fp8_weights import BlockFP8WeightLoader
from ignition.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
from ignition.components.markov_sampling import fp8_markov_head
from ignition.preparation.persistent import persistent_copy, allocated_tensor_bytes

from .config import ModelConfig


@dataclass(frozen=True)
class ExpertWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: object
    shared: MLPWeights | None
    vision_correction: torch.Tensor | None = None


@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    experts: ExpertWeights
    residual: dict
    engram: dict | None


@dataclass(frozen=True)
class DSparkWeights:
    layers: tuple[LayerWeights, ...]
    main_projection: object
    context_kv: object
    main_norm: torch.Tensor
    norm: torch.Tensor
    markov_embedding: torch.Tensor
    markov_head: object
    confidence_head: torch.Tensor | None = None
    draft_head: object | None = None

    @classmethod
    def load(cls, loader, *, confidence=False, vocabulary_parallel_sampling=False, markov_head_quantization=None):
        config = loader.config.draft
        last = f"mtp.{config.layers - 1}"
        markov_name = last + ".markov_head.head.weight"
        temporary = markov_head_quantization in ("nvfp4", "fp8")
        return cls(tuple(loader.load_layer(i, draft=True) for i in range(config.layers)),
            loader.load_linear("mtp.0.main_proj"),
            loader.load_linear_group(tuple(f"mtp.{i}.attn.wkv" for i in range(config.layers))),
            loader.load_tensor("mtp.0.main_norm.weight"),
            loader.load_tensor(last + ".norm.weight"), loader.load_tensor(last + ".markov_head.embed.weight"),
            loader.load_padded_tensor(markov_name, rows=loader.shard.padded_partition(129280, alignment=64),
                                      temporary=temporary) if vocabulary_parallel_sampling
            else loader.load_tensor(markov_name, temporary=temporary),
            loader.load_tensor(last + ".confidence_head.proj.weight") if confidence else None)


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    draft: DSparkWeights | None
    allocated_bytes: int
    vision: dict | None = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, dspark=False, confidence=False,
             expert_scale_compression=None, draft_head_quantization=None, markov_head_quantization=None,
             vocabulary_parallel_sampling=False, megamoe_layouts=None, fused_attention=None):
        config = ModelConfig.load(manifest.metadata_path)
        if megamoe_layouts is not None:
            required = {f"layers.{i}.ffn" for i in range(config.layers)}
            if dspark:
                required.update(f"mtp.{i}.ffn" for i in range(config.draft.layers))
            if set(megamoe_layouts) != required:
                raise ValueError("MegaMoE preparation requires explicit layouts for every enabled target and draft layer")
            if any(not isinstance(layout, MegaMoeLayerLayout) for layout in megamoe_layouts.values()):
                raise TypeError("every MegaMoE layer requires a concrete MegaMoeLayerLayout")
        before = allocated_tensor_bytes(device)
        with DeepSeekWeightLoader(manifest, device, shard=shard, progress=progress, dspark=dspark,
                                 expert_scale_compression=expert_scale_compression,
                                 megamoe_layouts=megamoe_layouts, fused_attention=fused_attention) as loader:
            embedding, norm = loader.load_tensors(("embed.weight", "norm.weight"))
            head = loader.load_padded_tensor("head.weight", rows=shard.padded_partition(129280, alignment=64))
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = DSparkWeights.load(loader, confidence=confidence,
                vocabulary_parallel_sampling=vocabulary_parallel_sampling,
                markov_head_quantization=markov_head_quantization) if dspark else None
            if draft is not None:
                if draft_head_quantization == "nvfp4":
                    draft = replace(draft, draft_head=nvfp4_draft_head(head))
                elif draft_head_quantization == "mxfp8":
                    draft = replace(draft, draft_head=mxfp8_draft_head(head))
                if markov_head_quantization == "nvfp4":
                    draft = replace(draft, markov_head=nvfp4_draft_head(draft.markov_head))
                elif markov_head_quantization == "fp8":
                    draft = replace(draft, markov_head=fp8_markov_head(draft.markov_head))
            vision = ({name: loader.load_tensor(name) for name in loader._sources
                       if name.startswith(("vision.", "aligner.", "image_"))}
                      if config.vision is not None else None)
        return cls(config, shard, embedding, norm, head, layers, draft,
                   allocated_tensor_bytes(device) - before, vision)


class DeepSeekWeightLoader(BlockFP8WeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None, dspark=False,
                 expert_scale_compression=None, megamoe_layouts=None, fused_attention=None):
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("DeepSeek V4.1 expert scale compression must be None or 'csf'")
        self.dspark = dspark
        self.expert_scale_compression = expert_scale_compression
        if fused_attention is not None and (not isinstance(fused_attention, FusedMLAPreparation)
                or shard.world_size != 1 or len(fused_attention.cos_sin) != 2):
            raise ValueError("fused MLA preparation requires TP1 and both ordinary and compressed rotary tables")
        self.fused_attention = fused_attention
        self.megamoe_layouts = {} if megamoe_layouts is None else dict(megamoe_layouts)
        if any(not isinstance(layout, MegaMoeLayerLayout) for layout in self.megamoe_layouts.values()):
            raise TypeError("every MegaMoE layer requires a concrete MegaMoeLayerLayout")
        if self.megamoe_layouts and shard.world_size != 1:
            raise ValueError("MegaMoE expert layouts require unsharded TP1 weights")
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config = ModelConfig.load(manifest.metadata_path)
        self.shard = shard
        self._expert_scale_scratch = {}

    def load_experts(self, prefixes, interval):
        text = self.config.text
        return self.load_mxfp4_experts(prefixes, hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"], intermediate_range=interval,
            swiglu_limit=text["swiglu_limit"], scale_name="scale", projection_names=("w3", "w1", "w2"),
            scale_compression=self.expert_scale_compression, scale_scratch=self._expert_scale_scratch)

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if (name.startswith("layers.") or (self.dspark and name.startswith("mtp."))
                    or name.startswith(("vision.", "aligner.", "image_"))
                    or name in ("embed.weight", "norm.weight", "head.weight"))
                and ".engram.embed." not in name}

    def load_layer(self, layer, *, draft=False):
        prefix = f"mtp.{layer}" if draft else f"layers.{layer}"
        result = self.load_attention(layer, draft=draft)
        shard = self.shard
        routed = self.load_ffn(layer, draft=draft)
        residual = {name: self.load_tensor(prefix + "." + name) for name in (
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            "attn_norm.weight", "ffn_norm.weight")}
        engram = None
        if not draft and layer in self.config.text["engram_layer_ids"]:
            engram = {"projection": self.load_linear_columns(prefix + ".engram.wkv",
                ((head * 256, (head + 1) * 256) for head in range(shard.rank, 24, shard.world_size))),
                "weight": persistent_copy(self.load_tensor(prefix + ".engram.q_weight", dtype=torch.float32, temporary=True)
                          * self.load_tensor(prefix + ".engram.k_weight", dtype=torch.float32, temporary=True))}
        return LayerWeights(result, routed, residual, engram)

    def load_attention(self, layer, *, draft=False):
        prefix = f"mtp.{layer}" if draft else f"layers.{layer}"
        text, shard = self.config.text, self.shard
        attn = prefix + ".attn"
        first, last = shard.padded_partition(8)
        heads = first * 8 * 512, last * 8 * 512
        groups = first * 1024, last * 1024
        result = {
            "qkv_a": self.load_linear_group((attn + ".wq_a", attn + ".wkv")),
            "q_norm": self.load_tensor(attn + ".q_norm.weight"),
            "kv_norm": self.load_tensor(attn + ".kv_norm.weight"),
            "sink": self.load_padded_tensor(attn + ".attn_sink", rows=(first * 8, last * 8)),
        }
        if self.fused_attention is None:
            result["q_b"] = self.load_linear_group((attn + ".wq_b",), row_ranges=(heads,), pad=True)
            result["output"] = self.load_attention_output(attn, groups, pad=True)
        else:
            from b12x.gemm.blockscaled import pack_weight
            from ignition.preparation.persistent import persistent_empty

            q, qs = self._dense_blocks(attn + ".wq_b")
            a, sa = self._dense_blocks(attn + ".wo_a")
            self._session.flush()
            qs = qs.repeat_interleave(self.block_size, 0).repeat_interleave(self.block_size // 32, 1)
            q, qs = query_planes(q, qs)
            result["q_b"] = pack_weight(q, qs, recipe="mxfp8", allocate=persistent_empty)
            sa = sa.repeat_interleave(self.block_size, 0).repeat_interleave(self.block_size // 32, 1)
            grouped = pack_output_planes(*output_planes(a.view(8, 1024, 4096), sa.view(8, 1024, 128)))
            result["output"] = FusedMLAOutputWeights(grouped, self.load_linear(attn + ".wo_b"))
            result["fused_layout"] = self.fused_attention
        if not draft and layer in text["kv_source_layer_ids"]:
            ratio = text["compress_ratios"][layer]
            result["compressor"] = {
                "values": self.load_tensor(attn + ".compressor.wkv.weight"),
                "norm": self.load_tensor(attn + ".compressor.norm.weight", dtype=torch.float32),
            }
            if ratio == 2:
                result["compressor"]["gates"] = self.load_tensor(attn + ".compressor.wgate.weight")
            result["index_k"] = self.load_tensor(attn + ".indexer.wk.weight")
            result["index_norm"] = self.load_tensor(attn + ".indexer.k_norm.weight")
        if not draft and layer in text["index_source_layer_ids"]:
            # Replicated index heads avoid a context-sized TP score reduction.
            result["index_q"] = self.load_linear(attn + ".indexer.wq_b")
            result["index_weights"] = self.load_tensor(attn + ".indexer.weights_proj.weight")
        return result

    def load_ffn(self, layer, *, draft=False):
        prefix = f"mtp.{layer}" if draft else f"layers.{layer}"
        ffn = prefix + ".ffn"
        text, shard = self.config.text, self.shard
        interval = shard.partition(text["moe_intermediate_size"], alignment=32)
        prefixes = [f"{ffn}.experts.{i}" for i in range(
            self.config.draft.experts if draft else text["n_routed_experts"])]
        if self.megamoe_layouts and ffn not in self.megamoe_layouts:
            raise ValueError(f"MegaMoE preparation has no explicit layout for {ffn}")
        layout = self.megamoe_layouts.get(ffn)
        if layout is None:
            experts = self.load_experts(prefixes, interval)
            shared = MLPWeights(self.load_linear_group((ffn + ".shared_experts.w1", ffn + ".shared_experts.w3"),
                row_ranges=(interval, interval)), self.load_linear(ffn + ".shared_experts.w2", columns=interval))
        else:
            g = layout.workspace.geometry
            if ((g.hidden, g.intermediate, g.top_k, g.shared_experts) !=
                    (text["hidden_size"], text["moe_intermediate_size"],
                     self.config.draft.top_k if draft else text["num_experts_per_tok"], text["n_shared_experts"])):
                raise ValueError("MegaMoE layout does not match the checkpoint's expert geometry")
            experts = layout.load(self, prefixes, shared_prefix=ffn + ".shared_experts",
                block_size=tuple(self.config.quantization["weight_block_size"]),
                projection_names=("w1", "w3", "w2"), scale_name="scale", name=ffn)
            shared = None
        return ExpertWeights(self.load_tensor(ffn + ".gate.weight"),
            self.load_tensor(ffn + ".gate.bias"), experts, shared,
            self.load_tensor(ffn + ".gate.bias_vl") if self.config.vision is not None else None)
