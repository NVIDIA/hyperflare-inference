# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling2 image encoding with canonical blocks and prepared BF16 operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ignition.components.linear import UnquantizedProjection
from ignition.components.rotary import apply_cached_rotary, prepare_cached_rotary
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.persistent import persistent_cat, persistent_copy


@dataclass(frozen=True)
class VisionMetadata:
    position_ids: torch.Tensor
    interpolation_indices: torch.Tensor
    interpolation_weights: torch.Tensor
    cu_seqlens: torch.Tensor
    num_patches: torch.Tensor
    num_features: torch.Tensor

    @classmethod
    def allocate(cls, patches, images, device):
        if patches <= 0 or patches % 4 or images != 1:
            raise ValueError("Inkling2 vision requires one image and a positive patch capacity divisible by four")
        return cls(torch.zeros((patches + 1, 2), dtype=torch.int64, device=device),
                   torch.zeros((patches, 4), dtype=torch.int64, device=device),
                   torch.zeros((patches, 4), dtype=torch.float32, device=device),
                   torch.zeros(2, dtype=torch.int32, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device))


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    patch_hidden: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    attention: torch.Tensor
    projected: torch.Tensor
    intermediate: torch.Tensor
    merged: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device):
        if patches <= 0 or patches % 4:
            raise ValueError("Inkling2 vision patch capacity must be positive and divisible by four")
        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        rows = patches + 1
        return cls(zeros(patches, 1200), zeros(patches, 1024), zeros(rows, 1024), zeros(rows, 3072),
                   zeros(rows, 1024), zeros(rows, 1024), zeros(rows, 4096),
                   zeros(patches // 4, 4096), zeros(patches // 4, 4608))

    @property
    def query(self):
        return self.qkv.view(-1, 3, 16, 64)[:, 0]

    @property
    def key(self):
        return self.qkv.view(-1, 3, 16, 64)[:, 1]

    @property
    def value(self):
        return self.qkv.view(-1, 3, 16, 64)[:, 2]

    def rows(self, patches):
        if patches <= 0 or patches % 4 or patches > self.patches.shape[0]:
            raise ValueError("Inkling2 vision view exceeds its patch capacity")
        patch_fields = {"patches", "patch_hidden"}
        merged_fields = {"merged", "output"}
        return type(self)(*(getattr(self, name)[:patches if name in patch_fields else
                            patches // 4 if name in merged_fields else patches + 1]
                            for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class VisionWeights:
    parameters: dict[str, torch.Tensor]

    @classmethod
    def load(cls, module, loader):
        parameters = {}
        for path, parameter in module.named_parameters():
            parent, _, attribute = path.rpartition(".")
            if parent.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")):
                continue
            checkpoint_path = path.removesuffix(".weight") if path == "embeddings.position_embedding.weight" else path
            tensor = loader.load_tensor("model.vision_tower." + checkpoint_path)
            if tensor.dtype != torch.bfloat16 or tensor.shape != parameter.shape:
                raise ValueError(f"Inkling2 vision parameter {checkpoint_path} must retain its BF16 checkpoint geometry")
            parameters[path] = tensor
        for index, layer in enumerate(module.layers):
            prefix = f"layers.{index}.self_attn"
            for attribute in ("weight", "bias"):
                sources = [loader.load_tensor(f"model.vision_tower.{prefix}.{role}_proj.{attribute}", temporary=True)
                           for role in ("q", "k", "v")]
                expected = getattr(layer.self_attn.q_proj, attribute)
                if any(t.dtype != torch.bfloat16 or t.shape != expected.shape for t in sources):
                    raise ValueError(f"Inkling2 {prefix} Q/K/V {attribute} must retain its BF16 checkpoint geometry")
                parameters[f"{prefix}.qkv.{attribute}"] = persistent_cat(sources)
        return cls(parameters)

    def bind(self, module):
        for path, tensor in self.parameters.items():
            parent, _, attribute = path.rpartition(".")
            if parent.endswith(".self_attn.qkv"):
                attention = module.get_submodule(parent.removesuffix(".qkv"))
                for role, part in zip(("q", "k", "v"), tensor.chunk(3, dim=0), strict=True):
                    setattr(getattr(attention, f"{role}_proj"), attribute,
                            torch.nn.Parameter(part, requires_grad=False))
            else:
                setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(tensor, requires_grad=False))


class _Linear(torch.nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, source):
        return self.projection(source)


class _InplaceGELU(torch.nn.Module):
    def forward(self, source):
        return torch.ops.aten.gelu.out(source, approximate="none", out=source)


class _Attention(torch.nn.Module):
    def __init__(self, module, scratch, metadata, packed, rotary_rows, *, session, name, shared=None,
                 attention_workspace=None):
        super().__init__()
        self.rotary_rows = rotary_rows
        self.qkv = UnquantizedProjection(scratch.hidden, packed[0], scratch.qkv,
            session=session, name=f"{name}.qkv", bias=packed[1])
        self.out_proj = UnquantizedProjection(scratch.attention, module.out_proj.weight, scratch.projected,
            session=session, name=f"{name}.out_proj", bias=module.out_proj.bias)
        self.attention = shared.attention if shared is not None else PreparedVarlenAttention(
            scratch.query, scratch.key, scratch.value, metadata.cu_seqlens,
            scratch.attention.view_as(scratch.query), session=session, workspace=attention_workspace, name=name)
        self.owns_attention = shared is None

    def functional(self, *, session):
        self.qkv.functional(session=session)
        self.out_proj.functional(session=session)
        if self.owns_attention:
            self.attention.functional(session=session)

    def forward(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
        qkv = self.qkv(hidden_states).view(-1, 3, 16, 64)
        query, key, value = qkv.unbind(1)
        apply_cached_rotary(query, key, position_embeddings, self.rotary_rows, query, key, True)
        attended = self.attention(query, key, value, cu_seqlens)
        return self.out_proj(attended.view(-1, 1024)), None


class PreparedVision:
    def __init__(self, module, scratch, metadata, weights, *, session, name="vision",
                 projection_workspace=None, attention_workspace=None):
        from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionRotaryEmbedding

        del projection_workspace
        expected = dict(hidden_size=1024, intermediate_size=4096, num_hidden_layers=23,
                        num_attention_heads=16, patch_size=20, num_channels=3,
                        spatial_merge_size=2, posemb_grid_size=32,
                        projection_hidden_dim=4096, text_hidden_size=4608, hidden_act="gelu")
        if any(getattr(module.config, key) != value for key, value in expected.items()):
            raise ValueError("Inkling2 vision geometry differs from the checkpoint contract")
        capacity = scratch.patches.shape[0]
        if metadata.position_ids.shape != (capacity + 1, 2) or metadata.interpolation_indices.shape != (capacity, 4):
            raise ValueError("Inkling2 vision metadata and scratch capacities differ")
        if metadata.num_patches.item() or metadata.num_features.item() or bool(metadata.cu_seqlens.any().item()):
            raise ValueError("Inkling2 vision preparation requires inactive metadata")
        self.module, self.scratch, self.metadata, self.capacity = module, scratch, metadata, capacity
        weights.bind(module)
        module.rotary_emb = Inkling2VisionRotaryEmbedding(module.config).to(scratch.hidden.device)
        rotary_rows = persistent_copy(torch.arange(capacity + 1, dtype=torch.int32, device=scratch.hidden.device))
        prepare_cached_rotary(session)
        self.patch_embed = _Linear(UnquantizedProjection(scratch.patches,
            module.embeddings.patch_embedding.weight, scratch.patch_hidden,
            session=session, name=f"{name}.patch_embed"))
        for index, layer in enumerate(module.layers):
            prefix = f"{name}.layers.{index}"
            packed = tuple(weights.parameters[f"layers.{index}.self_attn.qkv.{key}"] for key in ("weight", "bias"))
            layer.self_attn = _Attention(layer.self_attn, scratch, metadata, packed, rotary_rows,
                session=session, name=f"{prefix}.self_attn", shared=module.layers[0].self_attn if index else None,
                attention_workspace=attention_workspace)
            for key, source, output in (("fc1", scratch.hidden, scratch.intermediate),
                                       ("fc2", scratch.intermediate, scratch.projected)):
                original = getattr(layer.mlp, key)
                setattr(layer.mlp, key, _Linear(UnquantizedProjection(source, original.weight, output,
                    session=session, name=f"{prefix}.mlp.{key}", bias=original.bias)))
            layer.mlp.activation_fn = _InplaceGELU()
        for key, source, output in (("linear_1", scratch.patch_hidden.view(-1, 4096), scratch.merged),
                                   ("linear_2", scratch.merged, scratch.output)):
            original = getattr(module.merger, key)
            setattr(module.merger, key, _Linear(UnquantizedProjection(source, original.weight, output,
                session=session, name=f"{name}.merger.{key}", bias=original.bias)))
        module.merger.act = _InplaceGELU()

    def functional(self, *, session):
        self.patch_embed.projection.functional(session=session)
        for layer in self.module.layers:
            layer.self_attn.functional(session=session)
            for key in ("fc1", "fc2"):
                getattr(layer.mlp, key).projection.functional(session=session)
        for key in ("linear_1", "linear_2"):
            getattr(self.module.merger, key).projection.functional(session=session)
        self.scratch = self.metadata = None
        return self

    def __call__(self, pixels, metadata):
        module = self.module
        patches = self.patch_embed(pixels)
        positions = (module.embeddings.position_embedding(metadata.interpolation_indices)
                     * metadata.interpolation_weights[:, :, None]).sum(1)
        patches = patches + positions.to(patches.dtype)
        cls = (module.embeddings.class_embedding + module.embeddings.cls_position_embedding).to(patches.dtype)
        hidden = module.pre_layernorm(torch.cat((cls[None], patches), dim=0))
        cosine, sine = module.rotary_emb(hidden, metadata.position_ids)
        rotary = torch.cat((cosine[:, ::2], sine[:, ::2]), dim=-1)
        for layer in module.layers:
            hidden = layer(hidden, cu_seqlens=metadata.cu_seqlens, position_embeddings=rotary)
        output = module.merger(hidden[1:].reshape(-1, 4096))
        live = torch.arange(self.capacity // 4, device=pixels.device) < metadata.num_features
        return output.masked_fill_(~live[:, None], 0)
