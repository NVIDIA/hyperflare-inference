# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit TP1 expert placement and reference attention preparation on SM103."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

from ignition.components.expert_placement import ExpertPlacement


@dataclass(frozen=True)
class SM103Config:
    placements: tuple[tuple[str, ExpertPlacement], ...]
    flashmla_library: Path
    deepselect_library: Path
    host_lane_clusters: int = 4

    @classmethod
    def load(cls, path):
        path = Path(path).resolve(strict=True)
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate SM103 preparation field: {key}")
                result[key] = value
            return result
        value = json.loads(path.read_text(), object_pairs_hook=unique)
        required = {"version", "placements", "flashmla_library", "deepselect_library", "host_lane_clusters"}
        if (not isinstance(value, dict) or set(value) != required
                or type(value["version"]) is not int or value["version"] != 1
                or not isinstance(value["placements"], dict)):
            raise ValueError("SM103 preparation requires version 1, placements, both library paths and host_lane_clusters")
        placements = []
        for name, rows in value["placements"].items():
            if (not isinstance(rows, dict) or set(rows) != {"experts", "host", "hbm"}
                    or not isinstance(rows["host"], list) or not isinstance(rows["hbm"], list)):
                raise ValueError(f"{name}: placement requires experts, host and hbm arrays")
            placements.append((name, ExpertPlacement(rows["experts"], tuple(rows["host"]), tuple(rows["hbm"]))))
        libraries = []
        for name in ("flashmla_library", "deepselect_library"):
            if not isinstance(value[name], str) or not value[name]:
                raise ValueError(f"{name} must name a prepared shared library")
            libraries.append(path.parent / value[name])
        return cls(tuple(placements), *libraries, value["host_lane_clusters"])

    def __post_init__(self):
        if (not isinstance(self.placements, tuple) or not self.placements
                or any(not isinstance(item, tuple) or len(item) != 2
                       or not isinstance(item[0], str) or not isinstance(item[1], ExpertPlacement)
                       for item in self.placements)
                or len(dict(self.placements)) != len(self.placements)):
            raise ValueError("SM103 preparation requires distinct named expert placements")
        if type(self.host_lane_clusters) is not int or not 0 <= self.host_lane_clusters < 76:
            raise ValueError("SM103 host lane clusters must be in [0, 75]")
        for name in ("flashmla_library", "deepselect_library"):
            path = Path(getattr(self, name)).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{name} must name a prepared shared library")
            object.__setattr__(self, name, path)

    def validate_model(self, model, *, dspark):
        expected = {f"layers.{i}.ffn": model.text["n_routed_experts"] for i in range(model.layers)}
        if dspark:
            expected.update({f"mtp.{i}.ffn": model.draft.experts for i in range(model.draft.layers)})
        if {name: placement.experts for name, placement in self.placements} != expected:
            raise ValueError("SM103 placements must match every enabled target and draft layer's canonical experts")

    def to_dict(self):
        return dict(version=1, placements={name: placement.to_dict() for name, placement in self.placements},
                    flashmla_library=str(self.flashmla_library), deepselect_library=str(self.deepselect_library),
                    host_lane_clusters=self.host_lane_clusters)

    def expert_bytes(self, model, *, dspark):
        """Native packed MXFP4 payload and expanded UE8M0 scales, including copies."""
        self.validate_model(model, dspark=dspark)
        elements = 3 * model.text["hidden_size"] * model.text["moe_intermediate_size"]
        per_expert = elements // 2 + elements // 32
        return {tier: sum(len(getattr(placement, tier)) for _, placement in self.placements) * per_expert
                for tier in ("hbm", "host")}

    def planning_inputs(self, recipe, checkpoint, snapshot, *, dspark):
        """Estimate native tensor extents from headers, independently of CSF file sizes."""
        from ignition.preparation.checkpoint import inspect_indexed_checkpoint
        from ignition.recipes.declaration import RankCosts
        from .config import ModelConfig

        if recipe.name != "deepseek41":
            raise ValueError("SM103 reference preparation is integrated with DeepSeek V4.1")
        model = ModelConfig.load(snapshot)
        tier = self.expert_bytes(model, dspark=dspark)
        manifest = inspect_indexed_checkpoint(snapshot)
        other = draft = 0
        for name, tensor in manifest.tensors.items():
            if ".ffn.experts." in name or ".engram.embed." in name:
                continue
            is_draft = name.startswith("mtp.")
            if is_draft and not dspark:
                continue
            # Native 1x32 scale planes replace block scales for eligible FP8
            # projections. Retaining the source scale bytes makes this an upper
            # estimate, including projections that retain their block format.
            size = tensor.nbytes + (tensor.nbytes // 32 if tensor.dtype == "F8_E4M3" else 0)
            size = (size + 511) // 512 * 512
            other += size
            if is_draft:
                draft += size
        if dspark:
            # Draft context uses another grouped copy of each layer's KV weight.
            context = sum(t.nbytes + (t.nbytes // 32 if t.dtype == "F8_E4M3" else 0)
                for name, t in manifest.tensors.items() if name.startswith("mtp.") and ".attn.wkv." in name)
            other += context
            draft += context
            per_expert = 3 * model.text["hidden_size"] * model.text["moe_intermediate_size"] * 17 // 32
            draft += sum(len(p.hbm) for name, p in self.placements if name.startswith("mtp.")) * per_expert
        # Physical row/page costs are checked against ContinuationPool in tests.
        # Graph storage is an estimate; measured capture accounting supersedes it.
        ranks = {1: RankCosts(3_045_888, 114_688, 8 << 30)}
        plain = {1: RankCosts(2_843_136, 114_688, 8 << 30)}
        return (replace(recipe, ranks=ranks, plain_ranks=plain, expert_scale_compression=None),
                replace(checkpoint, device_bytes=tier["hbm"] + other, draft_bytes=draft,
                    mapped_host_bytes=tier["host"], expansion_bytes=0,
                    scale_compressed_device_bytes={}, ranks=ranks, plain_ranks=plain))

    def prepare(self, model, config, *, device, storage):
        import torch
        from ignition.components.fused_mla import FusedMLAPreparation
        from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
        from ignition.components.megamoe_tiers import MegaMoeLayerLayout
        from ignition.components.rotary import CompressedRotary

        self.validate_model(model, dspark=config.dspark)
        if torch.cuda.get_device_capability(device) != (10, 3):
            raise ValueError("reference TP1 expert and attention preparation requires SM103")
        workspaces, layouts = {}, {}
        for name, placement in self.placements:
            draft = name.startswith("mtp.")
            capacity = (max(config.request_capacity * config.draft_tokens, 1) if draft else
                        max(config.prefill_chunk_size, config.decode_tokens))
            geometry = MegaMoeGeometry(placement.physical_rows, capacity,
                model.draft.top_k if draft else model.text["num_experts_per_tok"],
                model.text["hidden_size"], model.text["moe_intermediate_size"],
                model.text["n_shared_experts"])
            if geometry not in workspaces:
                workspaces[geometry] = MegaMoeWorkspace(geometry, device=device, kernel_cache=config.kernel_cache)
            layouts[name] = MegaMoeLayerLayout(workspaces[geometry], placement.host, placement.hbm,
                                              storage, self.host_lane_clusters)
        rotary = CompressedRotary(model, device)
        attention = FusedMLAPreparation.create((rotary.frequencies[False], rotary.frequencies[True]),
            config.context, kernel_cache=config.kernel_cache, library=self.flashmla_library,
            deepselect_library=self.deepselect_library)
        return layouts, attention
