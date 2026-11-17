# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit TP1 expert placement and reference attention preparation on SM103."""

from dataclasses import dataclass
from pathlib import Path

from ignition.components.expert_placement import ExpertPlacement


@dataclass(frozen=True)
class SM103Config:
    placements: tuple[tuple[str, ExpertPlacement], ...]
    flashmla_library: Path
    deepselect_library: Path
    host_lane_clusters: int = 4

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
        return dict(placements={name: placement.to_dict() for name, placement in self.placements},
                    flashmla_library=str(self.flashmla_library), deepselect_library=str(self.deepselect_library),
                    host_lane_clusters=self.host_lane_clusters)

    def expert_bytes(self, model, *, dspark):
        """Native packed MXFP4 payload and expanded UE8M0 scales, including copies."""
        self.validate_model(model, dspark=dspark)
        elements = 3 * model.text["hidden_size"] * model.text["moe_intermediate_size"]
        per_expert = elements // 2 + elements // 32
        return {tier: sum(len(getattr(placement, tier)) for _, placement in self.placements) * per_expert
                for tier in ("hbm", "host")}

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
