# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mini checkpoint names and repeated-core controller weights."""

from dataclasses import replace

import torch

from ignition.preparation.persistent import persistent_copy

from .config import TEXT_PREFIX, VISION_PREFIX
from .loop import GateWeights, LayerControllers, STREAMS
from .weights import InklingWeightLoader


def checkpoint_aliases(text):
    """Map the shared Inkling loader's tensor roles to Mini's stored names."""
    aliases = {f"{TEXT_PREFIX}.embed_tokens.weight": "model.llm.embed.weight",
               "lm_head.weight": "model.llm.unembed.weight",
               f"{TEXT_PREFIX}.embed_norm.weight": "model.llm.embed_norm.weight",
               f"{TEXT_PREFIX}.norm.weight": "model.llm.norm.weight",
               f"{TEXT_PREFIX}.final_input.query": "model.llm.attn_res.output_query"}
    for i in range(text.layers):
        dst, src = f"{TEXT_PREFIX}.layers.{i}", f"model.llm.layers.{i}"
        for role, name in (("q_proj", "wq_du"), ("k_proj", "wk_dv"), ("v_proj", "wv_dv"),
                           ("o_proj", "wo_ud"), ("gate_proj", "gate_attn"), ("r_proj", "wr_du"),
                           ("sink_proj", "attn_sink_proj")):
            aliases[f"{dst}.self_attn.{role}.weight"] = f"{src}.attn.{name}.weight"
        for name in ("q_l2_clip.scale", "rel_l2_clip.scale", "rel_logits_proj.proj"):
            aliases[f"{dst}.self_attn.{name}"] = f"{src}.attn.{name}"
        for kind in ("k", "v"):
            if text.local(i):
                aliases[f"{dst}.self_attn.{kind}_sconv.conv1d.weight"] = f"{src}.attn.{kind}_sconv.weight"
            else:
                aliases[f"{dst}.self_attn.{kind}_sconv.static_weight"] = f"{src}.attn.{kind}_sconv.static_w"
                aliases[f"{dst}.self_attn.{kind}_sconv.weight_proj"] = f"{src}.attn.{kind}_sconv.weight_proj"
        for role, name in (("input_layernorm", "attn_norm"), ("q_input_layernorm", "q_input_norm"),
                           ("k_input_layernorm", "k_input_norm"), ("v_input_layernorm", "v_input_norm"),
                           ("post_attention_layernorm", "mlp_norm"),
                           ("post_attention_convolution.conv1d", "attn_sconv"),
                           ("post_mlp_convolution.conv1d", "mlp_sconv"), ("mlp.gate", "mlp.gate")):
            if role == "input_layernorm" and not text.local(i):
                continue
            if role in ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm") and text.local(i):
                continue
            aliases[f"{dst}.{role}.weight"] = f"{src}.{name}.weight"
        aliases[f"{dst}.mlp.gate.e_score_correction_bias"] = f"{src}.mlp.gate.bias"
        aliases[f"{dst}.mlp_input.query"] = f"model.llm.attn_res.mlp_queries.{i}"
        if i:
            aliases[f"{dst}.attention_input.query"] = f"model.llm.attn_res.attn_queries.{i - 1}"
    return aliases


def vision_checkpoint_aliases(names):
    aliases = {}
    for source in names:
        if source.startswith("model.visual.backbone."):
            path = source.removeprefix("model.visual.backbone.")
            for raw, canonical in (("ln_pre.", "pre_layernorm."),
                                   (".ln_1.", ".layernorm_before."), (".ln_2.", ".layernorm_after."),
                                   (".ls_1.gamma", ".lambda_1"), (".ls_2.gamma", ".lambda_2"),
                                   (".mlp.c_fc.", ".mlp.fc1."), (".mlp.c_proj.", ".mlp.fc2.")):
                path = path.replace(raw, canonical)
            aliases[f"{VISION_PREFIX}.{path}"] = source
    aliases[f"{VISION_PREFIX}.merger.linear_1.weight"] = "model.visual.proj_in.weight"
    aliases[f"{VISION_PREFIX}.merger.linear_2.weight"] = "model.visual.proj_out.weight"
    return aliases


class MiniWeightLoader(InklingWeightLoader):
    def __enter__(self):
        super().__enter__()
        try:
            for alias, source in checkpoint_aliases(self.config.text).items():
                self._sources[alias] = self._sources[source]
            if self.vision:
                for alias, source in vision_checkpoint_aliases(self._sources).items():
                    self._sources[alias] = self._sources[source]
        except BaseException:
            import sys
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if self.vision or name.startswith("model.llm.")}

    def load_layer(self, index):
        layer = super().load_layer(index)
        c = self.config.text
        controllers, queries = None, None
        if c.loop is not None and c.loop.core_start <= index < c.loop.core_end:
            prefix = f"model.llm.layers.{index}.controllers"
            gates = []
            for stream in STREAMS:
                down = self.load_tensor(f"{prefix}.{stream}.down_weight")
                gain = self.load_tensor(f"{prefix}.{stream}.gain_weight", temporary=True)
                forget = self.load_tensor(f"{prefix}.{stream}.forget_weight", temporary=True)
                gates.append(GateWeights(
                    down=down,
                    forget_gain=persistent_copy(torch.cat((forget, gain), dim=0).contiguous()),
                    forget_t=persistent_copy(forget.t().contiguous()),
                    gain_t=persistent_copy(gain.t().contiguous()),
                ))
            controllers = LayerControllers(*gates)
            start = 2 * (index - c.loop.core_start)
            q = self.load_tensor("model.llm.middle_cycle_cross_pass_queries", extent=slice(start, start + 2), temporary=True)
            kv = self.load_tensor("model.llm.middle_cycle_cross_pass_kv_queries", extent=slice(start, start + 2), temporary=True)
            queries = persistent_copy(torch.stack((q[0], kv[0], kv[1], q[1])).float() * c.hidden ** -0.5)
        return replace(layer, controllers=controllers, message_queries=queries)
