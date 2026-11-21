# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM sigmoid routing, NVFP4 routed experts, and the BF16 shared expert."""

import torch

from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.preparation.examples import declaration_tensor


class PreparedMoE:
    """One rank's intermediate-sharded expert contribution before all-reduce.

    Through 256 rows, an arena that permits overlap runs the shared expert on its auxiliary stream
    beside routing and the routed experts.
    """

    def __init__(self, module, source, weights, *, session, name, workspace, examples, prefill=False):
        gate, rows, hidden = module.gate, source.shape[0], source.shape[1]
        if gate.num_group != 1 or gate.topk_group != 1 or not gate.norm_topk_prob:
            raise ValueError("GLM MoE DSA routing requires one expert group and normalized route weights")
        experts, top_k = gate.num_experts, gate.top_k
        local = weights.shared.down.shape[1]

        def tensor(role, width, dtype=torch.bfloat16):
            return declaration_tensor(examples, "glm_dsa.moe." + role, (rows, width), dtype, source.device)

        logits = tensor("router_logits", experts, torch.float32)
        routes = RouterScratch(tensor("selected_logits", top_k, torch.float32), tensor("route_ids", top_k, torch.int32),
                               tensor("route_weights", top_k, torch.float32).fill_(1 / top_k))
        self.router_projection = UnquantizedProjection(source, weights.router, logits, session=session,
                                                       name=name + ".router")
        self.routing = PreparedRouter(logits, routes, session=session, name=name + ".route_topk",
            score_func="sigmoid", renormalize=True, routed_scaling_factor=gate.routed_scaling_factor,
            correction_bias=weights.correction)
        self.shared = PreparedMLP(source, weights.shared, tensor("shared_gate_up", 2 * local),
            tensor("shared_intermediate", local), tensor("shared_output", hidden), session=session,
            name=name + ".shared", workspace=workspace)
        self.experts = PreparedRoutedExperts(source, weights.experts, routes.ids, routes.weights,
            tensor("routed", hidden), session=session, name=name + ".experts", workspace=workspace,
            router_plan=self.routing.plan, a4_prefill=prefill and weights.experts_a4_prefill)
        self.concurrent_shared = None
        if rows <= 256 and workspace.overlap:
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.routing.plan, self.experts.plan), session=session, workspace=workspace,
                defer_launch=rows == 1)
        for operation in (self.router_projection, self.routing, self.shared, self.experts):
            operation.functional(session=session)
        if self.concurrent_shared is not None:
            self.concurrent_shared.functional(session=session)

    def __call__(self, source):
        logits = self.router_projection(source)
        pending = None if self.concurrent_shared is None else self.concurrent_shared.start(source)
        routes = self.routing(logits)
        routed = self.experts(source, routes.ids, routes.weights)
        shared = self.shared(source) if pending is None else self.concurrent_shared.finish(pending, routed)
        return torch.add(routed, shared)
