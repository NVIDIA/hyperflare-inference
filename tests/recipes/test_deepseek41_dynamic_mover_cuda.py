# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 hardware qualification for the inline on-GPU routing predictor + 8-SM 1D-TMA expert mover."""

import ctypes as C
import json
import os
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import pytest
from tests.components.conftest import megamoe_core_storage  # noqa: F401
import torch

from tests.components.test_sm103_gpu_expert_mover_cuda import (
    _GreenCtxPair,
    _make_random_mxfp4_weights,
    _make_random_shared_fp8,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned SM103 CUDA device (FLARE_RUN_CUDA_TESTS=1)",
)


@pytest.fixture(scope="module")
def sm103_pred_storage(megamoe_core_storage):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 GB300 required")
    return megamoe_core_storage


def test_reference_expert_preparation_load_dynamic_mover_pinned_and_row_order(
    sm103_pred_storage, tmp_path, monkeypatch
):
    """Exercise ReferenceExpertPreparation.load(..., dynamic_mover=...) with pinned HBM-only, overlapped dynamic, and cold host experts."""
    from b12x.preparation import PreparationSession
    from safetensors.torch import save_file

    from ignition.components.expert_placement import ExpertPlacement
    from ignition.components.sm103_expert_mover import query_green_context_compute_sms
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.deepseek41.config import ModelConfig
    from ignition.recipes.deepseek41.reference_model import ReferenceExpertPreparation
    from ignition.recipes.deepseek41.sm103 import DynamicExpertMoverConfig

    device = torch.device("cuda", 0)
    worker_sms, compute_sms = query_green_context_compute_sms(0, 8)
    E, HIDDEN, INTER, TOP_K, ROWS = 8, 1024, 512, 2, 16

    # Write a real miniature indexed safetensors checkpoint with distinct per-canonical-expert byte tags.
    ckpt_dir = tmp_path / "mini_dynamic_ckpt"
    ckpt_dir.mkdir()
    tensors = {}
    for layer_idx in range(2):
        name = f"layers.{layer_idx}.ffn"
        for canonical in range(E):
            tag = 0x10 + layer_idx * 16 + canonical
            tensors[f"{name}.experts.{canonical}.w1.weight"] = torch.full((INTER, HIDDEN // 2), tag, dtype=torch.uint8)
            tensors[f"{name}.experts.{canonical}.w1.scale"] = torch.full((INTER, HIDDEN // 32), 120 + canonical, dtype=torch.uint8)
            tensors[f"{name}.experts.{canonical}.w3.weight"] = torch.full((INTER, HIDDEN // 2), tag + 0x20, dtype=torch.uint8)
            tensors[f"{name}.experts.{canonical}.w3.scale"] = torch.full((INTER, HIDDEN // 32), 120 + canonical, dtype=torch.uint8)
            tensors[f"{name}.experts.{canonical}.w2.weight"] = torch.full((HIDDEN, INTER // 2), tag, dtype=torch.uint8)
            tensors[f"{name}.experts.{canonical}.w2.scale"] = torch.full((HIDDEN, INTER // 32), 120 + canonical, dtype=torch.uint8)
        tensors[f"{name}.shared_experts.w1.weight"] = torch.zeros((INTER, HIDDEN), dtype=torch.float8_e4m3fn)
        tensors[f"{name}.shared_experts.w1.scale"] = torch.full((INTER // 32, HIDDEN // 32), 127, dtype=torch.uint8)
        tensors[f"{name}.shared_experts.w3.weight"] = torch.zeros((INTER, HIDDEN), dtype=torch.float8_e4m3fn)
        tensors[f"{name}.shared_experts.w3.scale"] = torch.full((INTER // 32, HIDDEN // 32), 127, dtype=torch.uint8)
        tensors[f"{name}.shared_experts.w2.weight"] = torch.zeros((HIDDEN, INTER), dtype=torch.float8_e4m3fn)
        tensors[f"{name}.shared_experts.w2.scale"] = torch.full((HIDDEN // 32, INTER // 32), 127, dtype=torch.uint8)
        tensors[f"{name}.gate.weight"] = torch.zeros((E, HIDDEN), dtype=torch.bfloat16)
        tensors[f"{name}.gate.bias"] = torch.zeros((E,), dtype=torch.float32)

    save_file(tensors, str(ckpt_dir / "weights.safetensors"))
    (ckpt_dir / "config.json").write_text("{}")
    (ckpt_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": {key: "weights.safetensors" for key in tensors},
    }))

    mini_config = ModelConfig(
        text={
            "num_hidden_layers": 2,
            "n_routed_experts": E,
            "num_experts_per_tok": TOP_K,
            "hidden_size": HIDDEN,
            "moe_intermediate_size": INTER,
            "n_shared_experts": 1,
            "routed_scaling_factor": 1.5,
            "scoring_func": "sqrtsoftplus",
            "swiglu_limit": 7.0,
        },
        quantization={"weight_block_size": [32, 32]},
        vision=None,
    )
    monkeypatch.setattr(ModelConfig, "load", classmethod(lambda cls, path: mini_config))
    checkpoint = SimpleNamespace(source=inspect_indexed_checkpoint(ckpt_dir))

    # Layer 0 mixes 3 pinned HBM-only (3, 1, 6), 2 overlapped dynamic HBM (7, 0), and 3 cold host-only (5, 2, 4).
    # Layer 1 is all-HBM and must remain non-dynamic.
    placements = {
        "layers.0.ffn": ExpertPlacement(experts=E, host=(5, 2, 7, 0, 4), hbm=(3, 7, 1, 0, 6)),
        "layers.1.ffn": ExpertPlacement(experts=E, host=(), hbm=(2, 0, 5, 1, 7, 3, 6, 4)),
    }
    mover_cfg = DynamicExpertMoverConfig(
        worker_sm_count=worker_sms,
        spare_slots=2,
        queue_capacity=64,
        decay_shift=2,
        promote_threshold=256,
        hysteresis=0,
        cooldown_steps=1,
        max_enqueues=2,
        max_predict_rows=64,
    )

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    experts = None
    try:
        with PreparationSession(device=device, autotune=False) as session:
            experts = ReferenceExpertPreparation.load(
                checkpoint,
                placements,
                target_rows=(ROWS,),
                storage=sm103_pred_storage,
                kernel_cache=kernel_cache,
                session=session,
                host_lane_clusters=0,
                dspark=False,
                dynamic_mover=mover_cfg,
                compute_sm_count=compute_sms,
            )

        assert experts.mover is not None
        assert len(experts.mover.layers) == 1, "only tiered target layers should register with the dynamic mover"

        w0 = experts.weights["layers.0.ffn"]
        w1 = experts.weights["layers.1.ffn"]
        assert w0.host_tier is not None and w1.host_tier is None
        tier0 = w0.host_tier

        # Internal row order: movable host experts (5, 2, 7, 0, 4) at 0..4; pinned HBM-only (3, 1, 6) at 5..7.
        expected_canonical_to_internal = [3, 6, 1, 5, 4, 0, 7, 2]
        assert w0.expert_map.cpu().tolist() == expected_canonical_to_internal
        assert experts.modules["layers.0.ffn"].placement.data_ptr() == w0.expert_map.data_ptr()
        assert w0.host_experts == (5, 2, 7, 0, 4)
        assert w0.hbm_experts == (3, 1, 6, 7, 0)
        assert tier0.rows == 5, "pinned HBM-only experts must not allocate host rows"
        assert tier0.hbm_slots == 7, "HBM slots must equal len(hbm) + spare_slots"

        # Initial live_slot_map (indexed by internal ID) and slot_owner (indexed by HBM slot).
        # Slots 0..2 hold pinned internal experts (5, 6, 7); slots 3..4 hold dynamic (2, 3); slots 5..6 are spare (-1).
        assert tier0.live_slot_map.cpu().tolist() == [-1, -1, 3, 4, -1, 0, 1, 2]
        assert tier0.slot_owner.cpu().tolist() == [5, 6, 7, 2, 3, -1, -1]

        # Verify actual packed down-projection bytes landed in the exact host rows, pinned slots, dynamic slots, and zeroed spares.
        host_down = tier0.host_views[2].view(torch.uint8)
        for host_row, canonical in enumerate((5, 2, 7, 0, 4)):
            assert (host_down[host_row] == (0x10 + canonical)).all(), (
                f"host row {host_row} did not receive canonical expert {canonical}"
            )
        hbm_down = w0.routed.down[0].cpu().view(torch.uint8)
        for slot, canonical in enumerate((3, 1, 6, 7, 0)):
            assert (hbm_down[slot] == (0x10 + canonical)).all(), (
                f"HBM slot {slot} did not receive canonical expert {canonical}"
            )
        assert (hbm_down[5:] == 0).all(), "spare HBM slots must be zero-initialized"
        assert (w0.routed.gate_up[0].cpu()[5:] == 0).all(), "spare HBM gate_up slots must be zero-initialized"

        # Verify descriptor and predictor metadata encode num_host_experts == 5 so pinned [5..7] are never evictable.
        desc_fields = struct.unpack_from("<4Q4Q4Q3Q4I", bytes(experts.mover.layer_descs[:136].cpu().tolist()), 0)
        desc_num_experts, desc_hbm_slots, _, desc_num_host = desc_fields[15:19]
        assert (desc_num_experts, desc_hbm_slots, desc_num_host) == (E, 7, 5)

        (binding0,) = experts.mover.predictor_bindings
        assert binding0 is tier0.predictor_binding
        assert binding0.scores.shape == (5,)
        assert binding0.last_seen.shape == (5,)
        assert binding0.cooldown.shape == (5,)
        pred_fields = struct.unpack_from("<3Q9I", bytes(binding0.state[:60].cpu().tolist()), 0)
        assert pred_fields[11] == 5, f"expected PredictorLayerState.num_host_experts=5, got {pred_fields[11]}"

        # Live pinned-protection check: boost movable resident experts (2, 3) above a 1-row cold miss on internal 0.
        # Pinned experts (5, 6, 7) are excluded from [0, num_host_experts), so no eviction may enqueue.
        experts.mover.reset_runtime_state()
        torch.cuda.synchronize()
        warm_resident = torch.tensor([[2, 3]] * ROWS, dtype=torch.int64, device=device)
        experts.mover.binding.launch_layer_enter_predict(
            tier0.live_slot_map, tier0.control, tier0.layer_epoch, warm_resident,
            ROWS, TOP_K, E, binding0.state, binding0.queue_header, binding0.queue_ring,
            binding0.metrics_ptr, binding0.metrics_count,
        )
        single_cold = torch.tensor([[0, 2]], dtype=torch.int64, device=device)
        experts.mover.binding.launch_layer_enter_predict(
            tier0.live_slot_map, tier0.control, tier0.layer_epoch, single_cold,
            1, TOP_K, E, binding0.state, binding0.queue_header, binding0.queue_ring,
            binding0.metrics_ptr, binding0.metrics_count,
        )
        torch.cuda.synchronize()
        _, tail_protected = struct.unpack_from("<2I", bytes(binding0.queue_header[0:8].cpu().tolist()), 0)
        assert tail_protected == 0, "predictor enqueued eviction against a pinned HBM-only expert"

        # When movable residents (2, 3) have score 0, a cold miss on internal 0 must pick a movable victim in {2, 3}, never {5, 6, 7}.
        experts.mover.reset_runtime_state()
        torch.cuda.synchronize()
        cold_batch = torch.tensor([[0, 1]] * ROWS, dtype=torch.int64, device=device)
        experts.mover.binding.launch_layer_enter_predict(
            tier0.live_slot_map, tier0.control, tier0.layer_epoch, cold_batch,
            ROWS, TOP_K, E, binding0.state, binding0.queue_header, binding0.queue_ring,
            binding0.metrics_ptr, binding0.metrics_count,
        )
        torch.cuda.synchronize()
        _, tail_enq = struct.unpack_from("<2I", bytes(binding0.queue_header[0:8].cpu().tolist()), 0)
        assert tail_enq == 1, f"expected 1 enqueued swap request, got {tail_enq}"
        req_layer, req_in, req_victim, req_seq = struct.unpack_from(
            "<IiiI", bytes(binding0.queue_ring[:16].cpu().tolist()), 0
        )
        assert req_layer == 0 and req_seq == 1
        assert req_in in (0, 1), f"unexpected promoted expert_in={req_in}"
        assert req_victim in (2, 3), f"victim_expert={req_victim} escaped movable resident set {{2, 3}}"
    finally:
        if experts is not None:
            experts.close_host_regions()


