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


def test_sm103_inline_gpu_predictor_adapts_across_regime_shifts_bit_exact(sm103_pred_storage):
    """Verify inline on-GPU predictor promotes cold working sets across topic shifts with bit-exact MegaMoE output."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import (
        PredictorConfig,
        SM103ExpertMoverRuntime,
        query_green_context_compute_sms,
    )

    device = torch.device("cuda", 0)
    worker_sms, compute_sms = query_green_context_compute_sms(0, 8)
    gc = _GreenCtxPair(0, worker_sms)
    assert gc.worker_sms == worker_sms and gc.compute_sms == compute_sms

    E = 64
    S_ACTIVE = 16
    K_SPARE = 2
    HBM_SLOTS = S_ACTIVE + K_SPARE  # 18 slots
    HIDDEN = 1024
    INTER = 512
    TOP_K = 4
    CAPACITY = 128
    ROWS = 16
    CLAMP = 7.0

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    storage = sm103_pred_storage
    mover = None
    tier = None
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        gen = torch.Generator(device="cpu").manual_seed(20261121)
        geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
        ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        ws_dyn = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

        full_weights = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
        shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

        # Start with experts 0..15 resident in HBM; 16..63 are cold in HOST_NUMA.
        tier = MegaMoeDynamicHostTier(
            ws_dyn.geometry,
            hbm_slots=HBM_SLOTS,
            initial_hbm_experts=tuple(range(S_ACTIVE)),
            storage=storage,
            name="pred_layer0",
        )
        tier.copy_weights(
            type(full_weights)(
                (full_weights.gate_up[0].cpu(), full_weights.gate_up[1].cpu()),
                (full_weights.down[0].cpu(), full_weights.down[1].cpu()),
            )
        )
        hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
        for dst, src in zip((*hbm_pool.gate_up, *hbm_pool.down), (*full_weights.gate_up, *full_weights.down)):
            dst[:S_ACTIVE].copy_(src[:S_ACTIVE])
            dst[S_ACTIVE:].zero_()

        mover = SM103ExpertMoverRuntime(
            [(tier, hbm_pool)],
            storage=storage,
            kernel_cache=kernel_cache,
            worker_sm_count=worker_sms,
            queue_capacity=1024,
        )
        # Fast adaptation settings: 256 = 1-hit promote threshold, 4 enqueues/step, decay_shift=2 (0.75x) for rapid regime shift.
        mover.attach_predictors(
            PredictorConfig(
                decay_shift=2,
                promote_threshold=256,
                hysteresis=64,
                cooldown_steps=1,
                max_enqueues=4,
                max_predict_rows=64,
            )
        )

        with PreparationSession(device=device, autotune=False) as session:
            ex = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
            probs = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)
            ref_op = PreparedMegaMoe(
                ex, ids, probs, pad, full_weights, shared_weights,
                workspace=ws_ref, clamp=CLAMP, session=session, name="pred_ref",
                host_shared_first=True,
            )
            dyn_op = PreparedMegaMoe(
                ex, ids, probs, pad, hbm_pool, shared_weights,
                workspace=ws_dyn, clamp=CLAMP, session=session, name="pred_dyn",
                host_tier=tier, host_shared_first=True, compute_sm_count=compute_sms,
            )

        def make_regime_batches(expert_start: int, count: int):
            out = []
            for _ in range(count):
                x = (torch.randn((ROWS, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16)
                perm = torch.stack([torch.randperm(S_ACTIVE, generator=gen)[:TOP_K] + expert_start for _ in range(ROWS)])
                ids_cpu = perm.to(torch.int32).contiguous()
                probs_cpu = torch.softmax(torch.randn((ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32)
                pad_cpu = torch.zeros((ROWS,), dtype=torch.bool)
                gold = ref_op(x.to(device), ids_cpu.to(device), probs_cpu.to(device), pad_cpu.to(device)).cpu()
                out.append((x, ids_cpu, probs_cpu, pad_cpu, gold))
            torch.cuda.synchronize()
            return out

        regime_a = make_regime_batches(32, 24)  # Cold experts 32..47 (0% initial HBM hit rate)
        regime_b = make_regime_batches(48, 24)  # Abrupt topic shift to cold experts 48..63.

        # Capture compute graph (with inline predictor in sm103_layer_enter_predict_kernel) and mover graph.
        x_buf = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        ids_buf = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
        probs_buf = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
        pad_buf = torch.zeros((ROWS,), dtype=torch.bool, device=device)
        out_buf = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        torch.cuda.synchronize()

        cap_stream = torch.cuda.Stream(device=device)
        comp_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(comp_graph, pool=storage.graph_pool, stream=cap_stream):
                out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        comp_graph.instantiate()

        worker_decl = mover.capture_worker_graph(name="sm103_pred_mover", retain_graphs=True, stream=cap_stream)
        mover.reset_runtime_state()
        torch.cuda.synchronize()

        host_out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16)

        def run_step(x_cpu, ids_cpu, probs_cpu, pad_cpu, gold_cpu=None):
            gc.set_compute()
            for dst, src in ((x_buf, x_cpu), (ids_buf, ids_cpu), (probs_buf, probs_cpu), (pad_buf, pad_cpu)):
                gc._call(
                    "cuMemcpyHtoDAsync_v2",
                    C.c_uint64(dst.data_ptr()),
                    C.c_void_p(src.data_ptr()),
                    C.c_size_t(src.numel() * src.element_size()),
                    gc.c_stream,
                )
            gc.launch_graph(int(comp_graph.raw_cuda_graph_exec()), worker=False)
            gc._call(
                "cuMemcpyDtoHAsync_v2",
                C.c_void_p(host_out.data_ptr()),
                C.c_uint64(out_buf.data_ptr()),
                C.c_size_t(host_out.numel() * host_out.element_size()),
                gc.c_stream,
            )
            gc.sync_compute()
            if gold_cpu is not None:
                assert torch.equal(host_out, gold_cpu), (
                    f"inline predictor live swap mismatch: max_diff={(host_out.float() - gold_cpu.float()).abs().max().item()}"
                )

        gc.launch_graph(int(worker_decl.graph.raw_cuda_graph_exec()), worker=True)
        try:
            # --- Regime A: Cold experts 32..47 (starts at 0% HBM hit rate) ---
            for step_i, batch in enumerate(regime_a):
                run_step(*batch)
                # Give background 8-SM mover realistic inter-step decode window (~2ms) during warmup steps
                if step_i < 10:
                    time.sleep(0.002)
            m_before_tail_a = mover.read_metrics()
            for batch in regime_a[-6:]:
                run_step(*batch)
            m_after_a = mover.read_metrics()
            tail_hits_a = m_after_a.predict_hits - m_before_tail_a.predict_hits
            tail_misses_a = m_after_a.predict_misses - m_before_tail_a.predict_misses
            tail_rate_a = tail_hits_a / max(1, tail_hits_a + tail_misses_a)
            print(
                f"\n[Predictor Regime A (32..47)] tail_hit_rate={tail_rate_a*100:.1f}% "
                f"({tail_hits_a}/{tail_hits_a+tail_misses_a}), swaps={m_after_a.swaps_completed}, "
                f"enqueues={m_after_a.predict_enqueues}"
            )
            assert tail_rate_a >= 0.95, f"expected Regime A tail hit rate >= 95%, got {tail_rate_a:.3f} ({m_after_a})"

            # --- Regime B: Abrupt shift to cold experts 48..63 (drops back to 0% HBM hit rate) ---
            for step_i, batch in enumerate(regime_b):
                run_step(*batch)
                if step_i < 12:
                    time.sleep(0.002)
            m_before_tail_b = mover.read_metrics()
            for batch in regime_b[-6:]:
                run_step(*batch)
            m_after_b = mover.read_metrics()
            tail_hits_b = m_after_b.predict_hits - m_before_tail_b.predict_hits
            tail_misses_b = m_after_b.predict_misses - m_before_tail_b.predict_misses
            tail_rate_b = tail_hits_b / max(1, tail_hits_b + tail_misses_b)
            print(
                f"[Predictor Regime B (48..63)] tail_hit_rate={tail_rate_b*100:.1f}% "
                f"({tail_hits_b}/{tail_hits_b+tail_misses_b}), swaps={m_after_b.swaps_completed}, "
                f"enqueues={m_after_b.predict_enqueues}"
            )
            assert tail_rate_b >= 0.95, f"expected Regime B tail hit rate >= 95%, got {tail_rate_b:.3f} ({m_after_b})"

            # --- Principle 7: Inactive/padded rows perform ZERO predictor state mutations ---
            m_before_pad = mover.read_metrics()
            all_pad_x = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16)
            all_pad_ids = torch.zeros((ROWS, TOP_K), dtype=torch.int32)
            all_pad_probs = torch.zeros((ROWS, TOP_K), dtype=torch.float32)
            all_pad_mask = torch.ones((ROWS,), dtype=torch.bool)
            for _ in range(5):
                run_step(all_pad_x, all_pad_ids, all_pad_probs, all_pad_mask)
            m_after_pad = mover.read_metrics()
            assert m_after_pad.predict_hits == m_before_pad.predict_hits
            assert m_after_pad.predict_misses == m_before_pad.predict_misses
            assert m_after_pad.predict_enqueues == m_before_pad.predict_enqueues
        finally:
            gc.sync_compute()
            mover.signal_stop()
            gc.sync_worker()
            gc.set_primary()
            worker_decl.graph.reset()
            comp_graph.reset()

        assert int(tier.layer_epoch.item()) % 2 == 0
    finally:
        if mover is not None:
            mover.close()
        if tier is not None:
            for r in tier.regions:
                r.close()
        gc.close()


def test_sm103_inline_predictor_mixed_active_and_padded_rows_invariant(sm103_pred_storage):
    """Verify mixed active + padded rows through real MegaMoE staging and bf16_mega_gate masking leave predictor state identical to the unpadded active subset."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import (
        PredictorConfig,
        SM103ExpertMoverRuntime,
        query_green_context_compute_sms,
    )
    from ignition.preparation.persistent import persistent_copy

    device = torch.device("cuda", 0)
    worker_sms, compute_sms = query_green_context_compute_sms(0, 8)
    E, S_ACTIVE, K_SPARE = 64, 16, 2
    HBM_SLOTS = S_ACTIVE + K_SPARE
    HIDDEN, INTER, TOP_K, CAPACITY, CLAMP = 1024, 512, 4, 128, 7.0
    ACTIVE_ROWS = 8
    TOTAL_ROWS = 16

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(20261121)
    geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
    ws_a = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
    ws_b = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

    full_weights = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
    shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)
    cpu_weights = type(full_weights)(
        (full_weights.gate_up[0].cpu(), full_weights.gate_up[1].cpu()),
        (full_weights.down[0].cpu(), full_weights.down[1].cpu()),
    )

    tier_a = MegaMoeDynamicHostTier(
        ws_a.geometry, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(S_ACTIVE)),
        storage=sm103_pred_storage, name="pred_mix_a",
    )
    tier_b = MegaMoeDynamicHostTier(
        ws_b.geometry, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(S_ACTIVE)),
        storage=sm103_pred_storage, name="pred_mix_b",
    )
    tier_a.copy_weights(cpu_weights)
    tier_b.copy_weights(cpu_weights)
    hbm_a = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
    hbm_b = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
    for dst_a, dst_b, src in zip(
        (*hbm_a.gate_up, *hbm_a.down),
        (*hbm_b.gate_up, *hbm_b.down),
        (*full_weights.gate_up, *full_weights.down),
    ):
        dst_a[:S_ACTIVE].copy_(src[:S_ACTIVE])
        dst_a[S_ACTIVE:].zero_()
        dst_b[:S_ACTIVE].copy_(src[:S_ACTIVE])
        dst_b[S_ACTIVE:].zero_()

    mover_a = SM103ExpertMoverRuntime(
        [(tier_a, hbm_a)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=worker_sms,
        control_buffer_name="mix_ctrl_a",
    )
    mover_b = SM103ExpertMoverRuntime(
        [(tier_b, hbm_b)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=worker_sms,
        control_buffer_name="mix_ctrl_b",
    )
    try:
        cfg = PredictorConfig(
            decay_shift=3, promote_threshold=256, hysteresis=64, cooldown_steps=2,
            max_enqueues=2, max_predict_rows=64,
        )
        (bind_a,) = mover_a.attach_predictors(cfg)
        (bind_b,) = mover_b.attach_predictors(cfg)

        gate_w = persistent_copy((torch.randn((E, HIDDEN), generator=gen) * 0.1).to(torch.bfloat16).to(device))
        gate_b = persistent_copy(torch.linspace(-0.2, 0.2, E, dtype=torch.float32).to(device))

        with PreparationSession(device=device, autotune=False) as session:
            ex_a = torch.zeros((ACTIVE_ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            ex_ids_a = torch.zeros((ACTIVE_ROWS, TOP_K), dtype=torch.int32, device=device)
            ex_probs_a = torch.full((ACTIVE_ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            ex_pad_a = torch.zeros((ACTIVE_ROWS,), dtype=torch.bool, device=device)
            op_a = PreparedMegaMoe(
                ex_a, ex_ids_a, ex_probs_a, ex_pad_a, hbm_a, shared_weights,
                workspace=ws_a, host_tier=tier_a, host_shared_first=True,
                clamp=CLAMP, session=session, name="pred_mix_op_a",
                compute_sm_count=compute_sms,
            )
            ex_b = torch.zeros((TOTAL_ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            ex_ids_b = torch.zeros((TOTAL_ROWS, TOP_K), dtype=torch.int32, device=device)
            ex_probs_b = torch.full((TOTAL_ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            ex_pad_b = torch.zeros((TOTAL_ROWS,), dtype=torch.bool, device=device)
            op_b = PreparedMegaMoe(
                ex_b, ex_ids_b, ex_probs_b, ex_pad_b, hbm_b, shared_weights,
                workspace=ws_b, host_tier=tier_b, host_shared_first=True,
                clamp=CLAMP, session=session, name="pred_mix_op_b",
                compute_sm_count=compute_sms,
            )

        def assert_predictor_state_equal(context_label: str):
            assert torch.equal(bind_a.scores, bind_b.scores), f"{context_label}: padded rows mutated scores"
            assert torch.equal(bind_a.last_seen, bind_b.last_seen), f"{context_label}: padded rows mutated last_seen"
            assert torch.equal(bind_a.cooldown, bind_b.cooldown), f"{context_label}: padded rows mutated cooldown"
            assert torch.equal(bind_a.state[24:28], bind_b.state[24:28]), f"{context_label}: padded rows mutated step_clock"
            assert torch.equal(bind_a.queue_header[0:12], bind_b.queue_header[0:12]), (
                f"{context_label}: padded rows mutated queue head/tail"
            )
            assert torch.equal(bind_a.queue_header[48:72], bind_b.queue_header[48:72]), (
                f"{context_label}: padded rows mutated HBM hit/miss/enqueue counters"
            )
            assert torch.equal(bind_a.queue_ring, bind_b.queue_ring), f"{context_label}: padded rows mutated queue_ring"

        # 1. Real MegaMoE Triton input stager (_prepare_megamoe_inputs_kernel) with boolean padding mask:
        # first verify active HBM hits + padded cold expert IDs record zero misses and zero promotions.
        mover_a.reset_runtime_state()
        mover_b.reset_runtime_state()
        torch.cuda.synchronize()
        pad_a = torch.zeros((ACTIVE_ROWS,), dtype=torch.bool, device=device)
        pad_b = torch.zeros((TOTAL_ROWS,), dtype=torch.bool, device=device)
        pad_b[1::2] = True

        for _ in range(4):
            x_a = (torch.randn((ACTIVE_ROWS, HIDDEN), generator=gen) * 0.25).to(torch.bfloat16).to(device)
            ids_a = torch.stack([torch.randperm(S_ACTIVE, generator=gen)[:TOP_K] for _ in range(ACTIVE_ROWS)]).to(torch.int32).to(device)
            probs_a = torch.softmax(torch.randn((ACTIVE_ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)

            x_b = (torch.randn((TOTAL_ROWS, HIDDEN), generator=gen) * 0.25).to(torch.bfloat16).to(device)
            x_b[0::2] = x_a
            # Populate padded rows with valid cold host expert IDs (>= S_ACTIVE) so only the stager's padding mask neutralizes them.
            cold_ids = torch.stack([torch.randperm(E - S_ACTIVE, generator=gen)[:TOP_K] + S_ACTIVE for _ in range(ACTIVE_ROWS)]).to(torch.int32).to(device)
            ids_b = torch.empty((TOTAL_ROWS, TOP_K), dtype=torch.int32, device=device)
            ids_b[0::2] = ids_a
            ids_b[1::2] = cold_ids
            probs_b = torch.full((TOTAL_ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            probs_b[0::2] = probs_a

            op_a(x_a, ids_a, probs_a, pad_a)
            op_b(x_b, ids_b, probs_b, pad_b)
        torch.cuda.synchronize()

        staged_ids_b = ws_b.views[2][:TOTAL_ROWS]
        assert (staged_ids_b[pad_b] == -1).all(), "MegaMoE stager did not mask padded rows to -1"
        hits_b, misses_b, enq_b = struct.unpack("<3Q", bytes(bind_b.queue_header[48:72].cpu().tolist()))
        assert hits_b == 4 * ACTIVE_ROWS * TOP_K
        assert misses_b == 0, f"padded cold rows incremented miss counter: {misses_b}"
        assert enq_b == 0, f"padded cold rows triggered promotions: {enq_b}"
        assert int(bind_b.scores[S_ACTIVE:].sum().item()) == 0, "padded cold rows incremented cold expert scores"
        assert_predictor_state_equal("stager_all_hbm_active")

        # Next drive mixed hits/misses on active rows interleaved with padded cold rows through the real stager.
        for _ in range(8):
            x_a = (torch.randn((ACTIVE_ROWS, HIDDEN), generator=gen) * 0.25).to(torch.bfloat16).to(device)
            ids_a = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(ACTIVE_ROWS)]).to(torch.int32).to(device)
            probs_a = torch.softmax(torch.randn((ACTIVE_ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)

            x_b = (torch.randn((TOTAL_ROWS, HIDDEN), generator=gen) * 0.25).to(torch.bfloat16).to(device)
            x_b[0::2] = x_a
            cold_ids = torch.stack([torch.randperm(E - S_ACTIVE, generator=gen)[:TOP_K] + S_ACTIVE for _ in range(ACTIVE_ROWS)]).to(torch.int32).to(device)
            ids_b = torch.empty((TOTAL_ROWS, TOP_K), dtype=torch.int32, device=device)
            ids_b[0::2] = ids_a
            ids_b[1::2] = cold_ids
            probs_b = torch.full((TOTAL_ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            probs_b[0::2] = probs_a

            op_a(x_a, ids_a, probs_a, pad_a)
            op_b(x_b, ids_b, probs_b, pad_b)
        torch.cuda.synchronize()

        assert (ws_b.views[2][:TOTAL_ROWS][pad_b] == -1).all(), "MegaMoE stager did not mask padded rows to -1"
        assert_predictor_state_equal("stager_mixed_active_and_padded")
        _, mix_misses, mix_enq = struct.unpack("<3Q", bytes(bind_a.queue_header[48:72].cpu().tolist()))
        assert int(bind_a.scores.sum().item()) > 0 and mix_misses > 0 and mix_enq > 0

        # 2. Encoded path with DeepGEMM bf16_mega_gate + valid_token_mask -> op.encoded:
        mover_a.reset_runtime_state()
        mover_b.reset_runtime_state()
        torch.cuda.synchronize()
        valid_a = torch.ones((ACTIVE_ROWS,), dtype=torch.bool, device=device)
        valid_b = torch.zeros((TOTAL_ROWS,), dtype=torch.bool, device=device)
        valid_b[0::2] = True
        cold_probe = gate_w[32:32 + ACTIVE_ROWS].clone() * 4.0
        expert_map = persistent_copy(torch.arange(E, dtype=torch.int32, device=device))

        for _ in range(6):
            bf16_a = (torch.randn((ACTIVE_ROWS, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16).to(device)
            bf16_b = torch.empty((TOTAL_ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            bf16_b[0::2] = bf16_a
            # Strongly drive padded rows toward cold experts so any mask leak mutates cold scores.
            bf16_b[1::2] = cold_probe

            xa_fp8, xa_sf, ids_out_a, probs_out_a, _, _ = op_a.encoded_inputs()
            xb_fp8, xb_sf, ids_out_b, probs_out_b, _, _ = op_b.encoded_inputs()
            xa_fp8.zero_()
            xa_sf.fill_(0x7F7F7F7F)
            ws_a.views[5].fill_(0x7F7F7F7F)
            xb_fp8.zero_()
            xb_sf.fill_(0x7F7F7F7F)
            ws_b.views[5].fill_(0x7F7F7F7F)

            deep_gemm.bf16_mega_gate(
                bf16_a, gate_w, num_topk=TOP_K, use_shared_as_routed=False,
                num_shared_experts=0, routed_scaling_factor=1.5, ep_rank=0,
                scoring_func="sqrtsoftplus", mask=valid_a, bias=gate_b,
                out=(ids_out_a, probs_out_a),
            )
            deep_gemm.bf16_mega_gate(
                bf16_b, gate_w, num_topk=TOP_K, use_shared_as_routed=False,
                num_shared_experts=0, routed_scaling_factor=1.5, ep_rank=0,
                scoring_func="sqrtsoftplus", mask=valid_b, bias=gate_b,
                out=(ids_out_b, probs_out_b),
            )
            op_a.encoded(bf16_a, (xa_fp8, xa_sf))
            op_b.encoded(bf16_b, (xb_fp8, xb_sf))
        torch.cuda.synchronize()

        gated_ids_a = op_a.encoded_inputs()[2]
        gated_ids_b = op_b.encoded_inputs()[2]
        assert (gated_ids_b[~valid_b] == -1).all(), "bf16_mega_gate did not mask invalid rows to -1"
        assert torch.equal(gated_ids_b[valid_b], gated_ids_a), "active gated routes diverged between unpadded and padded batches"
        assert_predictor_state_equal("mega_gate_valid_token_mask")
        assert int(bind_a.scores.sum().item()) > 0, "expected gated active rows to produce nonzero predictor scores"
    finally:
        mover_a.close()
        mover_b.close()
        for r in (*tier_a.regions, *tier_b.regions):
            r.close()


def test_sm103_inline_predictor_max_predict_rows_boundary_and_bypass(sm103_pred_storage):
    """Verify predictor executes at rows == max_predict_rows (64) and leaves all predictor/queue state bit-for-bit unchanged at rows == 65 and 128."""
    from ignition.components.megamoe import MegaMoeGeometry
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import PredictorConfig, SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    E, S_ACTIVE, HBM_SLOTS, TOP_K = 64, 16, 18, 4
    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    geom = MegaMoeGeometry(E, 128, TOP_K, 128, 128, shared_experts=1)
    tier = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(S_ACTIVE)),
        storage=sm103_pred_storage, name="pred_boundary_layer0",
    )
    hbm = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    mover = SM103ExpertMoverRuntime(
        [(tier, hbm)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8,
        control_buffer_name="boundary_ctrl",
    )
    try:
        cfg = PredictorConfig(
            decay_shift=3, promote_threshold=256, hysteresis=64, cooldown_steps=1,
            max_enqueues=2, max_predict_rows=64,
        )
        (binding,) = mover.attach_predictors(cfg)

        def snapshot():
            return (
                binding.scores.clone(),
                binding.last_seen.clone(),
                binding.cooldown.clone(),
                binding.state.clone(),
                binding.queue_header.clone(),
                binding.queue_ring.clone(),
            )

        def assert_snapshots_equal(before, after, label):
            names = ("scores", "last_seen", "cooldown", "state", "queue_header", "queue_ring")
            for name, lhs, rhs in zip(names, before, after):
                assert torch.equal(lhs, rhs), f"{label}: bypassed call mutated {name}"

        def call_predict(rows: int, expert_offset: int):
            ids = torch.arange(expert_offset, expert_offset + TOP_K, dtype=torch.int64, device=device).expand(rows, -1).contiguous()
            # Poison active_slot_map to confirm bypassed calls still execute the Dekker live_slot_map snapshot.
            tier.control[16:16 + E].fill_(-99)
            mover.binding.launch_layer_enter_predict(
                tier.live_slot_map, tier.control, tier.layer_epoch, ids,
                rows, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                binding.metrics_ptr, binding.metrics_count,
            )
            torch.cuda.synchronize()
            assert torch.equal(tier.control[16:16 + E], tier.live_slot_map)

        # Before any active step: rows=65 and rows=128 must bypass prediction completely.
        initial = snapshot()
        call_predict(65, 16)
        assert int(tier.layer_epoch.item()) == 1
        assert_snapshots_equal(initial, snapshot(), "initial_rows_65")
        call_predict(128, 20)
        assert int(tier.layer_epoch.item()) == 2
        assert_snapshots_equal(initial, snapshot(), "initial_rows_128")

        # At exact inclusive boundary rows=64 (<= max_predict_rows): must update scores, recency, clock, counters, and queue.
        call_predict(64, 16)
        assert int(tier.layer_epoch.item()) == 3
        assert (binding.scores[16:16 + TOP_K] > 0).all(), "rows=64 did not update cold expert scores"
        assert (binding.last_seen[16:16 + TOP_K] == 2).all(), "rows=64 did not update last_seen"
        step_clock = struct.unpack_from("<I", bytes(binding.state[24:28].cpu().tolist()), 0)[0]
        assert step_clock == 2, f"expected step_clock=2 after rows=64, got {step_clock}"
        head, tail = struct.unpack_from("<2I", bytes(binding.queue_header[0:8].cpu().tolist()), 0)
        hits, misses, enqueues = struct.unpack_from("<3Q", bytes(binding.queue_header[48:72].cpu().tolist()), 0)
        assert head == 0 and tail == enqueues and enqueues > 0, (
            f"rows=64 did not enqueue swap request: head={head}, tail={tail}, enqueues={enqueues}"
        )
        assert hits == 0 and misses == 64 * TOP_K
        assert int((binding.cooldown > 0).sum().item()) == enqueues

        # With nonzero scores and queue state: rows=65 and rows=128 must not decay scores, advance step_clock, or mutate counters/queue.
        after_64 = snapshot()
        call_predict(65, 32)
        assert int(tier.layer_epoch.item()) == 4
        assert_snapshots_equal(after_64, snapshot(), "post64_rows_65")
        call_predict(128, 40)
        assert int(tier.layer_epoch.item()) == 5
        assert_snapshots_equal(after_64, snapshot(), "post64_rows_128")
    finally:
        mover.close()
        for r in tier.regions:
            r.close()


def test_sm103_inline_predictor_kernel_latency_384_experts(sm103_pred_storage):
    """Measure sm103_layer_enter_predict_kernel latency at E=384, top_k=6 across rows=1..8192."""
    from ignition.components.megamoe import MegaMoeGeometry
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import PredictorConfig, SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    E, HBM_SLOTS, TOP_K = 384, 290, 6
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, TOP_K, 128, 128, shared_experts=1)
    tier = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(288)),
        storage=sm103_pred_storage, name="pred_lat_layer0",
    )
    hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    mover = SM103ExpertMoverRuntime(
        [(tier, hbm_pool)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8
    )
    try:
        (binding,) = mover.attach_predictors(PredictorConfig(max_enqueues=2, max_predict_rows=64))
        gen = torch.Generator(device="cpu").manual_seed(99)
        cap_s = torch.cuda.Stream(device=device)
        iters = 200

        def measure_graph(fn):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(cap_s):
                with torch.cuda.graph(g, stream=cap_s):
                    for _ in range(iters):
                        fn()
            g.replay()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            us = (s.elapsed_time(e) * 1000.0) / iters
            g.reset()
            return us

        # Warmup
        warm_ids = torch.zeros((16, TOP_K), dtype=torch.int64, device=device)
        for _ in range(10):
            mover.binding.launch_layer_enter_predict(
                tier.live_slot_map, tier.control, tier.layer_epoch, warm_ids,
                16, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                binding.metrics_ptr, binding.metrics_count,
            )
        torch.cuda.synchronize()
        mover.reset_runtime_state()
        torch.cuda.synchronize()

        base_us = measure_graph(
            lambda: mover.binding.launch_layer_enter(tier.live_slot_map, tier.control, tier.layer_epoch, E)
        )
        results = {}
        for rows in (1, 16, 64, 1024, 8192):
            ids_hit = torch.stack([torch.randperm(288, generator=gen)[:TOP_K] for _ in range(rows)]).to(torch.int64).to(device).contiguous()
            ids_miss = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(rows)]).to(torch.int64).to(device).contiguous()
            hit_us = measure_graph(
                lambda r=rows, t=ids_hit: mover.binding.launch_layer_enter_predict(
                    tier.live_slot_map, tier.control, tier.layer_epoch, t,
                    r, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                    binding.metrics_ptr, binding.metrics_count,
                )
            )
            miss_us = measure_graph(
                lambda r=rows, t=ids_miss: mover.binding.launch_layer_enter_predict(
                    tier.live_slot_map, tier.control, tier.layer_epoch, t,
                    r, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                    binding.metrics_ptr, binding.metrics_count,
                )
            )
            results[rows] = (hit_us, miss_us)

        summary = ", ".join(
            f"rows={r}: hit={h:.2f}us miss={m:.2f}us" for r, (h, m) in results.items()
        )
        print(f"\n[Predictor Latency (Graph) E={E} top_k={TOP_K}] base_enter={base_us:.2f} us | {summary}")
        assert results[16][0] < 8.0, f"rows=16 steady hit too slow: {results[16][0]:.2f} us"
        assert results[16][1] < 12.0, f"rows=16 active miss too slow: {results[16][1]:.2f} us"
        # Prefill chunks above max_predict_rows=64 skip route tallying and pay only Dekker snapshot latency.
        assert results[8192][1] < 6.0, f"rows=8192 prefill bypass too slow: {results[8192][1]:.2f} us"
    finally:
        mover.close()
        for r in tier.regions:
            r.close()
