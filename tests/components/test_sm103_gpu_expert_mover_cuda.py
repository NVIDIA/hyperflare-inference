# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 hardware qualification for indirection-aware MegaMoE + 8-SM 1D-TMA persistent mover."""

import ctypes as C
import os
from pathlib import Path
import time

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned SM103 CUDA device (FLARE_RUN_CUDA_TESTS=1)",
)


class _GreenCtxPair:
    """Minimal CUDA driver helper splitting device SMs into worker + compute green contexts."""

    CU_DEV_RESOURCE_TYPE_SM = 1
    CU_GREEN_CTX_DEFAULT_STREAM = 1
    CU_STREAM_NON_BLOCKING = 1

    class _DevResource(C.Structure):
        _fields_ = [
            ("type", C.c_int),
            ("_internal_padding", C.c_ubyte * 92),
            ("sm_count", C.c_uint32),
            ("min_sm_partition_size", C.c_uint32),
            ("sm_coscheduled_alignment", C.c_uint32),
            ("flags", C.c_uint32),
            ("_tail", C.c_ubyte * 32),
        ]

    def __init__(self, device_ordinal: int, worker_sm_count: int):
        self.lib = C.CDLL("libcuda.so.1")
        self.device = C.c_int(device_ordinal)
        self.primary_ctx = C.c_void_p()
        self._call("cuDevicePrimaryCtxRetain", C.byref(self.primary_ctx), self.device)
        self._call("cuCtxSetCurrent", self.primary_ctx)

        sm_res = self._DevResource()
        self._call("cuDeviceGetDevResource", self.device, C.byref(sm_res), C.c_int(self.CU_DEV_RESOURCE_TYPE_SM))
        w_res = self._DevResource()
        c_res = self._DevResource()
        nb = C.c_uint(1)
        self._call(
            "cuDevSmResourceSplitByCount",
            C.byref(w_res), C.byref(nb), C.byref(sm_res), C.byref(c_res),
            C.c_uint(0), C.c_uint(worker_sm_count),
        )
        self.worker_sms = int(w_res.sm_count)
        self.compute_sms = int(c_res.sm_count)

        w_desc, c_desc = C.c_void_p(), C.c_void_p()
        self._call("cuDevResourceGenerateDesc", C.byref(w_desc), C.byref(w_res), C.c_uint(1))
        self._call("cuDevResourceGenerateDesc", C.byref(c_desc), C.byref(c_res), C.c_uint(1))

        self.w_gctx, self.c_gctx = C.c_void_p(), C.c_void_p()
        self._call("cuGreenCtxCreate", C.byref(self.w_gctx), w_desc, self.device, C.c_uint(self.CU_GREEN_CTX_DEFAULT_STREAM))
        self._call("cuGreenCtxCreate", C.byref(self.c_gctx), c_desc, self.device, C.c_uint(self.CU_GREEN_CTX_DEFAULT_STREAM))

        self.w_ctx, self.c_ctx = C.c_void_p(), C.c_void_p()
        self._call("cuCtxFromGreenCtx", C.byref(self.w_ctx), self.w_gctx)
        self.w_stream, self.c_stream = C.c_void_p(), C.c_void_p()
        self._call("cuCtxFromGreenCtx", C.byref(self.c_ctx), self.c_gctx)
        self._call("cuGreenCtxStreamCreate", C.byref(self.w_stream), self.w_gctx, C.c_uint(self.CU_STREAM_NON_BLOCKING), C.c_int(0))
        self._call("cuGreenCtxStreamCreate", C.byref(self.c_stream), self.c_gctx, C.c_uint(self.CU_STREAM_NON_BLOCKING), C.c_int(0))

    def _call(self, name, *args):
        fn = getattr(self.lib, name)
        fn.restype = C.c_int
        rc = fn(*args)
        if rc != 0:
            msg = C.c_char_p()
            self.lib.cuGetErrorString(C.c_int(rc), C.byref(msg))
            raise RuntimeError(f"{name} failed with CUDA {rc}: {msg.value!r}")

    def set_worker(self):
        self._call("cuCtxSetCurrent", self.w_ctx)

    def set_compute(self):
        self._call("cuCtxSetCurrent", self.c_ctx)

    def set_primary(self):
        self._call("cuCtxSetCurrent", self.primary_ctx)

    def launch_graph(self, graph_exec: int, *, worker: bool):
        self._call("cuCtxSetCurrent", self.w_ctx if worker else self.c_ctx)
        stream = self.w_stream if worker else self.c_stream
        self._call("cuGraphLaunch", C.c_void_p(graph_exec), stream)

    def create_compute_stream(self):
        stream = C.c_void_p()
        self._call("cuGreenCtxStreamCreate", C.byref(stream), self.c_gctx, C.c_uint(self.CU_STREAM_NON_BLOCKING), C.c_int(0))
        return stream

    def launch_graph_on_stream(self, graph_exec: int, stream, *, worker: bool = False):
        self._call("cuCtxSetCurrent", self.w_ctx if worker else self.c_ctx)
        self._call("cuGraphLaunch", C.c_void_p(graph_exec), stream)

    def sync_stream(self, stream, *, worker: bool = False):
        self._call("cuCtxSetCurrent", self.w_ctx if worker else self.c_ctx)
        self._call("cuStreamSynchronize", stream)

    def destroy_stream(self, stream):
        self._call("cuCtxSetCurrent", self.primary_ctx)
        self._call("cuStreamDestroy_v2", stream)

    def sync_worker(self):
        self._call("cuCtxSetCurrent", self.w_ctx)
        self._call("cuStreamSynchronize", self.w_stream)

    def sync_compute(self):
        self._call("cuCtxSetCurrent", self.c_ctx)
        self._call("cuStreamSynchronize", self.c_stream)

    def close(self):
        self._call("cuCtxSetCurrent", self.primary_ctx)
        self._call("cuStreamDestroy_v2", self.c_stream)
        self._call("cuStreamDestroy_v2", self.w_stream)
        self._call("cuGreenCtxDestroy", self.c_gctx)
        self._call("cuGreenCtxDestroy", self.w_gctx)
        self._call("cuDevicePrimaryCtxRelease_v2", self.device)


def _make_random_mxfp4_weights(experts, hidden, intermediate, device, gen):
    from ignition.components.megamoe_weights import allocate_mxfp4_weights

    weights = allocate_mxfp4_weights(experts, hidden, intermediate, device=device)
    # Use deterministic non-zero FP4 nibbles and moderate UE8M0 exponents (124..127 -> 0.125..1.0)
    weights.gate_up[0].copy_(torch.randint(-120, 120, weights.gate_up[0].shape, dtype=torch.int8, generator=gen).to(device))
    weights.down[0].copy_(torch.randint(-120, 120, weights.down[0].shape, dtype=torch.int8, generator=gen).to(device))
    for scale in (weights.gate_up[1], weights.down[1]):
        raw = torch.randint(123, 127, (*scale.shape, 4), dtype=torch.uint8, generator=gen)
        packed = raw.view(torch.int32).squeeze(-1).to(device)
        scale.copy_(packed)
    return weights


def _make_random_shared_fp8(hidden, intermediate, device, gen):
    from ignition.components.megamoe_weights import MegaMoeWeights

    def make_pair(rows, cols):
        sf_cols = cols // 128
        vals = (torch.randn((rows, cols), generator=gen) * 0.25).to(torch.float8_e4m3fn).to(device)
        raw = torch.randint(125, 128, (rows, sf_cols, 4), dtype=torch.uint8, generator=gen)
        sf_cpu = raw.view(torch.int32).squeeze(-1)
        sf = torch.empty_strided((rows, sf_cols), (1, rows), dtype=torch.int32, device=device)
        sf.copy_(sf_cpu.to(device))
        return (vals, sf)

    return MegaMoeWeights(make_pair(2 * intermediate, hidden), make_pair(hidden, intermediate))


@pytest.fixture(scope="module")
def sm103_storage(megamoe_core_storage):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 GB300 required")
    return megamoe_core_storage


def test_sm103_indirection_megamoe_bit_exact_under_concurrent_8sm_tma_swaps(sm103_storage):
    """Verify bit-exact parity of indirection MegaMoE vs all-HBM reference under live concurrent swaps."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    worker_sms = 8
    # Match CUDA green-context SM split rounding (always even for 2-CTA MegaMoE cluster).
    gc = _GreenCtxPair(0, worker_sms)
    compute_sms = gc.compute_sms
    assert gc.worker_sms == worker_sms and compute_sms > 0 and compute_sms % 2 == 0
    print(f"\n[SM103] total_sms={props.multi_processor_count} worker_sms={gc.worker_sms} compute_sms={compute_sms}")

    E = 32
    S_ACTIVE = 14
    K_SPARE = 2
    HBM_SLOTS = S_ACTIVE + K_SPARE  # 16 slots (< 32 canonical experts)
    HIDDEN = 1024
    INTER = 512
    TOP_K = 4
    CAPACITY = 128
    ROW_SHAPES = (1, 16, 128)
    CLAMP = 7.0

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    storage = sm103_storage
    mover = None
    tier = None
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        if True:
            gen = torch.Generator(device="cpu").manual_seed(20261120)
            geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
            ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
            ws_dyn = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

            full_weights = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
            shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

            # Prepare dynamic tier: ALL E canonical experts in HOST_NUMA, initial 0..S_ACTIVE-1 in HBM, K_SPARE free.
            initial_hbm = tuple(range(S_ACTIVE))
            tier = MegaMoeDynamicHostTier(
                ws_dyn.geometry, hbm_slots=HBM_SLOTS, initial_hbm_experts=initial_hbm, storage=storage, name="layer0"
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
                worker_sm_count=gc.worker_sms,
                queue_capacity=1024,
            )

            # Prepare static all-HBM reference and dynamic indirection MegaMoE for all row shapes.
            ref_ops = {}
            dyn_ops = {}
            with PreparationSession(device=device, autotune=False) as session:
                for rows in ROW_SHAPES:
                    ex = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=device)
                    ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
                    probs = torch.full((rows, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
                    pad = torch.zeros((rows,), dtype=torch.bool, device=device)
                    ref_ops[rows] = PreparedMegaMoe(
                        ex, ids, probs, pad, full_weights, shared_weights,
                        workspace=ws_ref, clamp=CLAMP, session=session, name=f"ref.{rows}",
                        host_shared_first=True,
                    )
                    dyn_ops[rows] = PreparedMegaMoe(
                        ex, ids, probs, pad, hbm_pool, shared_weights,
                        workspace=ws_dyn, clamp=CLAMP, session=session, name=f"dyn.{rows}",
                        host_tier=tier, host_shared_first=True, compute_sm_count=compute_sms,
                    )

            # Build deterministic test batches across all row shapes (mixing hot & cold experts).
            batches = []
            masked_steps = 0
            masked_shapes = set()
            for step in range(40):
                rows = ROW_SHAPES[step % len(ROW_SHAPES)]
                x = (torch.randn((rows, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16).to(device)
                # Unique top-k expert IDs per token across all 0..E-1.
                perm = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(rows)])
                ids = perm.to(torch.int32).to(device).contiguous()
                probs = torch.softmax(torch.randn((rows, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)
                pad = torch.zeros((rows,), dtype=torch.bool, device=device)
                if rows >= 8 and (step // len(ROW_SHAPES)) % 2 == 1:
                    pad[rows // 2:] = True  # Exercise masked inactive rows on both rows=16 and rows=128
                    masked_steps += 1
                    masked_shapes.add(rows)
                batches.append((rows, x, ids, probs, pad))
            assert masked_steps > 0 and masked_shapes == {16, 128}, (
                f"expected masked inactive-row batches on rows=16 and rows=128, got steps={masked_steps} shapes={masked_shapes}"
            )

            # 1. Compute golden all-HBM reference outputs on primary context.
            print("[SM103] Session prepared! Computing golden all-HBM reference...", flush=True)
            golden = []
            for rows, x, ids, probs, pad in batches:
                out = ref_ops[rows](x, ids, probs, pad)
                golden.append(out.clone())
            torch.cuda.synchronize()
            print(f"[SM103] Golden reference ready ({len(golden)} batches). Checking static indirection...", flush=True)

            # 2. Verify static indirection parity BEFORE starting mover (mix of HBM + HOST_NUMA experts).
            for idx, (rows, x, ids, probs, pad) in enumerate(batches[:6]):
                out = dyn_ops[rows](x, ids, probs, pad)
                torch.cuda.synchronize()
                max_diff = (out.float() - golden[idx].float()).abs().max().item()
                assert max_diff == 0.0, f"static indirection mismatch at batch {idx} (rows={rows}): max_diff={max_diff}"

            # 3. Pre-capture compute graphs (zero GPU allocation while worker is running) and worker graph.
            print("[SM103] Static indirection bit-exact! Pre-capturing compute + worker graphs...", flush=True)
            # Capture one compute graph per row shape using fixed persistent I/O buffers.
            graph_io = {}
            compute_graphs = {}
            cap_stream = torch.cuda.Stream(device=device)
            for rows in ROW_SHAPES:
                x_buf = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=device)
                ids_buf = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
                probs_buf = torch.full((rows, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
                pad_buf = torch.zeros((rows,), dtype=torch.bool, device=device)
                out_buf = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
                # Warmup once outside capture
                out_buf.copy_(dyn_ops[rows](x_buf, ids_buf, probs_buf, pad_buf))
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.stream(cap_stream):
                    with torch.cuda.graph(g, pool=storage.graph_pool, stream=cap_stream):
                        out_buf.copy_(dyn_ops[rows](x_buf, ids_buf, probs_buf, pad_buf))
                g.instantiate()
                graph_io[rows] = (x_buf, ids_buf, probs_buf, pad_buf, out_buf)
                compute_graphs[rows] = g

            # Capture a batch-swap enqueue graph (rotates through all E experts) and the persistent mover graph.
            swap_layers = torch.zeros((E,), dtype=torch.int32, device=device)
            swap_experts = torch.arange(E, dtype=torch.int32, device=device)
            swap_victims = torch.full((E,), -1, dtype=torch.int32, device=device)
            enqueue_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.stream(cap_stream):
                with torch.cuda.graph(enqueue_graph, pool=storage.graph_pool, stream=cap_stream):
                    mover.enqueue_swap_batch(swap_layers, swap_experts, swap_victims)
            enqueue_graph.instantiate()

            worker_decl = mover.capture_worker_graph(name="sm103_mover_test", retain_graphs=True, stream=cap_stream)
            worker_exec = int(worker_decl.graph.raw_cuda_graph_exec())
            enqueue_exec = int(enqueue_graph.raw_cuda_graph_exec())

            # Pre-allocate pinned host staging for bit-exact output comparison without touching GPU allocator.
            cpu_batches = [
                (rows, x.cpu(), ids.cpu(), probs.cpu(), pad.cpu(), g_out.cpu())
                for (rows, x, ids, probs, pad), g_out in zip(batches, golden)
            ]
            host_out = {rows: torch.empty((rows, HIDDEN), dtype=torch.bfloat16, pin_memory=False) for rows in ROW_SHAPES}
            mover.reset_stop_flag()
            torch.cuda.synchronize()

            print("[SM103] Launching persistent mover graph on 8-SM worker green context...", flush=True)
            gc.launch_graph(worker_exec, worker=True)

            try:
                # Step 3a: Verify mover executes a batch of swaps cleanly before compute interleaving.
                gc.launch_graph(enqueue_exec, worker=False)
                gc.sync_compute()
                deadline = time.monotonic() + 5.0
                while mover.read_metrics().swaps_completed < 10 and time.monotonic() < deadline:
                    time.sleep(0.005)
                m0 = mover.read_metrics()
                print(f"[SM103] Initial mover burst OK: {m0}", flush=True)
                assert m0.swaps_completed >= 10, f"mover did not progress on worker green context: {m0}"

                # Step 3b: Hammer 144-SM compute graphs concurrently with live 8-SM swaps!
                for round_idx in range(4):
                    shift = (round_idx * 7) % E
                    rotated = torch.roll(torch.arange(E, dtype=torch.int32), shifts=shift)
                    gc.set_compute()
                    gc._call(
                        "cuMemcpyHtoDAsync_v2",
                        C.c_uint64(swap_experts.data_ptr()),
                        C.c_void_p(rotated.data_ptr()),
                        C.c_size_t(rotated.numel() * 4),
                        gc.c_stream,
                    )
                    gc.launch_graph(enqueue_exec, worker=False)

                    for idx, (rows, x_cpu, ids_cpu, probs_cpu, pad_cpu, gold_cpu) in enumerate(cpu_batches):
                        x_buf, ids_buf, probs_buf, pad_buf, out_buf = graph_io[rows]
                        gc.set_compute()
                        for dst, src in ((x_buf, x_cpu), (ids_buf, ids_cpu), (probs_buf, probs_cpu), (pad_buf, pad_cpu)):
                            gc._call(
                                "cuMemcpyHtoDAsync_v2",
                                C.c_uint64(dst.data_ptr()),
                                C.c_void_p(src.data_ptr()),
                                C.c_size_t(src.numel() * src.element_size()),
                                gc.c_stream,
                            )
                        gc.launch_graph(int(compute_graphs[rows].raw_cuda_graph_exec()), worker=False)
                        dst_cpu = host_out[rows]
                        gc._call(
                            "cuMemcpyDtoHAsync_v2",
                            C.c_void_p(dst_cpu.data_ptr()),
                            C.c_uint64(out_buf.data_ptr()),
                            C.c_size_t(dst_cpu.numel() * dst_cpu.element_size()),
                            gc.c_stream,
                        )
                        gc.sync_compute()
                        if not torch.equal(dst_cpu, gold_cpu):
                            max_diff = (dst_cpu.float() - gold_cpu.float()).abs().max().item()
                            raise AssertionError(
                                f"LIVE SWAP CORRUPTION at round={round_idx} batch={idx} (rows={rows}): "
                                f"max_diff={max_diff}"
                            )
                    print(f"[SM103] Round {round_idx} bit-exact! metrics={mover.read_metrics()}", flush=True)
            finally:
                # Drain compute BEFORE signaling mover stop!
                gc.sync_compute()
                mover.signal_stop()
                gc.sync_worker()
                gc.set_primary()
                worker_decl.graph.reset()
                enqueue_graph.reset()
                for g in compute_graphs.values():
                    g.reset()

            metrics = mover.read_metrics()
            print(
                f"[SM103] LIVE SWAP PARITY PASSED! rounds=4 batches={4 * len(batches)} "
                f"swaps_completed={metrics.swaps_completed} bytes_copied={metrics.bytes_copied} "
                f"rcu_waits={metrics.rcu_waits} skipped_resident={metrics.skipped_already_resident}"
            )
            assert metrics.swaps_completed >= 20, f"expected active live swaps, got {metrics}"
            assert metrics.bytes_copied == metrics.swaps_completed * sum(tier.plane_row_bytes)
            assert int(tier.layer_epoch.item()) % 2 == 0, "layer_epoch must finish even after drain"

            # Verify slot_map <-> slot_owner invariant after hundreds of live swaps!
            final_map = tier.live_slot_map.cpu().tolist()
            final_owner = tier.slot_owner.cpu().tolist()
            resident_from_map = {e: s for e, s in enumerate(final_map) if s >= 0}
            resident_from_owner = {e: s for s, e in enumerate(final_owner) if e >= 0}
            assert resident_from_map == resident_from_owner, (
                f"slot_map / slot_owner diverged: {resident_from_map} vs {resident_from_owner}"
            )
            assert len(resident_from_map) == S_ACTIVE
            assert final_owner.count(-1) == K_SPARE
    finally:
        if mover is not None:
            mover.close()
        if tier is not None:
            for r in tier.regions:
                r.close()
        gc.close()


def test_sm103_deepseek41_expert_geometry_tma_bandwidth_and_parity(sm103_storage):
    """Verify full DeepSeek V4.1 Flash expert geometry (E=384, H=5120, I=2304, top_k=6) bit-exact against all-HBM MegaMoE."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    worker_sms = 8
    gc = _GreenCtxPair(0, worker_sms)
    compute_sms = gc.compute_sms
    E = 384
    S_ACTIVE = 32
    K_SPARE = 2
    HBM_SLOTS = S_ACTIVE + K_SPARE
    HIDDEN = 5120
    INTER = 2304
    TOP_K = 6
    CAPACITY = 128
    ROWS = 16
    CLAMP = 7.0

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    storage = sm103_storage
    mover = None
    tier = None
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        gen = torch.Generator(device="cpu").manual_seed(4102026)
        geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
        ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        ws = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

        # Fill all 384 canonical experts in mapped host (~7.05 GiB) and copy into an independent all-HBM reference.
        tier = MegaMoeDynamicHostTier(
            ws.geometry, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(S_ACTIVE)),
            storage=storage, name="dsv41_layer0",
        )
        for dst_cpu in tier.host_views:
            if dst_cpu.dtype == torch.int8:
                vals = torch.randint(-120, 120, (E, 1, dst_cpu.shape[2]), dtype=torch.int8, generator=gen)
                dst_cpu.copy_(vals.expand_as(dst_cpu))
            else:
                raw = torch.randint(123, 127, (E, 1, dst_cpu.shape[2], 4), dtype=torch.uint8, generator=gen)
                packed = raw.view(torch.int32).squeeze(-1)
                dst_cpu.copy_(packed.expand_as(dst_cpu))

        full_hbm = allocate_mxfp4_weights(E, HIDDEN, INTER, device=device)
        for dst, src_cpu in zip((*full_hbm.gate_up, *full_hbm.down), tier.host_views):
            dst.copy_(src_cpu.to(device))

        hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
        for dst, src_cpu in zip((*hbm_pool.gate_up, *hbm_pool.down), tier.host_views):
            dst[:S_ACTIVE].copy_(src_cpu[:S_ACTIVE].to(device))
            dst[S_ACTIVE:].zero_()

        shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)
        mover = SM103ExpertMoverRuntime(
            [(tier, hbm_pool)], storage=storage, kernel_cache=kernel_cache,
            worker_sm_count=gc.worker_sms, queue_capacity=1024,
        )

        with PreparationSession(device=device, autotune=False) as session:
            ex = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
            probs = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)
            ref_op = PreparedMegaMoe(
                ex, ids, probs, pad, full_hbm, shared_weights,
                workspace=ws_ref, clamp=CLAMP, session=session, name="dsv41_ref_hbm",
                host_shared_first=True,
            )
            dyn_op = PreparedMegaMoe(
                ex, ids, probs, pad, hbm_pool, shared_weights,
                workspace=ws, clamp=CLAMP, session=session, name="dsv41_dyn",
                host_tier=tier, host_shared_first=True, compute_sm_count=compute_sms,
            )

        # Compute independent all-HBM golden reference across all 384 experts.
        x_buf = (torch.randn((ROWS, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16).to(device)
        ids_buf = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(ROWS)]).to(torch.int32).to(device).contiguous()
        probs_buf = torch.softmax(torch.randn((ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)
        pad_buf = torch.zeros((ROWS,), dtype=torch.bool, device=device)
        out_buf = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        golden = ref_op(x_buf, ids_buf, probs_buf, pad_buf).clone()
        torch.cuda.synchronize()
        assert torch.equal(dyn_op(x_buf, ids_buf, probs_buf, pad_buf), golden), "initial indirection mismatch vs all-HBM"

        cap_stream = torch.cuda.Stream(device=device)
        comp_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(comp_graph, pool=storage.graph_pool, stream=cap_stream):
                out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        comp_graph.instantiate()

        # Enqueue 32 full DeepSeek V4.1 Flash expert swaps (~18.80 MiB each).
        n_swaps = 32
        swap_layers = torch.zeros((n_swaps,), dtype=torch.int32, device=device)
        swap_experts = torch.arange(128, 128 + n_swaps, dtype=torch.int32, device=device)
        swap_victims = torch.full((n_swaps,), -1, dtype=torch.int32, device=device)
        enq_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(enq_graph, pool=storage.graph_pool, stream=cap_stream):
                mover.enqueue_swap_batch(swap_layers, swap_experts, swap_victims)
        enq_graph.instantiate()

        worker_decl = mover.capture_worker_graph(name="dsv41_mover", retain_graphs=True, stream=cap_stream)
        mover.reset_stop_flag()
        torch.cuda.synchronize()

        gc.launch_graph(int(worker_decl.graph.raw_cuda_graph_exec()), worker=True)
        try:
            t0 = time.perf_counter()
            gc.launch_graph(int(enq_graph.raw_cuda_graph_exec()), worker=False)
            gc.sync_compute()
            host_out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16)
            gold_cpu = golden.cpu()
            for _ in range(16):
                gc.launch_graph(int(comp_graph.raw_cuda_graph_exec()), worker=False)
                gc.set_compute()
                gc._call(
                    "cuMemcpyDtoHAsync_v2",
                    C.c_void_p(host_out.data_ptr()),
                    C.c_uint64(out_buf.data_ptr()),
                    C.c_size_t(host_out.numel() * host_out.element_size()),
                    gc.c_stream,
                )
                gc.sync_compute()
                assert torch.equal(host_out, gold_cpu), "DS V4.1 full-geometry live swap mismatch vs all-HBM reference"
            deadline = time.monotonic() + 5.0
            while mover.read_metrics().swaps_completed < n_swaps and time.monotonic() < deadline:
                time.sleep(0.0005)
            elapsed = time.perf_counter() - t0
        finally:
            gc.sync_compute()
            mover.signal_stop()
            gc.sync_worker()
            gc.set_primary()
            worker_decl.graph.reset()
            enq_graph.reset()
            comp_graph.reset()

        m = mover.read_metrics()
        expert_mib = sum(tier.plane_row_bytes) / (1024 * 1024)
        bw_gbs = (m.bytes_copied / 1e9) / elapsed
        print(
            f"\n[DS V4.1 Flash Geometry vs All-HBM] E={E} H={HIDDEN} I={INTER} top_k={TOP_K} ({expert_mib:.3f} MiB/expert): "
            f"swaps={m.swaps_completed}/{n_swaps} bytes={m.bytes_copied / 1e6:.1f} MB "
            f"elapsed={elapsed * 1e3:.2f} ms effective_bw={bw_gbs:.1f} GB/s rcu_waits={m.rcu_waits} -> BIT-EXACT"
        )
        assert m.swaps_completed == n_swaps
        assert m.bytes_copied == n_swaps * sum(tier.plane_row_bytes)
        assert int(tier.layer_epoch.item()) % 2 == 0
    finally:
        if mover is not None:
            mover.close()
        if tier is not None:
            for r in tier.regions:
                r.close()
        gc.close()


def test_sm103_8sm_1d_tma_mover_host_numa_vmm_bandwidth():
    """Verify sustained 8-SM 1D-TMA mover throughput over CU_MEM_LOCATION_TYPE_HOST_NUMA VMM on SM103."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 GB300 required")

    import struct
    from ignition.components.sm103_mover_kernel import prepare as prepare_mover

    dev = torch.device("cuda", 0)
    gc = _GreenCtxPair(0, 8)
    binding = prepare_mover(Path("/tmp/flare-test-kcache"))
    binding.configure_persistent_mover()

    class _Loc(C.Structure):
        _fields_ = [("type", C.c_int), ("id", C.c_int)]

    class _Prop(C.Structure):
        _fields_ = [
            ("type", C.c_int),
            ("requestedHandleTypes", C.c_int),
            ("location", _Loc),
            ("win32HandleMetaData", C.c_void_p),
            ("allocFlags", C.c_ubyte * 8),
        ]

    class _Access(C.Structure):
        _fields_ = [("location", _Loc), ("flags", C.c_int)]

    numa_id = C.c_int(0)
    gc._call("cuDeviceGetAttribute", C.byref(numa_id), C.c_int(134), C.c_int(0))
    host_numa = max(numa_id.value, 0)
    vmm_allocs = []

    def alloc_host_numa(nbytes: int):
        prop = _Prop(type=1, requestedHandleTypes=0, location=_Loc(type=3, id=host_numa))
        gran = C.c_size_t(0)
        gc._call("cuMemGetAllocationGranularity", C.byref(gran), C.byref(prop), C.c_int(0))
        padded = ((nbytes + gran.value - 1) // gran.value) * gran.value
        handle = C.c_uint64(0)
        ptr = C.c_uint64(0)
        gc._call("cuMemCreate", C.byref(handle), C.c_size_t(padded), C.byref(prop), C.c_uint64(0))
        gc._call("cuMemAddressReserve", C.byref(ptr), C.c_size_t(padded), C.c_size_t(gran.value), C.c_uint64(0), C.c_uint64(0))
        gc._call("cuMemMap", ptr, C.c_size_t(padded), C.c_size_t(0), handle, C.c_uint64(0))
        acc = (_Access * 2)(_Access(_Loc(1, 0), 3), _Access(_Loc(3, host_numa), 3))
        gc._call("cuMemSetAccess", ptr, C.c_size_t(padded), acc, C.c_size_t(2))
        vmm_allocs.append((ptr.value, handle, padded))
        return ptr.value

    E, S = 64, 16
    row_bytes = [12582912, 786432, 6291456, 393216]  # Exact DeepSeek V4.1 MXFP4 + UE8M0 plane strides (19.125 MiB)
    g_work = None
    g_enq = None
    try:
        host_ptrs = [alloc_host_numa(E * rb) for rb in row_bytes]
        ctrl_ptr = alloc_host_numa(4096)
        C.memset(ctrl_ptr, 0, 128)

        hbm_planes = [torch.empty((S * rb,), dtype=torch.uint8, device=dev) for rb in row_bytes]
        live_map = torch.tensor([i if i < S - 2 else -1 for i in range(E)], dtype=torch.int32, device=dev)
        slot_owner = torch.tensor([i if i < S - 2 else -1 for i in range(S)], dtype=torch.int32, device=dev)
        layer_epoch = torch.zeros((1,), dtype=torch.int32, device=dev)

        entry = struct.pack(
            "<4Q4Q4Q3Q4I",
            *host_ptrs,
            *(int(t.data_ptr()) for t in hbm_planes),
            *row_bytes,
            int(live_map.data_ptr()),
            int(slot_owner.data_ptr()),
            int(layer_epoch.data_ptr()),
            E, S, 0, 0,
        )
        descs = torch.tensor(list(entry) + [0] * (192 - len(entry)), dtype=torch.uint8, device=dev)
        q_hdr = torch.zeros((64,), dtype=torch.uint8, device=dev)
        q_hdr[:12].copy_(torch.tensor(list(struct.pack("<III", 0, 0, 1023)), dtype=torch.uint8))
        q_ring = torch.zeros((1024 * 16,), dtype=torch.uint8, device=dev)

        cap_s = torch.cuda.Stream(device=dev)
        g_work = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_work, stream=cap_s):
                binding.launch_persistent_mover(descs, 1, q_hdr, q_ring, ctrl_ptr, ctrl_ptr + 8, 5, 8, int(cap_s.cuda_stream))
        g_work.instantiate()

        n_swaps = 48
        layers = torch.zeros((n_swaps,), dtype=torch.int32, device=dev)
        experts = torch.arange(S - 2, S - 2 + n_swaps, dtype=torch.int32, device=dev)
        victims = torch.full((n_swaps,), -1, dtype=torch.int32, device=dev)
        g_enq = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_enq, stream=cap_s):
                binding.launch_enqueue_swap_batch(q_hdr, q_ring, layers, experts, victims)
        g_enq.instantiate()
        torch.cuda.synchronize()

        gc.launch_graph(int(g_work.raw_cuda_graph_exec()), worker=True)
        try:
            t0 = time.perf_counter()
            gc.launch_graph(int(g_enq.raw_cuda_graph_exec()), worker=False)
            gc.sync_compute()
            deadline = time.monotonic() + 5.0
            while C.c_uint64.from_address(ctrl_ptr + 8).value < n_swaps and time.monotonic() < deadline:
                pass
            elapsed = time.perf_counter() - t0
        finally:
            C.c_uint32.from_address(ctrl_ptr).value = 1
            gc.sync_worker()
            gc.set_primary()

        swaps = C.c_uint64.from_address(ctrl_ptr + 8).value
        copied = C.c_uint64.from_address(ctrl_ptr + 16).value
        bw_gbs = (copied / 1e9) / elapsed
        print(
            f"\n[HOST_NUMA VMM 8-SM 1D TMA] swaps={swaps}/{n_swaps} bytes={copied / 1e6:.1f} MB "
            f"elapsed={elapsed * 1e3:.2f} ms -> {bw_gbs:.1f} GB/s ({elapsed * 1e6 / swaps:.1f} us/expert)"
        )
        assert swaps == n_swaps
        assert copied == n_swaps * sum(row_bytes)
        assert bw_gbs >= 50.0, f"expected >= 50 GB/s over HOST_NUMA VMM on SM103, got {bw_gbs:.1f} GB/s"
    finally:
        if g_work is not None:
            g_work.reset()
        if g_enq is not None:
            g_enq.reset()
        gc.set_primary()
        for ptr, h, sz in vmm_allocs:
            gc._call("cuMemUnmap", C.c_uint64(ptr), C.c_size_t(sz))
            gc._call("cuMemRelease", h)
            gc._call("cuMemAddressFree", C.c_uint64(ptr), C.c_size_t(sz))
        gc.close()


def test_sm103_dekker_rcu_blocks_slot_reclamation_while_epoch_odd(sm103_storage):
    """Verify the Dekker/epoch-RCU grace period blocks victim-slot reclamation while layer_epoch is odd."""
    from ignition.components.megamoe import MegaMoeGeometry
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    gc = _GreenCtxPair(0, 8)
    E, S_ACTIVE, HBM_SLOTS = 32, 4, 6
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, 4, 256, 256, shared_experts=1)
    tier = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(S_ACTIVE)),
        storage=sm103_storage, name="rcu_gate_layer",
    )
    hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, 256, 256, device=device)
    mover = SM103ExpertMoverRuntime(
        [(tier, hbm_pool)], storage=sm103_storage, kernel_cache=kernel_cache, worker_sm_count=gc.worker_sms
    )
    try:
        cap_s = torch.cuda.Stream(device=device)
        # Hold layer_epoch odd (= 1) to simulate an active MegaMoE reader holding a snapshot of victim slot 0.
        tier.layer_epoch.fill_(1)
        g_enq = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_enq, pool=sm103_storage.graph_pool, stream=cap_s):
                mover.enqueue_swap(0, 10, 0)
        g_enq.instantiate()

        # Pre-capture epoch-release graph (1 -> 2, marking layer idle) so no eager CUDA calls occur during worker run.
        g_release = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_release, pool=sm103_storage.graph_pool, stream=cap_s):
                tier.layer_epoch.add_(1)
        g_release.instantiate()

        worker_decl = mover.capture_worker_graph(name="rcu_gate_mover", retain_graphs=True, stream=cap_s)
        mover.reset_stop_flag()
        torch.cuda.synchronize()

        gc.launch_graph(int(worker_decl.graph.raw_cuda_graph_exec()), worker=True)
        try:
            gc.launch_graph(int(g_enq.raw_cuda_graph_exec()), worker=False)
            gc.sync_compute()
            # Give the mover plenty of time to finish the 1D TMA copy and publish live_slot_map;
            # it MUST stall on odd layer_epoch=1 before reclaiming victim slot 0 or incrementing swaps_completed.
            time.sleep(0.025)
            m_blocked = mover.read_metrics()
            assert m_blocked.swaps_completed == 0, f"mover reclaimed victim slot during odd epoch: {m_blocked}"

            # Release layer_epoch (1 -> 2): mover must now complete reclamation with rcu_waits == 1.
            gc.launch_graph(int(g_release.raw_cuda_graph_exec()), worker=False)
            gc.sync_compute()
            deadline = time.monotonic() + 2.0
            while mover.read_metrics().swaps_completed < 1 and time.monotonic() < deadline:
                time.sleep(0.0005)
            m_done = mover.read_metrics()
            assert m_done.swaps_completed == 1, f"mover did not complete after epoch became even: {m_done}"
            assert m_done.rcu_waits == 1, f"expected rcu_waits == 1 after odd-epoch block, got {m_done}"
        finally:
            mover.signal_stop()
            gc.sync_worker()
            gc.set_primary()
            worker_decl.graph.reset()
            g_enq.reset()
            g_release.reset()

        assert int(tier.live_slot_map[10].item()) >= 0
        assert int(tier.live_slot_map[0].item()) == -1
        assert int(tier.slot_owner[0].item()) == -1, "victim slot 0 must be reclaimed as spare after epoch exit"
    finally:
        mover.close()
        for r in tier.regions:
            r.close()
        gc.close()


def test_sm103_indirection_vs_static_megamoe_per_call_latency(sm103_storage):
    """Measure per-call latency of indirection MegaMoE (144 SMs, no host lane) vs static MegaMoE (152 SMs)."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier, MegaMoeHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights

    device = torch.device("cuda", 0)
    gc = _GreenCtxPair(0, 8)
    E, HBM_ACTIVE, HIDDEN, INTER, TOP_K, ROWS, CLAMP = 384, 288, 5120, 2304, 6, 16, 7.0
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, TOP_K, HIDDEN, INTER, shared_experts=1)
    gen = torch.Generator(device="cpu").manual_seed(777)

    static_tier = None
    dyn_tier = None
    try:
        gc.set_primary()
        ws_static = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        ws_dyn = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        hbm_static = allocate_mxfp4_weights(HBM_ACTIVE, HIDDEN, INTER, device=device)
        hbm_dyn = allocate_mxfp4_weights(HBM_ACTIVE + 2, HIDDEN, INTER, device=device)
        shared = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

        static_tier = MegaMoeHostTier(
            ws_static.geometry, rows=E - HBM_ACTIVE, storage=sm103_storage,
            name="lat_static_host", host_lane_clusters=4,
        )
        dyn_tier = MegaMoeDynamicHostTier(
            ws_dyn.geometry, hbm_slots=HBM_ACTIVE + 2, initial_hbm_experts=tuple(range(HBM_ACTIVE)),
            storage=sm103_storage, name="lat_dyn_host",
        )

        ex = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        ids_hit = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
        probs = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
        pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)

        deep_gemm.set_num_sms(gc.compute_sms + gc.worker_sms)
        with PreparationSession(device=device, autotune=False) as s1:
            op_static = PreparedMegaMoe(
                ex, ids_hit, probs, pad, hbm_static, shared,
                workspace=ws_static, clamp=CLAMP, session=s1, name="lat_static",
                host_tier=static_tier, host_shared_first=True,
            )
        deep_gemm.set_num_sms(gc.compute_sms)
        with PreparationSession(device=device, autotune=False) as s2:
            op_dyn = PreparedMegaMoe(
                ex, ids_hit, probs, pad, hbm_dyn, shared,
                workspace=ws_dyn, clamp=CLAMP, session=s2, name="lat_dyn",
                host_tier=dyn_tier, host_shared_first=True, compute_sm_count=gc.compute_sms,
            )

        out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        host_rows = E - HBM_ACTIVE
        # Static MegaMoeHostTier places host experts at [0, host_rows) and HBM experts at [host_rows, E).
        # Dynamic MegaMoeDynamicHostTier initialized HBM experts at [0, HBM_ACTIVE) and host at [HBM_ACTIVE, E).
        static_ids_hbm = torch.stack([torch.randperm(HBM_ACTIVE, generator=gen)[:TOP_K] + host_rows for _ in range(ROWS)]).to(torch.int32).to(device).contiguous()
        static_ids_cold = torch.stack([torch.cat([
            torch.randperm(HBM_ACTIVE, generator=gen)[:TOP_K - 1] + host_rows,
            torch.randint(0, host_rows, (1,), generator=gen),
        ]) for _ in range(ROWS)]).to(torch.int32).to(device).contiguous()
        dyn_ids_hbm = torch.stack([torch.randperm(HBM_ACTIVE, generator=gen)[:TOP_K] for _ in range(ROWS)]).to(torch.int32).to(device).contiguous()
        dyn_ids_cold = torch.stack([torch.cat([
            torch.randperm(HBM_ACTIVE, generator=gen)[:TOP_K - 1],
            torch.randint(HBM_ACTIVE, E, (1,), generator=gen),
        ]) for _ in range(ROWS)]).to(torch.int32).to(device).contiguous()

        def bench(op, ids_t, iters=60):
            for _ in range(5):
                out.copy_(op(ex, ids_t, probs, pad))
            torch.cuda.synchronize()
            s = torch.cuda.Stream(device=device)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, pool=sm103_storage.graph_pool, stream=s):
                    for _ in range(iters):
                        out.copy_(op(ex, ids_t, probs, pad))
            g.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            us = e0.elapsed_time(e1) * 1000.0 / iters
            g.reset()
            return us

        static_hit_us = bench(op_static, static_ids_hbm)
        static_cold_us = bench(op_static, static_ids_cold)
        dyn_hit_us = bench(op_dyn, dyn_ids_hbm)
        dyn_cold_us = bench(op_dyn, dyn_ids_cold)
        print(
            f"\n[Per-Call Latency E={E} H={HIDDEN} I={INTER} rows={ROWS} top_k={TOP_K}] "
            f"static(152SM,lane4) all_hbm={static_hit_us:.1f} us, 1_cold/row={static_cold_us:.1f} us | "
            f"indir(144SM,lane0) all_hbm={dyn_hit_us:.1f} us, 1_cold/row={dyn_cold_us:.1f} us"
        )
        assert dyn_hit_us > 0 and static_hit_us > 0
    finally:
        if static_tier is not None:
            for r in static_tier.regions:
                r.close()
        if dyn_tier is not None:
            for r in dyn_tier.regions:
                r.close()
        gc.close()


def test_sm103_pinned_hbm_only_experts_parity_and_no_eviction(sm103_storage):
    """Verify pinned HBM-only experts (host_experts < experts) match all-HBM parity and are never evicted or corrupted."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    gc = _GreenCtxPair(0, 8)
    compute_sms = gc.compute_sms
    E = 32
    HOST_EXPERTS = 24  # Movable experts [0..23] in mapped host; pinned HBM-only [24..31]
    INIT_MOVABLE = 8
    PINNED_EXPERTS = tuple(range(HOST_EXPERTS, E))
    INITIAL_HBM = tuple(range(INIT_MOVABLE)) + PINNED_EXPERTS
    K_SPARE = 2
    HBM_SLOTS = len(INITIAL_HBM) + K_SPARE  # 18 slots
    HIDDEN, INTER, TOP_K, CAPACITY, ROWS, CLAMP = 1024, 512, 4, 128, 16, 7.0

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    tier = None
    mover = None
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        gen = torch.Generator(device="cpu").manual_seed(20261122)
        geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
        ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        ws_dyn = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

        # Reject missing pinned experts from initial_hbm_experts.
        with pytest.raises(ValueError, match="pinned HBM-only"):
            MegaMoeDynamicHostTier(
                ws_dyn.geometry,
                hbm_slots=HBM_SLOTS,
                initial_hbm_experts=tuple(range(len(INITIAL_HBM))),
                storage=sm103_storage,
                name="pinned_invalid",
                host_experts=HOST_EXPERTS,
            )

        full_weights = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
        shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

        tier = MegaMoeDynamicHostTier(
            ws_dyn.geometry,
            hbm_slots=HBM_SLOTS,
            initial_hbm_experts=INITIAL_HBM,
            storage=sm103_storage,
            name="pinned_layer0",
            host_experts=HOST_EXPERTS,
        )
        # Pinned HBM-only experts consume zero host rows.
        assert tier.rows == HOST_EXPERTS
        assert all(v.shape[0] == HOST_EXPERTS for v in (*tier.host_views, *tier.device_views))

        tier.copy_weights(
            type(full_weights)(
                (full_weights.gate_up[0].cpu()[:HOST_EXPERTS], full_weights.gate_up[1].cpu()[:HOST_EXPERTS]),
                (full_weights.down[0].cpu()[:HOST_EXPERTS], full_weights.down[1].cpu()[:HOST_EXPERTS]),
            )
        )
        hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
        for dst, src in zip((*hbm_pool.gate_up, *hbm_pool.down), (*full_weights.gate_up, *full_weights.down)):
            dst[:INIT_MOVABLE].copy_(src[:INIT_MOVABLE])
            dst[INIT_MOVABLE:len(INITIAL_HBM)].copy_(src[HOST_EXPERTS:E])
            dst[len(INITIAL_HBM):].zero_()
        pinned_snapshots = [p[INIT_MOVABLE:len(INITIAL_HBM)].clone() for p in (*hbm_pool.gate_up, *hbm_pool.down)]

        mover = SM103ExpertMoverRuntime(
            [(tier, hbm_pool)],
            storage=sm103_storage,
            kernel_cache=kernel_cache,
            worker_sm_count=gc.worker_sms,
            queue_capacity=256,
        )

        with PreparationSession(device=device, autotune=False) as session:
            ex = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
            probs = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)
            ref_op = PreparedMegaMoe(
                ex, ids, probs, pad, full_weights, shared_weights,
                workspace=ws_ref, clamp=CLAMP, session=session, name="pinned_ref",
                host_shared_first=True,
            )
            dyn_op = PreparedMegaMoe(
                ex, ids, probs, pad, hbm_pool, shared_weights,
                workspace=ws_dyn, clamp=CLAMP, session=session, name="pinned_dyn",
                host_tier=tier, host_shared_first=True, compute_sm_count=compute_sms,
            )

        # Build batches mixing pinned HBM-only (24..31), dynamic HBM (0..7), and cold host (8..23) experts.
        batches = []
        for step in range(12):
            x = (torch.randn((ROWS, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16).to(device)
            rows_ids = []
            for _ in range(ROWS):
                pinned_pick = torch.randint(HOST_EXPERTS, E, (1,), generator=gen)
                hot_pick = torch.randint(0, INIT_MOVABLE, (1,), generator=gen)
                cold_picks = torch.randperm(HOST_EXPERTS - INIT_MOVABLE, generator=gen)[:2] + INIT_MOVABLE
                rows_ids.append(torch.cat([pinned_pick, hot_pick, cold_picks]))
            ids = torch.stack(rows_ids).to(torch.int32).to(device).contiguous()
            probs = torch.softmax(torch.randn((ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)
            pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)
            if step % 2 == 1:
                pad[ROWS // 2:] = True
            golden = ref_op(x, ids, probs, pad).clone()
            batches.append((x, ids, probs, pad, golden))
        torch.cuda.synchronize()

        for idx, (x, ids, probs, pad, golden) in enumerate(batches):
            out = dyn_op(x, ids, probs, pad)
            torch.cuda.synchronize()
            assert torch.equal(out, golden), f"static pinned+dynamic mismatch at batch {idx}"

        cap_s = torch.cuda.Stream(device=device)
        x_buf = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        ids_buf = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
        probs_buf = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
        pad_buf = torch.zeros((ROWS,), dtype=torch.bool, device=device)
        out_buf = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        torch.cuda.synchronize()
        comp_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(comp_graph, pool=sm103_storage.graph_pool, stream=cap_s):
                out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        comp_graph.instantiate()

        # Enqueue hostile requests targeting pinned experts as victims or sources plus normal movable rotations.
        req_experts = list(range(INIT_MOVABLE, INIT_MOVABLE + 8)) + list(PINNED_EXPERTS) + list(range(HOST_EXPERTS))
        req_victims = list(PINNED_EXPERTS) + [-1] * (len(PINNED_EXPERTS) + HOST_EXPERTS)
        swap_layers = torch.zeros((len(req_experts),), dtype=torch.int32, device=device)
        swap_experts = torch.tensor(req_experts, dtype=torch.int32, device=device)
        swap_victims = torch.tensor(req_victims, dtype=torch.int32, device=device)
        enq_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(enq_graph, pool=sm103_storage.graph_pool, stream=cap_s):
                mover.enqueue_swap_batch(swap_layers, swap_experts, swap_victims)
        enq_graph.instantiate()

        worker_decl = mover.capture_worker_graph(name="pinned_mover", retain_graphs=True, stream=cap_s)
        cpu_batches = [(x.cpu(), ids.cpu(), probs.cpu(), pad.cpu(), g.cpu()) for x, ids, probs, pad, g in batches]
        host_out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16)
        mover.reset_stop_flag()
        torch.cuda.synchronize()

        gc.launch_graph(int(worker_decl.graph.raw_cuda_graph_exec()), worker=True)
        try:
            for _round in range(3):
                gc.launch_graph(int(enq_graph.raw_cuda_graph_exec()), worker=False)
                for x_cpu, ids_cpu, probs_cpu, pad_cpu, gold_cpu in cpu_batches:
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
                    assert torch.equal(host_out, gold_cpu), "live swap parity failure with pinned HBM-only experts"
            deadline = time.monotonic() + 5.0
            while mover.read_metrics().swaps_completed < 20 and time.monotonic() < deadline:
                time.sleep(0.001)
        finally:
            gc.sync_compute()
            mover.signal_stop()
            gc.sync_worker()
            gc.set_primary()
            worker_decl.graph.reset()
            enq_graph.reset()
            comp_graph.reset()

        metrics = mover.read_metrics()
        assert metrics.swaps_completed >= 20, f"expected active movable swaps, got {metrics}"
        final_map = tier.live_slot_map.cpu().tolist()
        final_owner = tier.slot_owner.cpu().tolist()
        for idx, pinned_expert in enumerate(PINNED_EXPERTS):
            expected_slot = INIT_MOVABLE + idx
            assert final_map[pinned_expert] == expected_slot, f"pinned expert {pinned_expert} moved or evicted"
            assert final_owner[expected_slot] == pinned_expert, f"pinned slot {expected_slot} overwritten"
        for plane, snap in zip((*hbm_pool.gate_up, *hbm_pool.down), pinned_snapshots):
            assert torch.equal(plane[INIT_MOVABLE:len(INITIAL_HBM)], snap), "pinned HBM weight bytes corrupted"
        resident_from_map = {e: s for e, s in enumerate(final_map) if s >= 0}
        resident_from_owner = {e: s for s, e in enumerate(final_owner) if e >= 0}
        assert resident_from_map == resident_from_owner
        assert len(resident_from_map) == len(INITIAL_HBM)
        assert final_owner.count(-1) == K_SPARE
    finally:
        if mover is not None:
            mover.close()
        if tier is not None:
            for r in tier.regions:
                r.close()
        gc.close()


def test_sm103_all_hbm_megamoe_compute_sm_count_parity(sm103_storage):
    """Verify an all-HBM MegaMoE layer with compute_sm_count > 0 uses the clamped host binding and matches unclamped parity."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from b12x.preparation.types import plan_from_handle, require_prepared
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe

    device = torch.device("cuda", 0)
    gc = _GreenCtxPair(0, 8)
    compute_sms = gc.compute_sms
    assert compute_sms > 0 and compute_sms % 2 == 0
    E, HIDDEN, INTER, TOP_K, CAPACITY, CLAMP = 32, 1024, 512, 4, 128, 7.0
    ROW_SHAPES = (1, 16, 128)
    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    try:
        gc.set_primary()
        # Set ambient deep_gemm SM count to compute_sms to also verify workspace/stock path preserve it.
        deep_gemm.set_num_sms(compute_sms)
        gen = torch.Generator(device="cpu").manual_seed(20261123)
        geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
        ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        ws_clamped = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        assert deep_gemm.get_num_sms() == compute_sms

        weights = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
        shared = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

        ref_ops = {}
        clamped_ops = {}
        with PreparationSession(device=device, autotune=False) as session:
            for rows in ROW_SHAPES:
                ex = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=device)
                ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
                probs = torch.full((rows, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
                pad = torch.zeros((rows,), dtype=torch.bool, device=device)
                ref_ops[rows] = PreparedMegaMoe(
                    ex, ids, probs, pad, weights, shared,
                    workspace=ws_ref, clamp=CLAMP, session=session, name=f"all_hbm_ref.{rows}",
                )
                clamped_ops[rows] = PreparedMegaMoe(
                    ex, ids, probs, pad, weights, shared,
                    workspace=ws_clamped, clamp=CLAMP, session=session, name=f"all_hbm_clamped.{rows}",
                    compute_sm_count=compute_sms,
                )

        cap_s = torch.cuda.Stream(device=device)
        for rows in ROW_SHAPES:
            clamped_op = clamped_ops[rows]
            ref_op = ref_ops[rows]
            assert clamped_op.host_tier is None and clamped_op.control is None
            clamped_state = require_prepared(plan_from_handle(clamped_op.plan.handle), clamped_op.plan.component_id, device)
            ref_state = require_prepared(plan_from_handle(ref_op.plan.handle), ref_op.plan.component_id, device)
            assert ref_state.query.compute_sm_count == 0
            assert clamped_state.query.compute_sm_count == compute_sms

            x = (torch.randn((rows, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16).to(device)
            ids = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(rows)]).to(torch.int32).to(device).contiguous()
            probs = torch.softmax(torch.randn((rows, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)
            pad = torch.zeros((rows,), dtype=torch.bool, device=device)
            if rows >= 8:
                pad[rows // 2:] = True

            golden = ref_op(x, ids, probs, pad).clone()
            eager_clamped = clamped_op(x, ids, probs, pad).clone()
            torch.cuda.synchronize()
            assert deep_gemm.get_num_sms() == compute_sms
            assert torch.equal(eager_clamped, golden), f"all-HBM clamped SM mismatch (rows={rows})"

            # Replay the clamped all-HBM graph inside the partitioned compute green context.
            out_buf = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.stream(cap_s):
                with torch.cuda.graph(g, pool=sm103_storage.graph_pool, stream=cap_s):
                    out_buf.copy_(clamped_op(x, ids, probs, pad))
            g.instantiate()
            torch.cuda.synchronize()
            try:
                gc.launch_graph(int(g.raw_cuda_graph_exec()), worker=False)
                gc.sync_compute()
                gc.set_primary()
                assert torch.equal(out_buf, golden), f"all-HBM clamped green-context replay mismatch (rows={rows})"
            finally:
                gc.set_primary()
                g.reset()
    finally:
        gc.close()


def test_sm103_concurrent_two_producer_cas_enqueue_parity(sm103_storage):
    """Verify two concurrent producer streams enqueueing swaps simultaneously via CAS complete without loss or corruption."""
    import struct
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    gc = _GreenCtxPair(0, 8)
    compute_sms = gc.compute_sms
    E, S_ACTIVE, K_SPARE = 128, 14, 2
    HBM_SLOTS = S_ACTIVE + K_SPARE
    HIDDEN, INTER, TOP_K, CAPACITY, ROWS, CLAMP = 512, 256, 4, 128, 16, 7.0
    PER_PRODUCER_PER_ROUND = 48
    ROUNDS = 2
    TOTAL_SWAPS = 2 * PER_PRODUCER_PER_ROUND * ROUNDS  # 192 distinct cold swaps across 2 layers

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    tiers = []
    mover = None
    stream_a = gc.create_compute_stream()
    stream_b = gc.create_compute_stream()
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        gen = torch.Generator(device="cpu").manual_seed(20261124)
        geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
        ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        ws_dyn = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        shared = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

        full_weights = []
        hbm_pools = []
        for layer_idx in range(2):
            fw = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
            tier = MegaMoeDynamicHostTier(
                ws_dyn.geometry,
                hbm_slots=HBM_SLOTS,
                initial_hbm_experts=tuple(range(S_ACTIVE)),
                storage=sm103_storage,
                name=f"mpsc_layer{layer_idx}",
            )
            tier.copy_weights(
                type(fw)(
                    (fw.gate_up[0].cpu(), fw.gate_up[1].cpu()),
                    (fw.down[0].cpu(), fw.down[1].cpu()),
                )
            )
            pool = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
            for dst, src in zip((*pool.gate_up, *pool.down), (*fw.gate_up, *fw.down)):
                dst[:S_ACTIVE].copy_(src[:S_ACTIVE])
                dst[S_ACTIVE:].zero_()
            full_weights.append(fw)
            tiers.append(tier)
            hbm_pools.append(pool)

        mover = SM103ExpertMoverRuntime(
            list(zip(tiers, hbm_pools)),
            storage=sm103_storage,
            kernel_cache=kernel_cache,
            worker_sm_count=gc.worker_sms,
            queue_capacity=256,
        )

        ref_ops = []
        dyn_ops = []
        with PreparationSession(device=device, autotune=False) as session:
            ex = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
            probs = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)
            for layer_idx in range(2):
                ref_ops.append(
                    PreparedMegaMoe(
                        ex, ids, probs, pad, full_weights[layer_idx], shared,
                        workspace=ws_ref, clamp=CLAMP, session=session, name=f"mpsc_ref_{layer_idx}",
                        host_shared_first=True,
                    )
                )
                dyn_ops.append(
                    PreparedMegaMoe(
                        ex, ids, probs, pad, hbm_pools[layer_idx], shared,
                        workspace=ws_dyn, clamp=CLAMP, session=session, name=f"mpsc_dyn_{layer_idx}",
                        host_tier=tiers[layer_idx], host_shared_first=True, compute_sm_count=compute_sms,
                    )
                )

        x_buf = (torch.randn((ROWS, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16).to(device)
        ids_buf = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(ROWS)]).to(torch.int32).to(device).contiguous()
        probs_buf = torch.softmax(torch.randn((ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)
        pad_buf = torch.zeros((ROWS,), dtype=torch.bool, device=device)
        pad_buf[ROWS // 2:] = True
        goldens = [op(x_buf, ids_buf, probs_buf, pad_buf).clone().cpu() for op in ref_ops]
        out_bufs = [torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device) for _ in range(2)]
        for i in range(2):
            out_bufs[i].copy_(dyn_ops[i](x_buf, ids_buf, probs_buf, pad_buf))
        torch.cuda.synchronize()

        cap_s = torch.cuda.Stream(device=device)
        comp_graphs = []
        for i in range(2):
            g = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.stream(cap_s):
                with torch.cuda.graph(g, pool=sm103_storage.graph_pool, stream=cap_s):
                    out_bufs[i].copy_(dyn_ops[i](x_buf, ids_buf, probs_buf, pad_buf))
            g.instantiate()
            comp_graphs.append(g)

        layers_a = torch.zeros((PER_PRODUCER_PER_ROUND,), dtype=torch.int32, device=device)
        experts_a = torch.arange(S_ACTIVE, S_ACTIVE + PER_PRODUCER_PER_ROUND, dtype=torch.int32, device=device)
        victims_a = torch.full((PER_PRODUCER_PER_ROUND,), -1, dtype=torch.int32, device=device)
        enq_a = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(enq_a, pool=sm103_storage.graph_pool, stream=cap_s):
                mover.enqueue_swap_batch(layers_a, experts_a, victims_a)
        enq_a.instantiate()

        layers_b = torch.ones((PER_PRODUCER_PER_ROUND,), dtype=torch.int32, device=device)
        experts_b = torch.arange(S_ACTIVE, S_ACTIVE + PER_PRODUCER_PER_ROUND, dtype=torch.int32, device=device)
        victims_b = torch.full((PER_PRODUCER_PER_ROUND,), -1, dtype=torch.int32, device=device)
        enq_b = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(enq_b, pool=sm103_storage.graph_pool, stream=cap_s):
                mover.enqueue_swap_batch(layers_b, experts_b, victims_b)
        enq_b.instantiate()

        worker_decl = mover.capture_worker_graph(name="mpsc_mover", retain_graphs=True, stream=cap_s)
        host_out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16)
        mover.reset_stop_flag()
        torch.cuda.synchronize()

        gc.launch_graph(int(worker_decl.graph.raw_cuda_graph_exec()), worker=True)
        try:
            for round_idx in range(ROUNDS):
                start_e = S_ACTIVE + round_idx * PER_PRODUCER_PER_ROUND
                batch_cpu = torch.arange(start_e, start_e + PER_PRODUCER_PER_ROUND, dtype=torch.int32)
                gc.set_compute()
                gc._call(
                    "cuMemcpyHtoDAsync_v2",
                    C.c_uint64(experts_a.data_ptr()),
                    C.c_void_p(batch_cpu.data_ptr()),
                    C.c_size_t(batch_cpu.numel() * 4),
                    stream_a,
                )
                gc._call(
                    "cuMemcpyHtoDAsync_v2",
                    C.c_uint64(experts_b.data_ptr()),
                    C.c_void_p(batch_cpu.data_ptr()),
                    C.c_size_t(batch_cpu.numel() * 4),
                    stream_b,
                )
                # Launch both producers concurrently on separate compute streams without syncing between them.
                gc.launch_graph_on_stream(int(enq_a.raw_cuda_graph_exec()), stream_a)
                gc.launch_graph_on_stream(int(enq_b.raw_cuda_graph_exec()), stream_b)

                # Concurrently replay both MegaMoE layers on the primary compute stream while swaps drain.
                for layer_idx in range(2):
                    gc.launch_graph(int(comp_graphs[layer_idx].raw_cuda_graph_exec()), worker=False)
                    gc.set_compute()
                    gc._call(
                        "cuMemcpyDtoHAsync_v2",
                        C.c_void_p(host_out.data_ptr()),
                        C.c_uint64(out_bufs[layer_idx].data_ptr()),
                        C.c_size_t(host_out.numel() * host_out.element_size()),
                        gc.c_stream,
                    )
                    gc.sync_compute()
                    assert torch.equal(host_out, goldens[layer_idx]), (
                        f"MPSC concurrent enqueue parity mismatch (round={round_idx}, layer={layer_idx})"
                    )
                gc.sync_stream(stream_a)
                gc.sync_stream(stream_b)
                target_swaps = (round_idx + 1) * 2 * PER_PRODUCER_PER_ROUND
                deadline = time.monotonic() + 5.0
                while mover.read_metrics().swaps_completed < target_swaps and time.monotonic() < deadline:
                    time.sleep(0.0005)
                assert mover.read_metrics().swaps_completed == target_swaps, (
                    f"lost or stalled concurrent enqueue in round {round_idx}: {mover.read_metrics()}"
                )

            # Final parity check on both layers after all concurrent swaps have completed.
            for layer_idx in range(2):
                gc.launch_graph(int(comp_graphs[layer_idx].raw_cuda_graph_exec()), worker=False)
                gc.set_compute()
                gc._call(
                    "cuMemcpyDtoHAsync_v2",
                    C.c_void_p(host_out.data_ptr()),
                    C.c_uint64(out_bufs[layer_idx].data_ptr()),
                    C.c_size_t(host_out.numel() * host_out.element_size()),
                    gc.c_stream,
                )
                gc.sync_compute()
                assert torch.equal(host_out, goldens[layer_idx]), f"post-drain parity mismatch on layer {layer_idx}"
        finally:
            gc.sync_stream(stream_a)
            gc.sync_stream(stream_b)
            gc.sync_compute()
            mover.signal_stop()
            gc.sync_worker()
            gc.set_primary()
            worker_decl.graph.reset()
            enq_a.reset()
            enq_b.reset()
            for g in comp_graphs:
                g.reset()

        head, tail, mask = struct.unpack("<III", bytes(mover.queue_header[:12].cpu().tolist()))
        metrics = mover.read_metrics()
        assert head == TOTAL_SWAPS and tail == TOTAL_SWAPS and mask == 255, (
            f"unexpected ring counters after concurrent MPSC drain: head={head} tail={tail} mask={mask}"
        )
        assert metrics.swaps_completed == TOTAL_SWAPS, f"expected {TOTAL_SWAPS} swaps, got {metrics}"
        assert metrics.skipped_already_resident == 0, f"unexpected duplicate/overwritten entries: {metrics}"
        assert metrics.bytes_copied == TOTAL_SWAPS * sum(tiers[0].plane_row_bytes)
        for layer_idx, tier in enumerate(tiers):
            final_map = tier.live_slot_map.cpu().tolist()
            final_owner = tier.slot_owner.cpu().tolist()
            resident_from_map = {e: s for e, s in enumerate(final_map) if s >= 0}
            resident_from_owner = {e: s for s, e in enumerate(final_owner) if e >= 0}
            assert resident_from_map == resident_from_owner, f"layer {layer_idx} slot_map/slot_owner diverged"
            assert len(resident_from_map) == S_ACTIVE
            assert final_owner.count(-1) == K_SPARE
    finally:
        gc.destroy_stream(stream_a)
        gc.destroy_stream(stream_b)
        if mover is not None:
            mover.close()
        for tier in tiers:
            for r in tier.regions:
                r.close()
        gc.close()
