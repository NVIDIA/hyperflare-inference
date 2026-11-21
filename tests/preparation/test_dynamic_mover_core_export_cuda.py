# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export a miniature DeepSeek V4.1 SM103 core with dynamic_mover and serve it in fresh flare run processes."""

from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import struct
import subprocess

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest
import torch
from safetensors.torch import save_file

from ignition.components.expert_placement import ExpertPlacement
from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier, MegaMoeLayerWeights
from ignition.components.megamoe_weights import allocate_mxfp4_weights
from ignition.components.sm103_expert_mover import (
    SM103ExpertMoverRuntime,
    query_green_context_compute_sms,
)
from ignition.preparation.core_snapshot import export_options
from ignition.preparation.core_storage import CoreStorage
from ignition.preparation.model_core import export_core
from ignition.preparation.module_images import ModuleImages
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.protection import Publisher
from ignition.preparation.runtime_core import FileSource
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.bootstrap import PreparationConfig, _assemble_reference_prepared_model
from ignition.recipes.deepseek41.reference_engram import ReferenceEngramDelivery
from ignition.recipes.deepseek41.reference_experts import ReferenceExperts
from ignition.recipes.deepseek41.reference_model import ReferenceExpertPreparation, ReferenceModel
from ignition.recipes.deepseek41.sm103 import DynamicExpertMoverConfig, SM103Config
from tests.preparation.test_reference_core_export_cuda import (
    _allocate_synthetic_mxfp4,
    _allocate_synthetic_shared_fp8,
    _materialize_synthetic_ordinary,
    _verify_exported_mini_core,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or os.environ.get("FLARE_RUN_REFERENCE_CORE_EXPORT_CUDA_TESTS") != "1",
    reason="requires an assigned SM103 GPU, DeepSeek V4.1 metadata, and release flare CLI",
)


def _build_mini_dynamic_reference_experts(
    placements,
    *,
    target_rows,
    draft_rows,
    storage,
    kernel_cache,
    session,
    device,
    dynamic_mover,
    compute_sm_count,
):
    hidden, intermediate = 5120, 2304
    generator = torch.Generator(device=device).manual_seed(7711)
    workspaces, modules, owners = {}, {}, {}
    dynamic_layers = []
    for name, placement in placements.items():
        draft = name.startswith("mtp.")
        rows = draft_rows if draft else target_rows
        top_k = 3 if draft else 6
        use_dynamic = (not draft) and bool(placement.host)
        physical_experts = placement.experts if use_dynamic else placement.physical_rows
        geometry = MegaMoeGeometry(physical_experts, max(rows), top_k, hidden, intermediate, 1)
        if geometry not in workspaces:
            workspaces[geometry] = MegaMoeWorkspace(geometry, device=device, kernel_cache=kernel_cache)
        ws = workspaces[geometry]
        shared = _allocate_synthetic_shared_fp8(hidden, intermediate, device)
        if use_dynamic:
            hbm_slots = len(placement.hbm) + dynamic_mover.spare_slots
            tier = MegaMoeDynamicHostTier(
                ws.geometry,
                hbm_slots=hbm_slots,
                initial_hbm_experts=placement.hbm,
                storage=storage,
                name=name,
            )
            for dst in tier.host_views:
                if dst.dtype == torch.int32:
                    dst.fill_(0x78787878)
                else:
                    dst.fill_(0x11)
            routed = allocate_mxfp4_weights(hbm_slots, hidden, intermediate, device=device)
            for dst, src_cpu in zip((*routed.gate_up, *routed.down), tier.host_views):
                for slot, expert_id in enumerate(placement.hbm):
                    dst[slot].copy_(src_cpu[expert_id].to(device))
                dst[len(placement.hbm):].zero_()
            expert_map = persistent_copy(torch.arange(placement.experts, dtype=torch.int32, device=device))
            record_numeric_tensor(expert_map)
            layer_weights = MegaMoeLayerWeights(
                routed, shared, ws, tier, expert_map,
                tuple(range(placement.experts)), placement.hbm,
            )
            dynamic_layers.append((tier, routed))
        else:
            routed = _allocate_synthetic_mxfp4(len(placement.hbm), hidden, intermediate, device)
            expert_map = persistent_copy(torch.arange(placement.experts, dtype=torch.int32, device=device))
            record_numeric_tensor(expert_map)
            layer_weights = MegaMoeLayerWeights(
                routed, shared, ws, None, expert_map, (), placement.hbm,
            )
        owners[name] = layer_weights
        gate_w = persistent_copy(
            torch.randn((placement.experts, hidden), dtype=torch.bfloat16, device=device, generator=generator) * 0.05
        )
        gate_b = persistent_copy(torch.linspace(-0.2, 0.2, placement.experts, dtype=torch.float32, device=device))
        gate_vl = persistent_copy(torch.linspace(0.2, -0.2, placement.experts, dtype=torch.float32, device=device))
        modules[name] = ReferenceExperts.prepare(
            layer_weights,
            rows,
            weight=gate_w,
            bias=gate_b,
            image_bias=gate_vl,
            scaling=1.5,
            scoring_func="sqrtsoftplus",
            clamp=10.0,
            session=session,
            name=name,
            compute_sm_count=compute_sm_count,
        )
    mover = SM103ExpertMoverRuntime(
        dynamic_layers,
        storage=storage,
        kernel_cache=kernel_cache,
        worker_sm_count=dynamic_mover.worker_sm_count,
        queue_capacity=dynamic_mover.queue_capacity,
    )
    mover.attach_predictors(dynamic_mover.predictor_config())
    return ReferenceExpertPreparation(modules, owners, mover=mover)


@torch.no_grad()
def test_miniature_dynamic_mover_core_export_and_fresh_process_flare_run(tmp_path):
    """Export a miniature DeepSeek V4.1 core with dynamic_mover + Execution.persistent_worker and serve via flare run."""
    import deep_gemm
    from deepinfer.hugging_face.config import model_config_from_dict
    from deepinfer.model_executor.execution_settings import ExecutionSettings
    from deepinfer.model_executor.models.v4_modules.builder import V41Builder
    from deepinfer.model_executor.parallel_utils.parallel_state import PG

    source_snapshot = Path(os.environ["FLARE_DEEPSEEK41_CSF_SNAPSHOT"]).resolve(strict=True)
    repo_root = Path(__file__).resolve().parents[2]
    flare_bin = repo_root / "flare-cli" / "target" / "release" / "flare"
    assert flare_bin.is_file(), f"missing release flare binary at {flare_bin}"

    if PG.world is None:
        PG.initialize_local()

    meta_dir = tmp_path / "mini_checkpoint"
    meta_dir.mkdir()
    for asset in source_snapshot.iterdir():
        if asset.name != "config.json" and not asset.name.endswith(".safetensors") and asset.name != "model.safetensors.index.json":
            (meta_dir / asset.name).symlink_to(asset)

    raw = json.loads((source_snapshot / "config.json").read_text())
    text = raw["text_config"]
    text.update(
        num_hidden_layers=5,
        compress_ratios=[0, 0, 2, 2, 1, 0],
        kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4],
        candidate_source_layer_id=4,
        candidate_topk_blocks=16,
        index_topk=64,
        engram_layer_ids=[1],
        engram_vocab_size=64,
        engram_num_embeddings=[0],
        n_routed_experts=8,
        num_experts_per_tok=6,
        num_nextn_predict_layers=1,
        dspark_target_layer_ids=[1, 2, 4],
        dspark_n_routed_experts=8,
        dspark_num_experts_per_tok=3,
    )
    raw["vision_config"]["num_hidden_layers"] = 2
    (meta_dir / "config.json").write_text(json.dumps(raw, indent=2))

    decoded_raw = dict(
        raw,
        quantization_config=dict(
            quant_method="fp8",
            expert_dtype="fp4",
            activation_scheme="dynamic",
            weight_block_size=[32, 32],
            scale_fmt="ue8m0",
        ),
    )
    parsed_cfg = model_config_from_dict(decoded_raw, hf_dir=meta_dir)
    hash_meta = ReferenceEngramDelivery.export_hash_metadata(parsed_cfg.gpt)
    layer1_hash = hash_meta["table_hashes"][1]
    engram_rows = layer1_hash["offsets"][-1] + layer1_hash["primes"][-1]
    engram_width = 256
    engram_file = meta_dir / "engram.safetensors"
    save_file(
        {
            "layers.1.engram.embed.weight": torch.zeros((engram_rows, engram_width), dtype=torch.uint8),
            "layers.1.engram.embed.scale": torch.full((engram_rows, engram_width // 32), 120, dtype=torch.uint8),
        },
        str(engram_file),
    )
    with engram_file.open("rb") as fh:
        header_len, = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        data_base = 8 + header_len
    w_off = data_base + header["layers.1.engram.embed.weight"]["data_offsets"][0]
    s_off = data_base + header["layers.1.engram.embed.scale"]["data_offsets"][0]

    # 8 canonical experts: 4 initial HBM + 2 spare HBM slots = 6 HBM slots, 4 initial cold experts on host.
    target_placement = ExpertPlacement(8, (4, 5, 6, 7), (0, 1, 2, 3))
    draft_placement = ExpertPlacement(8, (), tuple(range(8)))
    placements = tuple((f"layers.{i}.ffn", target_placement) for i in range(5)) + (("mtp.0.ffn", draft_placement),)
    dynamic_mover = DynamicExpertMoverConfig(
        worker_sm_count=8,
        spare_slots=2,
        queue_capacity=1024,
        decay_shift=2,
        promote_threshold=256,
        hysteresis=0,
        cooldown_steps=1,
        max_enqueues=2,
        max_predict_rows=64,
    )
    sm103 = SM103Config(
        placements,
        host_lane_clusters=0,
        execution="reference",
        context=1024,
        dynamic_mover=dynamic_mover,
    )
    cache_override = os.environ.get("FLARE_TEST_REFERENCE_MINI_CACHE_DIR")
    kernel_cache = Path(cache_override) if cache_override else tmp_path / "kernel_cache"
    kernel_cache.mkdir(parents=True, exist_ok=True)

    prep_cfg = PreparationConfig(
        devices=(0,),
        context=1024,
        prefill_chunk_size=1024,
        max_active=2,
        image_concurrency=1,
        snapshot_budget_bytes=64 << 20,
        headroom_bytes=512 << 20,
        kv_pages=64,
        compile_workers=12,
        autotune_effort="off",
        compile_decoder=True,
        retain_graphs=True,
        kernel_cache=kernel_cache,
        dspark=True,
        draft_head_quantization="nvfp4",
        markov_head_quantization="fp8",
        speculative_widths=(1, 2, 3, 4),
        speculative_token_capacity=8,
        speculative_graph_budget=16,
        sm103=sm103,
    )

    class _ParameterCarrier:
        def __init__(self, tree, cfg_obj, metadata_root):
            self.model = tree
            self.config = cfg_obj
            self.checkpoint = type("CP", (), {"source": type("Src", (), {"metadata_path": metadata_root})()})()
            self.loaded = True
            self.assembled = False

    core_dir = tmp_path / "exported_dynamic_core"
    cache_dir = tmp_path / "core_cache"
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    pub = Publisher(
        Ed25519PrivateKey.from_private_bytes(bytes([43]) * 32),
        "local-development",
        "local-development",
        development=True,
    )

    _, compute_sm_count = query_green_context_compute_sms(0, dynamic_mover.worker_sm_count)
    deep_gemm.set_num_sms(compute_sm_count)

    recorder = ModuleImages()
    with export_options(cache_dir), recorder:
        torch.cuda.set_device(0)
        storage = CoreStorage(world=1)
        with storage:
            session = RecordedPreparationSession(
                compile_workers=12,
                autotune_effort="off",
                device=torch.device("cuda", 0),
                checkpoint="mini-reference-dynamic-mover",
                workload_schema="deepseek41-reference-mini-dynamic",
                numerical_policy="reference-checkpoint",
                headroom_bytes=prep_cfg.headroom_bytes,
                benchmark_storage=storage,
            )
            session.__enter__()
            with storage.transient():
                with torch.device("meta"):
                    raw_tree = V41Builder(parsed_cfg).build_model(vision_device="meta")
                _materialize_synthetic_ordinary(raw_tree, parsed_cfg, torch.device("cuda", 0))
                target_rows = tuple(sorted(
                    set(prep_cfg.prefill_capacities)
                    | set(prep_cfg.batch_capacities)
                    | {b for b, _ in prep_cfg.speculative_shapes}
                    | {t for _, _, t in prep_cfg.packed_speculative_depth_shapes}
                ))
                draft_rows = tuple(sorted({b * (w - 1) for b, w, _ in prep_cfg.packed_speculative_depth_shapes}))
                experts = _build_mini_dynamic_reference_experts(
                    dict(placements),
                    target_rows=target_rows,
                    draft_rows=draft_rows,
                    storage=storage,
                    kernel_cache=kernel_cache,
                    session=session,
                    device=torch.device("cuda", 0),
                    dynamic_mover=dynamic_mover,
                    compute_sm_count=compute_sm_count,
                )
                sequence = torch.zeros(1, dtype=torch.int64, device="cuda:0")
                frame = torch.zeros(1, dtype=torch.int32, device="cuda:0")
                delivery = ReferenceEngramDelivery(
                    {
                        1: (
                            FileSource(engram_file, w_off, engram_rows * engram_width),
                            FileSource(engram_file, s_off, engram_rows * (engram_width // 32)),
                        )
                    },
                    vocabulary=parsed_cfg.gpt.vocab_size,
                    heads=24,
                    width=engram_width,
                    capacity=max(prep_cfg.prefill_chunk_size, prep_cfg.decode_tokens),
                    sequence=sequence,
                    frame=frame,
                    host_allocator=storage.host_allocate,
                    **hash_meta,
                )
                execution = ExecutionSettings(
                    draft_tokens=3,
                    batch_invariant=False,
                    yoco_prefill_truncation="off",
                    index_memory_limit=16 << 20,
                    sampler_topk=0,
                    confidence_temperature=1.0,
                )
                carrier = _ParameterCarrier(raw_tree, parsed_cfg, meta_dir)
                ref_model = ReferenceModel(
                    carrier,
                    experts,
                    delivery,
                    execution=execution,
                    context=prep_cfg.context,
                    request_capacity=prep_cfg.request_capacity,
                    prefill_tokens=prep_cfg.prefill_chunk_size,
                    verification_tokens=prep_cfg.request_capacity * 4,
                )
                del raw_tree, carrier

            prepared = _assemble_reference_prepared_model(
                ref_model,
                config=prep_cfg,
                metadata_path=meta_dir,
                snapshot_name="mini-deepseek41-dynamic-mover",
                identity="1" * 64,
                session=session,
                core_storage=storage,
                sequence=sequence,
                frame=frame,
                weights_bytes=1 << 20,
                tier_bytes={"hbm": 1 << 20, "host": 1 << 20},
                has_vision=True,
            )

        with prepared:
            rebuild = dict(
                version=1,
                recipe="deepseek41",
                source_identity="1" * 64,
                checkpoint_repository="deepseek-ai/DeepSeek-V4.1-Flash-1017",
                checkpoint_revision="",
                local_checkpoint=True,
                tensor_parallel=1,
                max_concurrency=2,
                prefill_chunk=128,
                image_concurrency=1,
                context_parallel=False,
                context_parallel_size=0,
                stream_overlap=True,
                mixed_prefill=False,
                mla_compute_mode="bf16",
                roce_ring=False,
                roce_chain=False,
                online_mxfp8=False,
                online_mxfp8_scope="all",
                language_model_only=False,
                compiled_forwards=True,
                reserve_bytes=0,
                collective_tuning="static",
                sm103_preparation_json=json.dumps(sm103.to_dict(), separators=(",", ":"), sort_keys=True),
                speculation=dict(
                    draft_tokens=3,
                    mtp_only=False,
                    adaptive=True,
                    token_capacity=8,
                    packed_verification=True,
                    draft_prefix_selection="gpu",
                    vocabulary_parallel_sampling=False,
                    graph_budget=16,
                ),
            )
            options = dict(
                model_id="mini-deepseek41-dynamic-mover",
                served_names=["DeepSeek-V4.1-Flash"],
                reserve_bytes=0,
                queue_capacity=8,
                frontend_workers=2,
                max_body_bytes=32 << 20,
                context_hint="miniature dynamic-mover test core",
                capacity_hint="--max-concurrency sets the request limit",
                rebuild_manifest=rebuild,
            )
            export_core(
                prepared,
                recorder,
                storage,
                core_dir,
                options=options,
                publisher=pub,
                staging=staging_dir,
            )

    del prepared, ref_model, experts, delivery, sequence, frame, session
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Verify persistent_worker graph, control buffer, and all 20 host weight allocations are exported.
    inspect_proc = subprocess.run(
        [str(flare_bin), "inspect", "--all", "--json", "--unverified", str(core_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    inspected = json.loads(inspect_proc.stdout)
    rank0 = inspected["ranks"][0]
    assert any(g["name"] == "sm103_expert_mover" for g in rank0["graphs"]), "missing persistent worker graph"
    assert any(b["name"] == "sm103_mover_control" for b in rank0["execution_buffers"]), "missing mover control buffer"
    host_allocs = [a for a in rank0["allocations"] if a["memory"] == 1]
    assert len(host_allocs) >= 21, f"expected >=21 mapped host allocations (20 expert planes + control), got {len(host_allocs)}"
    assert (core_dir / "model.safetensors").stat().st_size < (64 << 20), "primary shard must hold metadata only"

    # Serve cleanly in fresh flare run subprocesses (ordinary + prefix reuse + compacted prefill + vision + adaptive MTP).
    _verify_exported_mini_core(core_dir, cache_dir, pub, flare_bin, tmp_path)
