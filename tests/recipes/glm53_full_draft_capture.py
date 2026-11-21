# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Capture two prepared draft steps, per-stage operands, repeats, and declared held-out prefixes."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.distributed as dist

from .glm53_full_support import reference_index_key


@torch.inference_mode()
def capture_rank(rank, rendezvous, snapshot, fixtures, destination):
    from ignition import runtime_patches

    runtime_patches.apply()
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.collectives import (
        CollectivePreparation,
        create_all_gather_runtime,
        create_all_reduce_runtime,
    )
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.glm53_full.attention import ExampleBuffers, Selection
    from ignition.recipes.glm53_full.mtp import PreparedDraftLayer, PreparedProposals
    from ignition.recipes.glm53_full.pool import ContinuationPool
    from ignition.recipes.glm53_full.weights import ModelWeights

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    timeout = timedelta(minutes=30)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=8, timeout=timeout
    )
    try:
        coordinator = CollectivePreparation(dist.group.WORLD, timeout=timeout)
        weights = ModelWeights.load(
            inspect_indexed_checkpoint(Path(snapshot)),
            device,
            shard=TensorParallelShard(rank, 8),
            layers=1,
            draft=True,
        )
        reduction = create_all_reduce_runtime(
            dist.group.WORLD, device, max_tokens=128, hidden_size=6144
        )
        gather = create_all_gather_runtime(
            dist.group.WORLD, device, max_rows=1, columns=weights.head.shape[0] * 8
        )
        pool = ContinuationPool.for_model(weights.config, 2, 11, device, draft=True)
        arena = ScratchArena(device, overlap=False)
        fixture = torch.load(Path(fixtures) / "long.pt", mmap=True)
        table = torch.arange(11, device=device, dtype=torch.int32)[None]
        slots = torch.zeros(1, device=device, dtype=torch.long)
        positions = torch.arange(128, device=device)
        requests = torch.zeros(128, device=device, dtype=torch.int32)
        inputs = SimpleNamespace(page_tables=table, row_slot_ids=slots)
        active = torch.ones(1, device=device, dtype=torch.bool)
        with RecordedPreparationSession(
            device=device,
            checkpoint="glm53-full-test",
            workload_schema="glm53-draft-arithmetic-stages",
            numerical_policy="default",
            coordinator=coordinator,
        ) as session:
            session.register_scratch(arena)
            options = dict(
                runtime=reduction, coordinator=coordinator, session=session, workspace=arena
            )
            context = PreparedDraftLayer(
                weights,
                pool,
                positions,
                requests,
                table,
                cache_only=True,
                name="context",
                buffers=ExampleBuffers(device),
                **options,
            )
            op = PreparedProposals(
                weights, pool, inputs, 1, 2, vocabulary_runtime=gather, name="proposals", **options
            )
            from ignition.components.sampling import DeviceRequestState, RequestAdmission
            from ignition.components.draft_sampling import PreparedDraftDistribution

            sample_state = DeviceRequestState.allocate(device, vocab_size=154880, capacity=1)
            sample_state.admit(0, RequestAdmission(generation=1, seed=17, top_k=-1, top_p=1), [])
            sample_prefix = torch.full((1, 2), 369, dtype=torch.int64, device=device)
            samplers = [
                PreparedDraftDistribution(
                    torch.zeros((1, 1, 154880), device=device),
                    sample_state,
                    slots.to(torch.int32),
                    torch.ones(1, dtype=torch.int64, device=device),
                    torch.ones(1, dtype=torch.int32, device=device),
                    sample_prefix,
                    seed=torch.tensor([17], device=device),
                    offset=torch.zeros(1, dtype=torch.int64, device=device),
                    sampled=torch.zeros(1, dtype=torch.int64, device=device),
                    prefix_offset=step,
                )
                for step in range(2)
            ]
            for sampler in samplers:
                sampler.prepare()
            session.tune()
            arena.release_preparation()
            plan = op.layer.layer.attention.attention.plan
            attention_configuration = plan.contract.config_payload(plan.selection.config).to_dict()
            observed = {}
            tag = [""]

            def observe(owner, field, name):
                original = getattr(owner, field)

                def capture(*args, **kwargs):
                    result = original(*args, **kwargs)
                    observed[name] = result.detach().cpu().clone()
                    if args and torch.is_tensor(args[0]):
                        observed[name + "_input"] = args[0].detach().cpu().clone()
                    return result

                setattr(owner, field, capture)

            observe(op.layer, "projection", "joined")
            layer = op.layer.layer
            attention = layer.attention
            for key in ("input_proj", "q_b_proj", "indexer.weights_proj"):
                original = attention.projections[key]

                def projection(source, operation=original, key=key):
                    result = operation(source)
                    observed[key] = result.cpu().clone()
                    observed[key + "_input"] = source.cpu().clone()
                    return result

                attention.projections[key] = projection
            for field, name in [
                ("absorb", "absorbed_query"),
                ("attention", "attended_latent"),
                ("expand", "expanded_value"),
            ]:
                observe(attention, field, name)
            observe(layer.indexer, "projection", "index_query_projection")
            for field, name in [("attention_reduce", "attention"), ("mlp_reduce", "mlp")]:
                observe(layer, field, name)
            observe(layer.mlp.shared, "up_projection", "shared_gate_up")
            observe(layer.mlp.shared, "down_projection", "shared_down")
            observe(layer.mlp, "shared", "shared_expert")
            from ignition.components import bf16_experts

            launch = bf16_experts.launch_triton

            def observe_expert_launch(kernel, compiled, operands, grid, **options):
                if kernel is bf16_experts._moe_bf16_stage2_kernel:
                    observed["expert_intermediate"] = operands[0].cpu().clone()
                return launch(kernel, compiled, operands, grid, **options)

            bf16_experts.launch_triton = observe_expert_launch
            observe(layer.mlp, "router_projection", "router_logits")
            normalized = layer._normalized

            def norm(source, name):
                result = normalized(source, name)
                observed[name + "_input"] = source.cpu().clone()
                observed[name + "_output"] = result.cpu().clone()
                return result

            layer._normalized = norm
            from flashinfer.triton.norm import rms_norm

            rotate = op.layer.angles.apply
            store_latent = attention._store

            def capture_latent(projected, metadata, angles, *, query=None):
                store_latent(projected, metadata, angles, query=query)
                source = projected[:, 2048:2560]
                reference = torch.empty_like(source, memory_format=torch.contiguous_format)
                rms_norm(source, attention.weights["kv_a_layernorm.weight"], reference, attention.eps,
                         round_before_weight=True)
                observed["latent_norm_input"] = source.cpu().clone()
                observed["reference_latent_norm"] = reference.cpu().clone()
                rotary = projected[:, 2560:2624].clone()
                observed["key_rotation_input"] = rotary.cpu().clone()
                rotate(rotary, angles)
                observed["reference_rotary_key"] = rotary.cpu().clone()
                slots = metadata.token_slots[:projected.shape[0]]
                observed["write_slots"] = slots.cpu().clone()
                addressed = attention.cache.view(-1, 656).index_select(0, slots)
                observed["key_rotation_output"] = addressed[:, 528:].contiguous().view(torch.bfloat16).cpu()

            attention._store = capture_latent
            store_index = layer.indexer.store

            def capture_index(source, metadata, angles):
                store_index(source, metadata, angles)
                reference = reference_index_key(source, layer.indexer.weights)
                observed["index_key_rotation_input"] = reference.cpu().clone()
                rotate(reference, angles)
                observed["index_key_rotation_output"] = reference.cpu().clone()
                slots = metadata.token_slots[:source.shape[0]]
                pages, offsets = slots // 64, slots % 64
                rows = layer.indexer.cache[pages]
                lanes = torch.arange(128, device=device)
                observed["stored_index_bytes"] = rows.gather(1, offsets[:, None] * 128 + lanes).cpu()
                observed["stored_index_scales"] = rows[:, 8192:].contiguous().view(torch.float32).gather(
                    1, offsets[:, None]).flatten().cpu()

            layer.indexer.store = capture_index

            def rotation(value, angles, **kwargs):
                name = (
                    ("query_rotation" if value.ndim == 3 else "key_rotation")
                    if value.shape[-1] != 128
                    else ("index_query_rotation" if value.ndim == 3 else "index_key_rotation")
                )
                before = value.cpu().clone()
                result = rotate(value, angles, **kwargs)
                observed[name + "_input"] = before
                observed[name + "_output"] = value.cpu().clone()
                observed["rotary_values"] = angles.cpu().clone()
                return result

            op.layer.angles.apply = rotation
            route = layer.mlp.routing

            def routing(logits):
                result = route(logits)
                observed["expert_ids"], observed["expert_weights"] = (
                    result.ids.cpu().clone(),
                    result.weights.cpu().clone(),
                )
                return result

            layer.mlp.routing = routing
            observe(layer.mlp.experts, "run", "expert_contributions")
            attend = attention.attend

            def selected(query, selection, metadata):
                if tag[0] == "recent_selection":
                    visible = int(metadata.positions[0]) + 1
                    selection = Selection(
                        torch.arange(
                            max(0, visible - 2048), visible, device=device, dtype=torch.int32
                        )[None],
                        selection.lengths,
                    )
                observed["query"] = query.cpu().clone()
                observed["selected"] = selection.records.cpu().clone()
                observed["positions"] = metadata.positions.cpu().clone()
                visible = int(metadata.positions[0]) + 1
                observed["cache"] = attention.cache.view(-1, 656)[:visible].cpu().clone()
                result = attend(query, selection, metadata)
                from b12x.attention._shared.mla.kernel import LAST_DECODE_PLAN

                observed["attention_split_plan"] = dict(LAST_DECODE_PLAN)
                return result

            attention.attend = selected
            for consumed in (2305, 2048, 2049, 2561):
                teacher = fixture["hidden"][:consumed].to(device=device, dtype=torch.bfloat16)
                tokens = fixture["token_ids"][1 : consumed + 1].to(device)
                recursive = 369 if consumed == 2305 else int(fixture["token_ids"][consumed + 1])
                pool.clear_for_preparation()
                for start in range(0, consumed - 1, 128):
                    count = min(128, consumed - 1 - start)
                    embeddings = torch.zeros((128, 6144), device=device, dtype=torch.bfloat16)
                    previous = torch.zeros_like(embeddings)
                    embeddings[:count] = weights.embedding[tokens[start : start + count]]
                    previous[:count] = teacher[start : start + count]
                    placed = torch.where(positions < count, positions + start, -1)
                    context(embeddings, previous, placed, requests, table)
                saved = {
                    name: pool.pages[name].clone() for name in ("draft.latent", "draft.indexer")
                }
                before = torch.tensor([consumed], device=device)
                results = {
                    "consumed": consumed,
                    "recursive_token": recursive,
                    "cases": {},
                    "attention_configuration": attention_configuration,
                    "sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
                    "rotary_frequencies": op.layer.angles.frequencies.cpu().clone(),
                }
                cases = ["repeat-0", "repeat-1", "repeat-2"]
                if consumed == 2305:
                    cases += ["wrong_hidden", "shifted_position", "recent_selection"]
                for case in cases:
                    for name, value in saved.items():
                        pool.pages[name].copy_(value)
                    pool.last_hidden[0].copy_(teacher[-2 if case == "wrong_hidden" else -1])
                    state = op.begin(slots)
                    steps = []
                    tag[0] = case
                    for index in range(2 if case.startswith("repeat-") else 1):
                        observed.clear()
                        observed["reused_selection"] = state[1] is not None
                        ids = (
                            tokens[-1:] if index == 0 else torch.tensor([recursive], device=device)
                        )
                        logits, state = op.step(
                            state,
                            index,
                            ids,
                            before - int(case == "shifted_position"),
                            slots,
                            active,
                            active,
                        )
                        observed["normalized"] = state[0].cpu().clone()
                        observed["logits"] = logits.cpu().clone()
                        observed["retained_selection"] = state[1].records.cpu().clone()
                        sample_prefix[0, 0] = recursive
                        sampler = samplers[index]
                        sampler.logits[:, 0].copy_(logits.float())
                        observed["proposal_probabilities"] = {}
                        for top_p in (1.0, 0.95):
                            sample_state.parameters[0, 1] = top_p
                            sampler()
                            observed["proposal_probabilities"][str(top_p)] = {
                                "probabilities": sampler.probabilities.cpu().clone(),
                                "sampled": sampler.sampled.cpu().clone(),
                                "valid": sampler.valid.cpu().clone(),
                            }
                        steps.append(dict(observed))
                    results["cases"][case] = steps
                torch.save(results, Path(destination) / f"consumed-{consumed}-r{rank}.pt")
                if rank == 0:
                    print("captured draft", consumed, "recursive token", recursive, flush=True)
            session.freeze()
    finally:
        dist.destroy_process_group()
