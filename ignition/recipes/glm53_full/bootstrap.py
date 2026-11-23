# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare one tensor-parallel rank of GLM 5.3: load, tune, capture, and describe its serving graphs."""

import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.capacity import (batch_capacities, prefill_capacities, request_capacity, serving_modes,
                                            speculative_shapes, speculative_token_capacity)
from ignition.components.causal_draft import PreparedCausalDraftRound
from ignition.components.completion import VECTOR_TOKENS_FIELD, CompletionOutputs
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.bootstrap import BasePreparationConfig
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
                                              create_all_reduce_runtime, create_result_broadcast_runtime)
from ignition.preparation.execution import GraphSignature
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.progress import phase

from .attention import PAGE_TOKENS, ExampleBuffers
from .contract import prepare as prepare_contract
from .decoder import PreparedDecoder
from .mtp import PreparedContext, PreparedProposals
from .pool import ContinuationPool, continuation_layout, page_bytes
from .weights import ModelWeights

_COMPILED_KINDS = ("prefill", "decode", "resume", "speculative")


@dataclass(frozen=True)
class PreparationConfig(BasePreparationConfig):
    prompt_logprobs: bool = False
    devices: tuple[int, ...] = tuple(range(8))
    context: int = 1048576
    max_active: int = 8
    stream_overlap: bool = True
    sampling_defaults: dict | None = None
    # Speculation with the checkpoint's draft layer: a round verifies the pending token and width - 1 drafts.
    mtp: bool = False
    speculative_widths: tuple[int, ...] = (1,)

    def __post_init__(self):
        if len(self.devices) != 8 or len(set(self.devices)) != 8 or any(d < 0 for d in self.devices):
            raise ValueError("GLM 5.3 preparation requires eight distinct GPUs")
        if self.prompt_logprobs:
            raise ValueError("GLM 5.3 preparation does not capture prompt-scoring graphs")
        request_capacity(self.max_active)
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError(f"GLM 5.3 context must be {PAGE_TOKENS}-aligned in [{PAGE_TOKENS},1048576]")
        if not 128 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 128:
            raise ValueError("GLM 5.3 prefill chunk must be 128-aligned and fit the context")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if self.mtp_only and (not self.mtp or max(self.speculative_widths, default=1) <= 1
                              or self.speculative_fixed_width == 1):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
        if max(self.speculative_widths) > 8:
            raise ValueError("GLM 5.3 speculation supports at most seven draft tokens")
        if not self.mtp and self.speculative_widths != (1,):
            raise ValueError("GLM 5.3 verification widths above one require the draft layer")
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the prepared widths")

    @property
    def speculative_shapes(self):
        """Request batch and verification width of every speculative round graph."""
        return (speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
                if self.mtp else ())

    @property
    def round_tokens(self):
        """Most target rows of one decode or speculative forward."""
        slots = request_capacity(self.max_active)
        return max((batch * width for batch, width in self.speculative_shapes), default=slots)

    @property
    def preparation_pages(self):
        """Pages backed while tuning and capturing: one prefill chunk and two pages for every request."""
        return -(-self.prefill_chunk_size // PAGE_TOKENS) + 2 * request_capacity(self.max_active)

    @property
    def tuning_state_rows(self):
        """The null row and one live row for every request."""
        return max(3, 1 + request_capacity(self.max_active))


def tuning_identity(config, headers, world_size):
    """Scope kernel selections to the checkpoint's geometry, shard headers, and rank count."""
    value = {"config": config.raw, "headers": headers, "world_size": world_size,
             "contract": "glm53-full.graph-inputs.v1"}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage):
    """Load, tune, and capture the text catalog on one rank of a supervised rank group.

    Core storage backs weights, state rows, pages, collective mappings, and
    graphs at addresses that native loading reproduces. Every rank must close
    its collectives together after native replay drains.
    """
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [item.header_sha256 for item in manifest.shards]
    with core_storage.transient():
        weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, draft=config.mtp)
        del manifest
        release_unused()
    model = weights.config
    layout = continuation_layout(model, draft=config.mtp)
    page_size = page_bytes(model, draft=config.mtp)
    shapes, round_tokens = config.speculative_shapes, config.round_tokens
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete GLM 5.3 continuation row")
    slots, blocks = request_capacity(config.max_active), config.context // PAGE_TOKENS
    state_rows = slots + snapshots
    capture_state_rows = min(state_rows + 1, 1 + 2 * slots)
    pages = config.preparation_pages
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    ipc = core_storage.peer_storage
    with phase(progress, "Allocate model state and collectives"):
        pool = ContinuationPool.for_model(model, state_rows + 1, pages, device, draft=config.mtp,
            storage=core_storage, maximum_pages=total_memory // page_size,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(capture_state_rows, config.tuning_state_rows))
        requests = DeviceRequestState.allocate(device, vocab_size=model.vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max(config.prefill_chunk_size, round_tokens), blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=model.vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + max(config.speculative_widths)),
                                                device=device)
        coordinator = CollectivePreparation(group)
        collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning,
            max_tokens=max(config.prefill_chunk_size, round_tokens), hidden_size=model.hidden, ipc=ipc)
        vocabulary_collective = create_all_gather_runtime(group, device, max_rows=round_tokens,
            columns=weights.head.shape[0] * shard.world_size, ipc=ipc)
        result_collective = None
        if config.mtp:
            from ignition.components.draft_sampling import PreparedDraftDistribution
            from ignition.components.speculation import SpeculativeAcceptance

            result_layouts = (SpeculativeAcceptance.broadcast_layout(slots, max(config.speculative_widths)),)
            if max(config.speculative_widths) > 1:
                result_layouts += (PreparedDraftDistribution.broadcast_layout(slots),)
            result_collective = create_result_broadcast_runtime(group, device, layouts=result_layouts, ipc=ipc)
        identity = tuning_identity(model, headers, shard.world_size)
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning,
            collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort,
            tuning_replica=config.tuning_replica, native_dense_activations=config.native_dense_activations,
            lock_gpu_clocks=config.lock_gpu_clocks, device=device, checkpoint=identity,
            workload_schema="glm53-full", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage)
        session.__enter__()
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token, sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame, submission_sequence=sequence, **completion.buffers())
    forwards, signatures = {}, {}
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    examples = ExampleBuffers(device)
    modes = serving_modes(config.mtp, config.mtp_only)
    collectives = dict(runtime=collective, vocabulary_runtime=vocabulary_collective, coordinator=coordinator)
    context = partial(PreparedContext, runtime=collective, coordinator=coordinator, buffers=examples)
    graph_count = sum(len(prefill_sizes) + (0 if enabled else len(batch_sizes)) + 4 * len(batch_sizes)
                      for enabled in modes) + len(shapes)
    with phase(progress, "Construct graph catalog", total=graph_count) as stage:
        def add(name, forward, kind, rows, tokens, fields, enabled=None):
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        # Largest capacities first: smaller forwards reuse prefixes of their example tensors. With the draft
        # layer, decode runs as width-one speculative rounds; graphs named "-no-mtp" serve requests that
        # disable speculation and leave the draft layer's records unwritten.
        for enabled in modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for mode, sizes in (("prefill", reversed(prefill_sizes)),
                                ("decode", () if enabled else reversed(batch_sizes))):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{mode}{capacity}" + (suffix if mode == "prefill" else "")
                    metadata = Metadata.allocate(capacity, rows, blocks, device)
                    decoder = PreparedDecoder(mode_weights, metadata, pool, mode=mode, session=session, name=name,
                                              workspace=arena, examples=examples, overlap=config.stream_overlap, **collectives)
                    batch = PreparedBatch(inputs, requests, metadata, mode=mode, shard=shard, pool=pool,
                                          page_tokens=PAGE_TOKENS)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=context if enabled else None)
                    add(name, forward, mode, rows, capacity if mode == "prefill" else 1, forward_inputs,
                        enabled if config.mtp else None)
        speculation = None
        if config.mtp:
            speculation = SimpleNamespace(weights=weights, pool=pool, inputs=inputs, requests=requests,
                scratch=SimpleNamespace(rows=lambda tokens, rows: None), draft_scratch=None, session=session,
                workspace=arena, collectives=collectives, result_runtime=result_collective,
                overlap=config.stream_overlap, completion=completion, frame=frame,
                sequence=sequence, rng=SpeculativeRngState.allocate(device, capacity=slots),
                offset=torch.zeros(slots, dtype=torch.int64, device=device),
                generation=torch.zeros(slots, dtype=torch.int64, device=device), page_tokens=PAGE_TOKENS)

            def verifier(weights, scratch, metadata, pool, *, mode, name, overlap, **common):
                # Verification rows of one request are consecutive positions, each attending to the
                # records the rows before it stored in this forward, as decode rows do.
                return PreparedDecoder(weights, metadata, pool, mode="decode", name=name, examples=examples, overlap=overlap, **common)

            def accepted(weights, metadata, pool, **common):
                # The round passes its context no name; `name` is the round under construction.
                return PreparedContext(weights, None, metadata, pool, name=name + ".context", buffers=examples,
                                       **common)

            def proposals(weights, pool, scratch, batch, depth, **common):
                return PreparedProposals(weights, pool, inputs, batch, depth, buffers=examples, **common)

            for batch, width in sorted(shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
                name = f"speculative{batch}x{width}"
                forward = PreparedCausalDraftRound(speculation, batch, width, name, decoder_factory=verifier,
                    context_factory=accepted, proposal_factory=proposals, draft_context_lag=1)
                forward.prepare()
                add(name, forward, "speculative", batch, width, forward_inputs, True)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name)
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"),
                    enabled if config.mtp else None)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name, capacity=capacity,
                        admission_buffers=admission, page_tokens=PAGE_TOKENS, draft_state=enabled)
                    fields = (row_inputs if mode == "snapshot"
                              else ("request_state_rows", *admission.__dataclass_fields__))
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()),
                        enabled if config.mtp else None)
    with phase(progress, "Prepare and tune kernels") as stage, core_storage.transient():
        tuning = session.tune(show_progress=progress is not None, console=stage.console)
    examples.clear()
    metadata = decoder = batch = forward = None
    arena.release_preparation()
    stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id

    def materialize():
        # Snapshot graphs copy between live rows and snapshot rows; operator tuning needs only live rows.
        core_storage.grow("STATE_ROWS", capture_state_rows)
        pool.memory_capacity[0] = capture_state_rows
        pool.clear_for_preparation()

    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
        pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
        progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
        materialize=materialize, warmup_storage=core_storage, compile_kinds=_COMPILED_KINDS)
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    assets = {"tokenizer": {"path": str(snapshot / "tokenizer.json")},
              "chat_template": annotated_template_asset((snapshot / "chat_template.jinja").read_bytes(),
                  Path(__file__).with_name("assets") / "chat_template.patch.json")}
    generation = json.loads((snapshot / "generation_config.json").read_text())
    sampling_defaults = config.sampling_defaults
    if sampling_defaults is None:
        from .recipe import RECIPE

        sampling_defaults = RECIPE.sampling_defaults.as_dict()
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=PAGE_TOKENS, frontend=prepare_frontend(snapshot, eos_token_ids=generation["eos_token_id"]),
        sampling_defaults=sampling_defaults, assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=model.vocabulary,
                                        draft_lag=1 if config.mtp else None))
    if config.mtp:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_fixed_width=config.speculative_fixed_width,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, max(config.speculative_widths), config.speculative_token_capacity))
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=config.devices[rank],
        weights_bytes=weights.allocated_bytes, mapped_host_bytes=0, offloaded_bytes=0,
        continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=page_size, page_tokens=PAGE_TOKENS, kv_bytes=pages * page_size,
        speculative_shapes=shapes,
        completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.tensor.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    return PreparedModel(rebuild_weights=declare_weights(weights,
            components=("embedding", "norm", "head", *(("draft",) if config.mtp else ()))),
        device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=completion.zero_ranges(),
        resources=(weights, pool, requests, inputs, admission, commands, forwards, compiled, stream, graph_pool,
                   session, coordinator, collective, vocabulary_collective, result_collective, speculation),
        close_resources=tuple(c for c in (result_collective, vocabulary_collective, collective) if c is not None))
