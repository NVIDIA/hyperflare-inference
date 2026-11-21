# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a GLM MoE DSA tensor-parallel rank before native serving takes ownership."""

import contextlib
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from ignition.components.capacity import (batch_capacities, prefill_capacities, request_capacity, serving_modes,
                                          speculative_shapes, speculative_token_capacity)
from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel import TensorParallelShard
from ignition.components.prompt_logprobs import prefill_graph_count, prefill_modes
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.speculation import SpeculativeRngState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.bootstrap import BasePreparationConfig
from ignition.preparation.capture import capture_forwards
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
                                              create_all_reduce_runtime, create_dma_all_reduce_runtime,
                                              create_head_exchange)
from ignition.preparation.execution import GraphSignature
from ignition.preparation.fixtures.recurrent import register_benchmark_state
from ignition.preparation.frontend import annotated_template_asset, prepare_frontend
from ignition.preparation.host_memory import release_unused
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.progress import phase

from .config import INDEX_TOPK, LATENT_RANK, PAGE_RECORDS, ROTARY_DIM
from .contract import prepare as prepare_contract
from .dcp import ContextParallel, ContextShards
from .decoder import PreparedDecoder, Scratch
from .mtp import PreparedFeedback
from .pool import ContinuationPool, continuation_layout, page_bytes
from .speculative import PreparedRound
from .weights import ModelWeights

# Most rows of one all-reduce through the one-shot channel. Larger reductions run in chunks of this many rows.
REDUCE_ROWS = 1024
# Prefill graphs of at least this many rows reduce through the copy-engine ring instead.
DMA_PREFILL_ROWS = 512


@dataclass(frozen=True)
class PreparationConfig(BasePreparationConfig):
    devices: tuple[int, ...] = tuple(range(8))
    context: int = 1048576
    max_active: int = 16
    minimum_pages: int | None = None
    # Activation precision of routed experts in decode graphs, and in prefill graphs. A16 experts keep
    # their calibrated activation scales, so either prefill precision loads the same weights.
    expert_activation: str = "a16"
    prefill_expert_activation: str = "a16"
    # Expand routed experts' CSF block scales once at load instead of before every expert launch.
    expand_expert_scales: bool = False
    # Share each page's latent records across the ranks instead of holding every record on every rank.
    context_parallel: bool = True
    # Run each layer's shared expert on an auxiliary stream beside its routing and routed experts.
    stream_overlap: bool = True
    # Ranks that share one copy of a context, or None to share it across every rank. Smaller groups of
    # consecutive ranks each hold a complete copy and exchange attention only among themselves.
    context_parallel_size: int | None = None
    # Prepare the checkpoint's draft layer and speculative rounds. A verification width counts the
    # pending input and the drafts that follow it.
    mtp: bool = False
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4)

    def __post_init__(self):
        if (not isinstance(self.devices, tuple) or len(self.devices) != 8 or len(set(self.devices)) != 8
                or any(type(device) is not int or device < 0 for device in self.devices)):
            raise ValueError("GLM MoE DSA preparation requires eight distinct CUDA devices")
        if self.context_parallel_size is not None and (
                type(self.context_parallel_size) is not int or self.context_parallel_size < 2
                or len(self.devices) % self.context_parallel_size):
            raise ValueError("context-parallel groups must split the ranks evenly, at least two to a group")
        if self.expert_activation not in ("a4", "a16") or self.prefill_expert_activation not in ("a4", "a16"):
            raise ValueError("GLM MoE DSA expert activation must be a4 or a16")
        if self.expert_activation == "a4" and self.prefill_expert_activation == "a16":
            raise ValueError("A4 experts keep no A16 prefill launches")
        for name in ("context", "prefill_chunk_size", "max_active", "snapshot_budget_bytes", "headroom_bytes"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if not self.page_tokens <= self.context <= 1048576 or self.context % self.page_tokens:
            raise ValueError(f"GLM MoE DSA context must be {self.page_tokens}-aligned in [{self.page_tokens},1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        if self.minimum_pages is not None and (type(self.minimum_pages) is not int or self.minimum_pages <= 0):
            raise ValueError("minimum page capacity must be positive")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")
        if type(self.mtp) is not bool or type(self.mtp_only) is not bool:
            raise TypeError("mtp and mtp_only must be booleans")
        if self.mtp_only and (not self.mtp or max(self.speculative_widths, default=1) <= 1
                              or self.speculative_fixed_width == 1):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.mtp:
            if max(self.speculative_widths, default=1) > 8:
                raise ValueError("GLM MoE DSA completions hold at most seven draft tokens")
            if (self.speculative_fixed_width is not None
                    and self.speculative_fixed_width not in self.speculative_widths):
                raise ValueError("fixed speculative width must belong to the configured widths")
            self.speculative_shapes

    @property
    def context_ranks(self):
        """Ranks that share one context's latent records and index keys."""
        return (self.context_parallel_size or len(self.devices)) if self.context_parallel else 1

    @property
    def page_tokens(self):
        """Global positions per KV page: each context rank holds PAGE_RECORDS positions of a page."""
        return PAGE_RECORDS * self.context_ranks

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def speculative_shapes(self):
        """Request batch and verification width of every speculative round."""
        if not self.mtp:
            return ()
        return speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)

    @property
    def decode_rows(self):
        """Most rows of one decode or verification forward."""
        return max((batch * width for batch, width in self.speculative_shapes), default=self.request_capacity)

    @property
    def completion_token_capacity(self):
        """Most tokens one completion record returns for a request."""
        return max((width for _, width in self.speculative_shapes), default=1)

    @property
    def preparation_pages(self):
        """Pages backed while tuning and capturing: a prefill chunk and boundary pages for every request row."""
        return -(-self.prefill_chunk_size // self.page_tokens) + 2 * self.request_capacity

    @property
    def tuning_state_rows(self):
        """Null and live decode rows, or one prefill row and its checkpoint."""
        return max(3, 1 + self.request_capacity)


def tuning_identity(config, headers, world_size, context_ranks=1, a4_prefill=False):
    """Scope kernel selections to the checkpoint's geometry, shards, rank count, context sharding, and
    prefill expert precision.

    Every context-parallel group size shares one scope: the operators that depend on the group size
    carry it in their own descriptors.
    """
    value = {"config": config.text, "headers": headers, "world_size": world_size,
             "contract": "glm_dsa.graph-inputs.v1"}
    if context_ranks != 1:
        value["context_ranks"] = world_size
    if a4_prefill:
        value["prefill_expert_activation"] = "a4"
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage):
    """Load, tune and capture the text catalog on one supervised tensor-parallel rank.

    The caller must terminate the complete rank group if any preparation fails.
    Core storage backs device memory, collective mappings and graphs with
    addresses that native restore reproduces.
    """
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    if group is None:
        raise ValueError("GLM MoE DSA preparation requires its complete rank group")
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
        headers = [record.header_sha256 for record in manifest.shards]
        metadata_path = manifest.metadata_path
    a4_prefill = config.prefill_expert_activation == "a4"
    with core_storage.transient():
        weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, mtp=config.mtp,
                                    expert_activation=config.expert_activation,
                                    expand_expert_scales=config.expand_expert_scales)
        del manifest
        release_unused()
    model = weights.config
    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    context_ranks = config.context_ranks
    layout = continuation_layout(model, draft=config.mtp)
    rank_page_bytes = page_bytes(model, draft=config.mtp)
    snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
    if snapshots < 1:
        raise ValueError("snapshot budget must hold a complete GLM MoE DSA continuation")
    slots, blocks = config.request_capacity, config.context // config.page_tokens
    decode_rows = config.decode_rows
    max_tokens = max(config.prefill_chunk_size, decode_rows)
    state_rows = slots + snapshots
    capture_state_rows = min(state_rows + 1, 1 + 2 * slots)
    pages = config.preparation_pages
    total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
    with phase(progress, "Allocate model state"):
        pool = ContinuationPool.for_model(model, state_rows + 1, pages, device, draft=config.mtp,
            storage=core_storage, maximum_pages=total_memory // rank_page_bytes,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(capture_state_rows, config.tuning_state_rows))
        vocabulary = model.vocabulary
        requests = DeviceRequestState.allocate(device, vocab_size=vocabulary, capacity=slots)
        inputs = BatchInputs.allocate(max_tokens, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=vocabulary, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        sequence = torch.zeros(1, dtype=torch.int64, device=device)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity),
                                                device=device)
        coordinator = CollectivePreparation(group)
        # Prefill activations reduce in chunks of the rows decode already needs a channel for.
        channel_rows = max(min(max_tokens, REDUCE_ROWS), decode_rows)
        collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning,
            max_tokens=channel_rows, hidden_size=model.hidden, ipc=core_storage.peer_storage)
        prefill_collective = create_dma_all_reduce_runtime(group, device, max_tokens=config.prefill_chunk_size,
            hidden_size=model.hidden, ipc=core_storage.peer_storage)
        first, last = model.vocabulary_partition(shard)
        vocabulary_collective = create_all_gather_runtime(group, device, max_rows=decode_rows,
            columns=(last - first) * shard.world_size, ipc=core_storage.peer_storage)
        context = None
        if context_ranks > 1:
            exchange, exchange_ipc = group, core_storage.peer_storage
            if context_ranks < shard.world_size:
                import torch.distributed as dist

                # Consecutive ranks form a group. Each group holds a complete copy of every context and
                # attends it with its own ranks' heads, so no exchange leaves the group.
                starts = range(0, shard.world_size, context_ranks)
                groups = [dist.new_group(list(range(start, start + context_ranks))) for start in starts]
                for start, members in zip(starts, groups, strict=True):
                    coordinator.register_group(tuple(range(start, start + context_ranks)), members)
                exchange = groups[rank // context_ranks]
                exchange_ipc = core_storage.peer_storage.for_group(exchange)
            # Decode exchanges selection candidates, absorbed query heads, and attended latents. Prefill
            # reads peers' page pools. A candidate is eight bytes, four BF16 columns.
            context = ContextParallel(ContextShards.on(rank % context_ranks, context_ranks, device),
                heads=create_head_exchange(exchange, device, max_rows=decode_rows,
                    heads=model.heads * context_ranks // shard.world_size, head_dim=LATENT_RANK,
                    query_head_dim=LATENT_RANK + ROTARY_DIM, ipc=exchange_ipc),
                candidates=create_all_gather_runtime(exchange, device, max_rows=decode_rows,
                    columns=INDEX_TOPK * 4 * context_ranks, ipc=exchange_ipc),
                coordinator=coordinator, peers=pool.share_layer_pages(core_storage, exchange, "latent"),
                index_peers=pool.share_layer_pages(core_storage, exchange, "indexer"),
                slab_positions=config.context)
        identity = tuning_identity(model, headers, shard.world_size, context_ranks, a4_prefill)
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning,
            collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort,
            tuning_replica=config.tuning_replica, native_dense_activations=config.native_dense_activations,
            lock_gpu_clocks=config.lock_gpu_clocks, device=device, checkpoint=identity,
            workload_schema="glm_dsa", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage)
        session.__enter__()
        register_benchmark_state(session, pool.reserved_backing, storage=pool.reserved_backing,
                                 restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        examples = {}
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, model, shard, device, examples=examples)
        decode_storage = Scratch.allocate(decode_rows, decode_rows, model, shard, device, examples=examples)
        buffers = dict(inputs.buffers(), **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            request_pending_token=requests.pending_token, sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame, submission_sequence=sequence, **completion.buffers())
        speculation = None
        if config.mtp:
            # What every speculative round shares: the target decoder's operands and the draft RNG stream.
            speculation = SimpleNamespace(weights=weights, pool=pool, inputs=inputs, requests=requests,
                scratch=decode_storage, draft_scratch=None, session=session, workspace=arena, overlap=None,
                collectives=dict(runtime=collective, vocabulary_runtime=vocabulary_collective,
                                 coordinator=coordinator, examples=examples, context=context),
                completion=completion, frame=frame, sequence=sequence, page_tokens=config.page_tokens,
                rng=SpeculativeRngState.allocate(device, capacity=slots),
                offset=torch.zeros(slots, dtype=torch.int64, device=device),
                generation=torch.zeros(slots, dtype=torch.int64, device=device))
    forwards, signatures, metadata = {}, {}, {}
    prefill_sizes, batch_sizes = prefill_capacities(config.prefill_chunk_size), batch_capacities(config.max_active)
    # Graphs of the speculative serving mode maintain draft state, but the plain mode's graphs carry no draft layer.
    modes = serving_modes(config.mtp, config.mtp_only)
    count = (len(modes) * (prefill_graph_count(prefill_sizes, None) + 4 * len(batch_sizes))
             + (len(batch_sizes) if False in modes else 0) + len(config.speculative_shapes))
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled):
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled)
            stage.update(advance=1)

        def prefill_feedback(weights, auxiliary, metadata, pool, *, session, name, workspace):
            return PreparedFeedback(weights, metadata, pool, mode="prefill", session=session, name=name,
                                    workspace=workspace, **speculation.collectives)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, mtp=None)
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            # Width-one speculative rounds decode for the speculative mode.
            for kind, mode, sizes in prefill_modes(prefill_sizes, () if enabled else reversed(batch_sizes), None):
                for capacity in sizes:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    m = metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    scratch = (prefill_storage if mode == "prefill" else decode_storage).rows(capacity, rows)
                    reducing = (prefill_collective if mode == "prefill" and capacity >= DMA_PREFILL_ROWS
                                else collective)
                    decoder = PreparedDecoder(mode_weights, scratch, m, pool, mode=mode, runtime=reducing,
                        vocabulary_runtime=vocabulary_collective, coordinator=coordinator, session=session,
                        name=name, workspace=arena, examples=examples, context=context, a4_prefill=a4_prefill)
                    batch = PreparedBatch(inputs, requests, m, mode=mode, shard=shard, pool=pool,
                                          page_tokens=config.page_tokens)
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=prefill_feedback if enabled else None)
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1, forward_inputs,
                        enabled=enabled)
        for batch, width in sorted(config.speculative_shapes, key=lambda shape: shape[0] * shape[1], reverse=True):
            name = f"speculative{batch}x{width}"
            forward = PreparedRound(speculation, batch, width, name)
            forward.prepare()
            add(name, forward, "speculative", batch, width, forward_inputs, enabled=True)
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            mode_weights = weights if enabled else replace(weights, mtp=None)
            suffix = "-no-mtp" if config.mtp and not enabled else ""
            for capacity in reversed(batch_sizes):
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name, workspace=arena)
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame,
                        sequence, runtime=collective, coordinator=coordinator, session=session, name=name,
                        capacity=capacity, admission_buffers=admission, draft_state=enabled,
                        page_tokens=config.page_tokens)
                    fields = (row_inputs if mode == "snapshot"
                              else ("request_state_rows", *admission.__dataclass_fields__))
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()),
                        enabled=enabled)
    if speculation is not None:
        speculation.scratch = speculation.collectives = None
    transient.close()
    with phase(progress, "Prepare and tune kernels") as stage, core_storage.transient():
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    examples.clear()
    metadata.clear()
    prefill_storage = decode_storage = scratch = decoder = batch = m = None
    arena.release_preparation()
    stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id

    def materialize():
        # Maintenance graphs need snapshot destinations beside live rows, but operator tuning does not.
        core_storage.grow("STATE_ROWS", capture_state_rows)
        pool.memory_capacity[0] = capture_state_rows
        pool.clear_for_preparation()

    graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
        pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
        progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
        materialize=materialize, warmup_storage=core_storage,
        compile_kinds=("prefill", "decode", "resume", "speculative"),
        retain_forwards=True)
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    release_unused()
    from .recipe import RECIPE

    assets = {"tokenizer": {"path": str(metadata_path / "tokenizer.json")},
              "chat_template": annotated_template_asset((metadata_path / "chat_template.jinja").read_bytes(),
                  Path(__file__).with_name("assets") / "chat_template.patch.json")}
    server_config = dict(identity=identity, model_id=snapshot.name, max_sequence_tokens=config.context,
        prefill_chunk_size=config.prefill_chunk_size, pages=pages, state_rows=state_rows, snapshots=snapshots,
        page_tokens=config.page_tokens,
        frontend=prepare_frontend(metadata_path, eos_token_ids=model.eos_token_ids),
        sampling_defaults=RECIPE.sampling_defaults.as_dict(), assets=assets,
        model_contract=prepare_contract(signatures, buffers, pool, vocabulary=vocabulary,
                                        page_tokens=config.page_tokens, draft_lag=0 if config.mtp else None))
    if config.mtp:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active,
                max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
    free, total = torch.cuda.mem_get_info(device)
    memory = dict(rank=rank, world_size=shard.world_size, device=config.devices[rank],
        weights_bytes=weights.allocated_bytes, continuation_bytes=pool.backing.nbytes, state_rows=state_rows,
        preparation_state_rows=int(pool.memory_capacity[0].item()),
        preparation_state_bytes=int(pool.memory_capacity[0].item()) * layout.row_bytes,
        pages=pages, page_bytes=rank_page_bytes, page_tokens=config.page_tokens, kv_bytes=pages * rank_page_bytes,
        context_ranks=context_ranks,
        completion_output_bytes=completion.nbytes, operator_scratch_bytes=arena.tensor.nbytes, graphs=len(graphs),
        prefill_capacities=prefill_sizes, free_device_bytes=free, total_device_bytes=total,
        headroom_bytes=config.headroom_bytes, tuning=tuning, compiled_decoder=config.compile_decoder)
    if config.mtp:
        memory.update(speculative_shapes=config.speculative_shapes)
    # The draft layer's weights form their own payload, named as the checkpoint names the layer, so
    # the payload that holds the embedding, final normalization, and head is the same with and without it.
    return PreparedModel(rebuild_weights=declare_weights(weights, components={
            "tree.model.embed_tokens": "embedding", "tree.model.norm": "norm", "head": "head",
            "mtp": f"layer-{model.layers}"}),
        device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        forwards={name: compiled.get(name, forwards[name]) for name in graphs},
        export_zero_ranges=completion.zero_ranges(),
        resources=(weights, pool, requests, inputs, admission, commands, forwards, compiled, stream, graph_pool,
                   session, coordinator, collective, prefill_collective, vocabulary_collective, context,
                   speculation))
