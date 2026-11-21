# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualified checkpoint, serving range, and memory costs of the GLM MoE DSA recipe."""

from types import MappingProxyType

from ignition.recipes.declaration import (Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange,
                                          Speculation)

from .config import INDEX_RECORD_BYTES, LATENT_RECORD_BYTES, PAGE_RECORDS

# GLM-5.3: every one of the 78 decoder layers stores one latent record per position, and the 21
# layers that own an indexer also store one index key.
_LATENT_LAYERS = 78
_INDEX_LAYERS = 21
# Each rank holds its own copy of the embedding, final normalization, indexer projections, layer
# normalizations, low-rank attention inputs, and routers. The planner divides stored weights by the
# rank count, so a rank's copies beyond that share count toward its fixed cost.
_REPLICATED_BYTES = 5_050_590_720
# Estimate of what a rank holds beyond pages, state rows, and weights: collectives, the
# context-parallel prefill slabs, the CUDA context, graphs, and operator scratch. Not measured.
_SCRATCH_BYTES = 8 << 30


def _costs(ranks, *, speculating):
    """One rank's costs of a continuation row, a page, and what it holds beyond them.

    A page holds PAGE_RECORDS positions of a rank: a latent record of each layer and, for each layer
    that owns an indexer, an index key. Context ranks share a page, which then covers PAGE_RECORDS
    positions for each of them at the same bytes. The draft layer adds one latent layer that owns an
    indexer.
    """
    draft = int(speculating)
    page = PAGE_RECORDS * ((_LATENT_LAYERS + draft) * LATENT_RECORD_BYTES
                           + (_INDEX_LAYERS + draft) * INDEX_RECORD_BYTES)
    # A continuation row holds every layer's partial page, the final hidden vector, and the consumed-token count.
    return RankCosts(page + 6144 * 2 + 256, page, _SCRATCH_BYTES + _REPLICATED_BYTES * (ranks - 1) // ranks,
                     page_tokens=PAGE_RECORDS * ranks)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare the plan's catalog on process-local GPUs 0 through N-1 of one host."""
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("GLM MoE DSA preparation requires GPUs on one host")
    if plan.prompt_logprobs:
        raise ValueError("GLM MoE DSA preparation does not capture prompt scoring graphs")
    speculation = plan.speculation
    drafting = {} if speculation is None else dict(mtp=True, mtp_only=speculation.mtp_only,
        speculative_widths=speculation.widths, speculative_token_capacity=plan.speculative_token_capacity,
        speculative_fixed_width=speculation.fixed_width)
    config = PreparationConfig(
        collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        compile_workers=plan.compile_workers, autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)), compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        max_active=plan.concurrency, stream_overlap=plan.stream_overlap, context_parallel=plan.context_parallel,
        context_parallel_size=plan.context_parallel_size,
        context=plan.recipe.context, prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes, kernel_cache=kernel_cache, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="glm_dsa", served_model_name="GLM-5.3", prepare=prepare, usage=usage,
    # Bounds span the HLE, DeepSWE, and coding presets: https://huggingface.co/zai-org/GLM-5.3#footnotes.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
        ranges={"temperature": SamplingRange(.95, 1.), "top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(
        alias="glm-5.3", repository="local-inference-lab/GLM-5.3-NVFP4-CSF",
        revision="a4d406932f2db96db8e852867c7939f448e0cd6f", architecture="GlmMoeDsaForCausalLM",
        stored_bytes=442_830_659_868,
        # Every tensor once as it resides on a device: routed-expert block scales stay CSF-compressed,
        # and the draft layer's BF16 experts are stored as NVFP4.
        device_bytes=428_902_348_024,
        # Decoder layer 78, the MTP module.
        draft_bytes=6_014_306_816),),
    context=1048576, page_tokens=PAGE_RECORDS, concurrency=16, max_concurrency=32, prefill_chunk=4096,
    # The draft layer proposes one token per pass, each from its own previous output.
    speculation=Speculation(draft_tokens=3, token_capacity=64), max_draft_tokens=7, verification_rows=0,
    # Decode, verification, and prefill chunks through 256 rows run shared experts on an auxiliary stream.
    stream_overlap=True,
    # b12x retains activation scales for A4 prefill launches in W4A16 expert paths when set.
    b12x_environment=MappingProxyType({"B12X_W4A16_A4_PREFILL": "1"}),
    # Two groups of four consecutive ranks, each with a complete copy of every context.
    hybrid_context_parallel=((8, 4),),
    ranks=MappingProxyType({8: _costs(8, speculating=True)}),
    plain_ranks=MappingProxyType({8: _costs(8, speculating=False)}),
)
