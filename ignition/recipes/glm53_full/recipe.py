# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualified checkpoint, serving range, and memory costs of the GLM 5.3 recipe."""

from types import MappingProxyType

from ignition.recipes.declaration import (Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, SamplingRange,
                                          Speculation)

# One rank's bytes per 256-token page: a 656-byte latent record per token in each of the 78 layers and a
# 132-byte index key per token in the 21 layers that select.
_PAGE_BYTES = 256 * (78 * 656 + 21 * 132)
# The draft layer adds one latent record and one index key per token.
_DRAFT_PAGE_BYTES = 256 * (656 + 132)
# One continuation row: a partial page of every region, the final hidden state, and the consumed-token
# count, each region aligned to 256 bytes.
_STATE_ROW_BYTES = 13_821_184
# Estimate of what a rank holds beyond stored weights, state rows, and pages: operator scratch, graphs,
# collectives, and the CUDA context.
_GRAPH_BYTES = 6 << 30


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare the plan's text catalog and speculation settings on process-local GPUs 0 through 7."""
    from ignition.preparation.bootstrap import common_preparation_kwargs, validate_host_topology
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    if ranks is not None:
        raise ValueError("GLM 5.3 preparation requires GPUs on one host")
    validate_host_topology(plan.roce_ring, plan.roce_chain, None)
    speculation = plan.speculation
    config = PreparationConfig(
        **common_preparation_kwargs(plan, compile_forwards=compile_forwards,
                                    kernel_cache=kernel_cache, retain_graphs=retain_graphs),
        mtp=speculation is not None,
        speculative_widths=(1,) if speculation is None else speculation.widths,
        speculative_fixed_width=None if speculation is None else speculation.fixed_width,
        speculative_token_capacity=None if speculation is None else plan.speculative_token_capacity,
        prompt_logprobs=plan.prompt_logprobs, devices=tuple(range(plan.world_size)),
        stream_overlap=plan.stream_overlap,
        sampling_defaults=RECIPE.sampling_defaults.as_dict())
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="glm53_full", served_model_name="GLM-5.3", prepare=prepare, usage=usage,
    # Evaluation settings span temperature and top-p 0.95 to 1.0: https://huggingface.co/zai-org/GLM-5.3#footnotes.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
        ranges={"temperature": SamplingRange(.95, 1.), "top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(
        alias="glm-5.3", repository="local-inference-lab/GLM-5.3-NVFP4",
        revision="cca10d1586255195d3279785fc85577bfc1e9227", architecture="GlmMoeDsaForCausalLM",
        stored_bytes=464_795_267_072, device_bytes=464_795_267_072,
        # Decoder layer 78, the draft layer, which preparation without speculation leaves unloaded.
        draft_bytes=19_905_841_664,
        # Seven further copies of the tensors every rank holds whole: embeddings, normalization, low-rank
        # attention inputs, routers and the indexer.
        expansion_bytes=35_354_135_040),),
    context=1048576, page_tokens=256, concurrency=8, max_concurrency=32, prefill_chunk=4096,
    # A round verifies the pending token and up to five drafts of the checkpoint's draft layer.
    speculation=Speculation(draft_tokens=5, token_capacity=128), max_draft_tokens=7, verification_rows=0, stream_overlap=True,
    b12x_environment=MappingProxyType({"B12X_PCIE_TP8_OWNER_REDUCE": "0"}),
    ranks=MappingProxyType({8: RankCosts(_STATE_ROW_BYTES + _DRAFT_PAGE_BYTES, _PAGE_BYTES + _DRAFT_PAGE_BYTES,
                                         _GRAPH_BYTES)}),
    plain_ranks=MappingProxyType({8: RankCosts(_STATE_ROW_BYTES, _PAGE_BYTES, _GRAPH_BYTES)}),
)
