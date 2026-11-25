# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Recipe declarations: qualified checkpoints, serving ranges, and memory costs.

Declarations import no model code, so planning and model listing run without
PyTorch.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
import math
from types import MappingProxyType

from ignition.components.capacity import request_capacity

@dataclass(frozen=True)
class SamplingRange:
    """Inclusive bounds from the model author's sampling recommendations."""

    min: float | int
    max: float | int


@dataclass(frozen=True)
class SamplingDefaults:
    """Complete serving parameters manually verified against the model card."""

    temperature: float
    top_p: float
    top_k: int
    seed: int = 0
    min_p: float = 0.
    repetition_penalty: float = 1.
    presence_penalty: float = 0.
    frequency_penalty: float = 0.
    ranges: Mapping[str, SamplingRange] = field(default_factory=dict)

    def __post_init__(self):
        values = (self.temperature, self.top_p, self.min_p, self.repetition_penalty,
                  self.presence_penalty, self.frequency_penalty)
        if (any(type(v) not in (int, float) or not math.isfinite(v)
                or abs(v) > float.fromhex("0x1.fffffep+127") for v in values)
                or any(v < float.fromhex("0x1p-126") for v in
                       (self.temperature, self.top_p, self.repetition_penalty))
                or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1
                or type(self.top_k) is not int or not -1 <= self.top_k < 2**31
                or type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError("invalid recipe sampling defaults")
        names = {item.name for item in fields(self)} - {"seed", "ranges"}
        for name, bounds in self.ranges.items():
            if (name not in names or not isinstance(bounds, SamplingRange)
                    or any(type(v) not in (int, float) or not math.isfinite(v)
                           for v in (bounds.min, bounds.max))
                    or not bounds.min <= getattr(self, name) <= bounds.max):
                raise ValueError(f"invalid sampling range for {name}")
            for value in (bounds.min, bounds.max):
                replace(self, **{name: value}, ranges={})
        object.__setattr__(self, "ranges", MappingProxyType(dict(self.ranges)))

    def as_dict(self):
        result = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "ranges"}
        if self.ranges:
            result["ranges"] = {name: asdict(bounds) for name, bounds in self.ranges.items()}
        return result


@dataclass(frozen=True)
class Checkpoint:
    """One qualified checkpoint revision and where its stored tensors reside."""

    alias: str
    repository: str
    revision: str
    architecture: str
    stored_bytes: int
    device_bytes: int
    mapped_host_bytes: int = 0
    # The share of `device_bytes` that holds the draft model, loaded only for learned drafts.
    draft_bytes: int = 0
    # Device bytes that loading adds beyond the stored tensors, such as block scales
    # expanded from a compressed encoding.
    expansion_bytes: int = 0
    # Estimated resident device weights after optional lossless expert-scale compression.
    scale_compressed_device_bytes: Mapping[str, int] = field(default_factory=dict)
    # Per-rank costs replacing the recipe's `ranks` and `plain_ranks` when this checkpoint's draft or state
    # layout differs, such as a draft model that only this checkpoint carries. The declared sizes may be
    # a subset of the recipe's when the checkpoint's stored shards exclude a tensor-parallel size.
    ranks: Mapping[int, "RankCosts"] | None = None
    # Bytes of weights one decode step reads for one request across every rank: dense weights
    # plus the experts one token activates, at their stored precision. Zero leaves the serving
    # contract without the value, and dashboards then show no weight bandwidth.
    decode_weight_bytes: int = 0
    plain_ranks: Mapping[int, "RankCosts"] | None = None
    # API model name advertised for this checkpoint; empty uses the recipe's `served_model_name`.
    served_model_name: str = ""


@dataclass(frozen=True)
class Speculation:
    """Multi-token speculation: how many draft tokens a round verifies and where drafts come from.

    A round verifies the pending input and up to `draft_tokens` drafts.
    Preparation captures width one at every request bucket and complete rounds
    within the target-token capacity. `adaptive` sets the serving default: choose
    prepared layouts from measured costs and conditional acceptance estimates
    when true, or use the fixed maximum verification width when false. Core
    loading accepts capacity reductions represented by captured graphs. Drafts
    come from the checkpoint's learned draft model.
    """

    draft_tokens: int
    adaptive: bool = True
    token_capacity: int | None = None
    packed_verification: bool = False
    draft_prefix_selection: str = "cpu"
    graph_budget: int | None = None
    draft_head_quantization: str | None = None
    markov_head_quantization: str | None = None
    vocabulary_parallel_sampling: bool = False
    method: str | None = None
    mtp_only: bool = False
    draft_body_quantization: str | None = None

    def __post_init__(self):
        if self.mtp_only and self.draft_tokens < 1:
            raise ValueError("--mtp-only requires positive learned draft depth")
        if self.method not in (None, "mtp", "dspark", "dflash2"):
            raise ValueError("speculative method must be omitted, mtp, dspark or dflash2")
        if self.vocabulary_parallel_sampling and self.draft_tokens < 1:
            raise ValueError("vocabulary-parallel sampling requires positive learned draft depth")
        if self.vocabulary_parallel_sampling and self.markov_head_quantization == "nvfp4":
            raise ValueError("vocabulary-parallel sampling requires BF16 or row-scaled FP8 Markov weights")
        for head, choices in (("draft", ("nvfp4", "mxfp8")), ("markov", ("nvfp4", "fp8"))):
            value = getattr(self, f"{head}_head_quantization")
            if value is not None and value not in choices:
                raise ValueError(f"{head} head quantization must be omitted or one of: {', '.join(choices)}")
            if value is not None and self.draft_tokens < 1:
                raise ValueError("draft head quantization requires positive learned draft depth")
        if self.draft_body_quantization not in (None, "mxfp8"):
            raise ValueError("draft body quantization must be omitted or mxfp8")
        if self.draft_body_quantization is not None and self.draft_tokens < 1:
            raise ValueError("draft body quantization requires positive learned draft depth")
        if self.graph_budget is not None:
            if type(self.graph_budget) is not int or self.graph_budget < 1:
                raise ValueError("speculative graph budget must be a positive integer")
            if self.draft_tokens < 1:
                raise ValueError("speculative graph budget requires positive learned draft depth")
        if self.draft_prefix_selection not in ("cpu", "gpu"):
            raise ValueError("draft prefix selection must be cpu or gpu")
        if self.draft_prefix_selection == "gpu" and (
                not self.packed_verification or not self.adaptive or self.draft_tokens < 1):
            raise ValueError("GPU draft prefix selection requires packed adaptive learned speculation")

    @property
    def widths(self) -> tuple[int, ...]:
        """Verification widths: each draft count from zero through `draft_tokens`, plus the pending input."""
        return tuple(range(1, self.draft_tokens + 2))

    @property
    def fixed_width(self) -> int | None:
        return None if self.adaptive else self.draft_tokens + 1


@dataclass(frozen=True)
class RankCosts:
    """Device bytes one tensor-parallel rank spends per unit of capacity."""

    state_row_bytes: int
    page_bytes: int
    graph_bytes: int
    # Global positions per page when ranks share each page's context; the recipe's page size otherwise.
    page_tokens: int | None = None
    # Additional costs of a plan with image slots, for a recipe whose language-only
    # plans prepare no vision encoder. `vision_bytes` holds the fixed image tower
    # and upload storage, `vision_slot_bytes` each feature slot, and
    # `image_state_row_bytes` the continuation row that replaces `state_row_bytes`;
    # None keeps the same row.
    vision_bytes: int = 0
    vision_slot_bytes: int = 0
    image_state_row_bytes: int | None = None

    def with_image_slots(self, slots: int) -> "RankCosts":
        """Costs of a plan with `slots` image slots: the image row, and the vision storage as graph bytes."""
        if not slots or (self.image_state_row_bytes is None and not self.vision_bytes and not self.vision_slot_bytes):
            return self
        return replace(self, state_row_bytes=self.image_state_row_bytes or self.state_row_bytes,
            graph_bytes=self.graph_bytes + self.vision_bytes + slots * self.vision_slot_bytes)


@dataclass(frozen=True)
class RankUsage:
    """What one rank allocated and left free after preparation."""

    pages: int
    free_bytes: int


@dataclass(frozen=True)
class SpeculativeMethod:
    """Physical preparation costs for a recipe's learned proposal source."""

    ranks: Mapping[int, RankCosts]
    draft_bytes: int | None = None


@dataclass(frozen=True)
class KvCacheFormat:
    """Per-rank KV page byte overrides for an alternative KV cache storage format.

    Keyed in `Recipe.kv_cache_formats` by the `--kv-cache-format` CLI value.
    `page_bytes` maps each recipe tensor-parallel size to its per-page byte cost.
    When `plain_page_bytes` is non-empty, it overrides `page_bytes` whenever a plan
    runs without speculation (`speculation is None`, including `--no-mtp` and
    non-speculating recipes); when empty, `page_bytes` applies to both speculative
    and non-speculative plans. All other `RankCosts` fields, checkpoint cost
    overrides, and `speculative_methods` costs are preserved.
    """

    page_bytes: Mapping[int, int]
    plain_page_bytes: Mapping[int, int] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Recipe:
    """One model family's qualified checkpoints, serving range, costs, and bootstrap.

    `prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress)` loads,
    tunes, and captures the plan's serving catalog and returns the prepared model.
    `usage(prepared)` reports each rank's device memory in rank order.
    `served_model_name` is the canonical model name advertised by the API,
    independent of checkpoint selection aliases and storage formats.

    `ranks` holds each tensor-parallel size's costs as the recipe serves by
    default. A recipe that declares `speculation` drafts with those settings
    unless a plan overrides them, and `plain_ranks` holds its costs with
    speculation disabled. Each active request has one execution state row per
    selected verification position unless `verification_rows` gives another count
    of extra rows.
    `stream_overlap` states that captured graphs run independent branches on an
    auxiliary stream, which a plan can serialize onto the main stream.
    `online_mxfp8` states that the recipe names BF16 projections a plan may store as
    MXFP8 while loading, trading checkpoint precision for decode bandwidth.
    `online_mxfp8_scope` selects the conversion group for preparation; `all` selects
    the complete eligible set. `online_mxfp8_scopes` also accepts groups recorded in
    prepared metadata so an explicit precision choice retains its meaning.
    `language_model_only` states that a plan may prepare the recipe without its vision
    encoder, rejecting images and leaving the encoder's memory to KV pages.
    `mla_compute_modes` declares the attention arithmetic constraints a plan may select,
    independently of KV storage encoding and kernel tuning. `mla_compute_mode` and
    `expert_scale_compression` supply defaults when their preparation flags are omitted.
    `kv_cache_format` names the recipe's default KV storage format when the recipe
    declares alternatives in `kv_cache_formats` (or `None` when the recipe does not
    expose `--kv-cache-format`). Recipes that declare `kv_cache_formats` read
    `plan.kv_cache_format` (forwarded into `BasePreparationConfig.kv_cache_format`
    by `common_preparation_kwargs`) during `prepare` to allocate the matching KV
    cache storage layout and attention kernels.
    `export_repository` names the Hugging Face destination and the default local snapshot
    namespace. Export writes locally; it does not upload or select a signing identity.
    `b12x_environment` holds the b12x settings the recipe prepares with. b12x reads them
    only from its environment, so Ignition sets them for the preparing process.
    `row_shard_prefill_world_sizes` lists the tensor-parallel world sizes (at least two ranks)
    that support reducing prefill branch outputs into contiguous per-rank row shards; empty
    leaves `--row-shard-prefill` unsupported.
    """

    name: str
    served_model_name: str
    checkpoints: tuple[Checkpoint, ...]
    context: int
    page_tokens: int
    concurrency: int
    max_concurrency: int
    max_draft_tokens: int
    prefill_chunk: int
    ranks: Mapping[int, RankCosts]
    prepare: Callable
    usage: Callable
    sampling_defaults: SamplingDefaults
    mapped_host_name: str = ""
    max_images: int = 0
    image_bytes: int = 0
    speculation: Speculation | None = None
    speculative_methods: Mapping[str, SpeculativeMethod] = field(default_factory=lambda: MappingProxyType({}))
    plain_ranks: Mapping[int, RankCosts] = field(default_factory=lambda: MappingProxyType({}))
    verification_rows: int | None = None
    stream_overlap: bool = False
    mixed_prefill: bool = False
    online_mxfp8: bool = False
    online_mxfp8_scope: str = "all"
    online_mxfp8_scopes: tuple[str, ...] = ("all",)
    language_model_only: bool = False
    b12x_environment: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    mla_compute_modes: tuple[str, ...] = ("auto",)
    mla_compute_mode: str = "auto"
    expert_scale_compression: str | None = None
    kv_cache_format: str | None = None
    kv_cache_formats: Mapping[str, KvCacheFormat] = field(default_factory=lambda: MappingProxyType({}))
    export_repository: str = ""
    # Preparation overrides for vocabulary and transition projections used only by proposals.
    draft_head_quantizations: tuple[str, ...] = ()
    draft_body_quantizations: tuple[str, ...] = ()
    markov_head_quantizations: tuple[str, ...] = ()
    # Supported (world_size, context_parallel_size) hybrid context-parallel topologies.
    hybrid_context_parallel: tuple[tuple[int, int], ...] = ()
    # Optional (max_bytes, max_pages) graph-memory credit for hybrid context-parallel topologies.
    hybrid_context_parallel_graph_credits: Mapping[tuple[int, int], tuple[int, int]] = field(
        default_factory=lambda: MappingProxyType({}))
    # Tensor-parallel world sizes that prepare standalone benchmark graph subsets.
    benchmark_world_sizes: tuple[int, ...] = ()
    # Whether standalone benchmark preparation can run the multi-width speculation study.
    speculation_study: bool = False
    # Whether standalone benchmark preparation prepares text-only graphs without a vision encoder.
    benchmark_language_model_only: bool = False
    # Total expert intermediate width when W4A16 SM120/121 expert decode override is supported.
    moe_decode_a16_intermediate_width: int | None = None
    # Whether tensor-parallel preparation can shard hyperconnection residuals across ranks.
    shard_hyperconnections: bool = False
    # Whether preparation supports joint decode/prefill kernel tuning.
    joint_tuning: bool = False
    # Tensor-parallel world sizes that support vocabulary-parallel speculative sampling.
    vocabulary_parallel_world_sizes: tuple[int, ...] = ()
    # Tensor-parallel world sizes that support row-sharded prefill reductions and gathers.
    row_shard_prefill_world_sizes: tuple[int, ...] = ()

    def mixed_shapes(self, concurrency, chunk, speculation):
        from ignition.components.capacity import mixed_shapes, serving_modes

        if not self.mixed_prefill:
            return ()
        modes = serving_modes(speculation is not None, speculation is not None and speculation.mtp_only)
        return tuple((enabled, b, w, p) for enabled in modes
            for b, w, p in mixed_shapes(concurrency, chunk, speculation.widths if enabled else (1,),
                                        speculation.token_capacity if enabled else None))

    def __post_init__(self):
        if self.online_mxfp8_scope not in self.online_mxfp8_scopes:
            raise ValueError("recipe online MXFP8 scope must belong to its supported conversion groups")
        if self.export_repository:
            import re
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*', self.export_repository) or '..' in self.export_repository:
                raise ValueError('export_repository must be a Hugging Face namespace/repository')
        if any(not name.startswith("B12X_") or not isinstance(value, str)
               for name, value in self.b12x_environment.items()):
            raise ValueError("b12x settings name B12X_ variables with string values")
        if self.speculation is not None and set(self.plain_ranks) != set(self.ranks):
            raise ValueError("a recipe that speculates declares its costs without speculation for every rank count")
        if any(type(size) is not int or size < 2 or size not in self.ranks
               for size in self.row_shard_prefill_world_sizes):
            raise ValueError("row_shard_prefill_world_sizes must name tensor-parallel sizes >= 2 declared in ranks")
        for name, method in self.speculative_methods.items():
            if name not in ("mtp", "dspark", "dflash2") or set(method.ranks) != set(self.ranks):
                raise ValueError("speculative methods require known names and every declared tensor-parallel size")
            if method.draft_bytes is not None and method.draft_bytes <= 0:
                raise ValueError("speculative draft bytes must be positive")
        if self.kv_cache_format is not None and not self.kv_cache_format:
            raise ValueError("kv_cache_format must be None or a non-empty format name")
        if self.kv_cache_formats and self.kv_cache_format is None:
            raise ValueError("recipes declaring kv_cache_formats must declare a non-empty default kv_cache_format")
        expected_sizes = set(self.ranks)
        for name, fmt in self.kv_cache_formats.items():
            if not name or name == self.kv_cache_format:
                raise ValueError("alternative KV cache formats must have non-empty names distinct from the default")
            for field_name, pages in (("page_bytes", fmt.page_bytes), ("plain_page_bytes", fmt.plain_page_bytes)):
                if field_name == "plain_page_bytes" and not pages:
                    continue
                if set(pages) != expected_sizes:
                    missing = sorted(expected_sizes - set(pages))
                    extra = sorted(set(pages) - expected_sizes)
                    raise ValueError(
                        f"kv_cache_formats[{name!r}].{field_name} must cover every recipe tensor-parallel size"
                        f" (missing={missing}, unexpected={extra})"
                    )
                if any(type(nbytes) is not int or nbytes <= 0 for nbytes in pages.values()):
                    raise ValueError(f"kv_cache_formats[{name!r}].{field_name} values must be positive integers")
        for checkpoint in self.checkpoints:
            if any(costs is not None and (not costs or not set(costs) <= set(self.ranks))
                   for costs in (checkpoint.ranks, checkpoint.plain_ranks)):
                raise ValueError(f"{checkpoint.alias} declares costs for tensor-parallel sizes its recipe does not serve")
            if (checkpoint.ranks is None) != (checkpoint.plain_ranks is None) or (
                    checkpoint.ranks is not None and set(checkpoint.ranks) != set(checkpoint.plain_ranks)):
                raise ValueError(f"{checkpoint.alias} declares costs with and without speculation for different sizes")

    @property
    def tensor_parallel(self) -> tuple[int, ...]:
        return tuple(sorted(self.ranks))

    def checkpoint_tensor_parallel(self, checkpoint: Checkpoint) -> tuple[int, ...]:
        """Tensor-parallel sizes `checkpoint` prepares: its own declared sizes, else the recipe's."""
        return self.tensor_parallel if checkpoint.ranks is None else tuple(sorted(checkpoint.ranks))

    def served_name(self, checkpoint: Checkpoint) -> str:
        return checkpoint.served_model_name or self.served_model_name

    @property
    def supported_kv_cache_formats(self) -> tuple[str, ...]:
        """Declared KV cache formats (default first), or empty when selection is unsupported."""
        if self.kv_cache_format is None:
            return ()
        return (self.kv_cache_format, *self.kv_cache_formats)

    def costs(self, world_size: int, speculation: Speculation | None, checkpoint: Checkpoint | None = None,
              kv_cache_format: str | None = None) -> RankCosts:
        """Return one rank's costs under the given speculation and KV cache settings."""
        ranks = self.ranks if checkpoint is None or checkpoint.ranks is None else checkpoint.ranks
        plain = self.plain_ranks if checkpoint is None or checkpoint.plain_ranks is None else checkpoint.plain_ranks
        if speculation is None:
            base = (plain or ranks)[world_size]
        else:
            method = self.speculative_methods.get(speculation.method)
            base = ranks[world_size] if method is None else method.ranks[world_size]
        if kv_cache_format is None or kv_cache_format == self.kv_cache_format:
            return base
        if kv_cache_format not in self.kv_cache_formats:
            raise ValueError(f"unknown KV cache format {kv_cache_format!r} for recipe {self.name}")
        fmt = self.kv_cache_formats[kv_cache_format]
        pages = fmt.plain_page_bytes if speculation is None and fmt.plain_page_bytes else fmt.page_bytes
        return replace(base, page_bytes=pages[world_size])

    def weights_bytes(self, checkpoint: Checkpoint, world_size: int, speculation: Speculation | None,
                      expert_scale_compression: str | None = None) -> int:
        """Estimate one rank's device weights from the checkpoint's stored tensors and their expansion."""
        learned = speculation is not None
        device_bytes = (checkpoint.device_bytes if expert_scale_compression is None else
                        checkpoint.scale_compressed_device_bytes[expert_scale_compression])
        draft_bytes = checkpoint.draft_bytes
        method = None if speculation is None else self.speculative_methods.get(speculation.method)
        if method is not None and method.draft_bytes is not None:
            draft_bytes = method.draft_bytes
        return -(-(device_bytes + checkpoint.expansion_bytes - checkpoint.draft_bytes
                   + (draft_bytes if learned else 0)) // world_size)

    def live_rows(self, concurrency: int, speculation: Speculation | None) -> int:
        """Permanently backed null, live, and speculative execution rows."""
        slots = request_capacity(concurrency)
        speculative = (0 if speculation is None else speculation.draft_tokens if self.verification_rows is None
                       else self.verification_rows)
        return 1 + slots * (1 + speculative)
