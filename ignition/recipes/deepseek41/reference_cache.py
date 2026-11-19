# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete continuation storage exposing DeepInfer's attention cache views."""

from dataclasses import dataclass
from math import lcm

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout, gather_rows, scatter_rows
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tensor_metadata import record_numeric_tensor


@dataclass(frozen=True)
class ReferenceLayerCache:
    window: str | None
    latent: str | None
    latent_state: str | None
    index: str | None
    index_state: str | None
    ratio: int
    latent_bytes: int
    index_bytes: int


@dataclass(frozen=True)
class ReferenceCacheLayout:
    continuation: StateLayout
    pages: tuple
    layers: tuple[ReferenceLayerCache, ...]
    window_tokens: int
    verification_width: int
    page_tokens: int

    @classmethod
    def from_config(cls, cfg, *, verification_width, page_tokens=128, dspark=True):
        from deepinfer.model_executor.layer_utils import iter_runtime_attn_modules
        from deepinfer.model_executor.layers.tilelang_ops.cache_common import V41_FORMAT, V41_FP4_FORMAT
        from deepinfer.model_executor.models.v4_modules.cross_layer import build_shared_layer_topology

        if (type(verification_width) is not int or verification_width < 1
                or type(page_tokens) is not int or page_tokens < 1
                or type(dspark) is not bool):
            raise ValueError("reference cache requires positive page and verification capacities")
        modules = tuple(iter_runtime_attn_modules(cfg.attn_module_list, cfg.num_hidden_layers))
        if not dspark:
            modules = modules[:cfg.num_hidden_layers]
        topology = build_shared_layer_topology(cfg, len(modules))
        dimension = cfg.kv_lora_rank + cfg.qk_rope_head_dim
        window = cfg.window_size[0]
        if window <= 0 or dimension != V41_FORMAT.num_elements:
            raise ValueError("reference cache requires the checkpoint's declared sliding-window MLA geometry")
        window_tokens = window + verification_width - 1
        latent_bytes = (V41_FP4_FORMAT if cfg.use_fp4_sparse_kv else V41_FORMAT).bytes_per_token
        specs, pages, layers = [], [], []
        for layer, _, module in modules:
            c = module.cfg
            ratio = c.get("kv_compress_ratio", 0)
            if ratio not in (0, 1, 2) or (ratio and page_tokens % ratio):
                raise ValueError("reference cache page must contain whole C1/C2 compression blocks")
            names = dict(window=None, latent=None, latent_state=None, index=None, index_state=None)
            if c.get("sink_mode", "") == "swa":
                names["window"] = f"layer{layer}.window"
                specs.append((names["window"], (window_tokens, 1, V41_FORMAT.bytes_per_token), torch.uint8))
            elif not ratio:
                raise ValueError("reference cache requires sliding-window or compressed attention")
            index_dim = c.attn_index.index_head_dim if c.get("index_topk", 0) else 0
            if index_dim and not ratio:
                raise ValueError("reference index cache binding requires compressed attention geometry")
            if index_dim % 32:
                raise ValueError("reference FP4 index cache requires complete 32-element scale blocks")
            index_bytes = index_dim // 2 + index_dim // 32
            for kind, source, record_bytes, state_dim, needs_state in (
                ("latent", layer if topology is None else topology.layer_kv_map[layer], latent_bytes,
                 dimension, ratio > 1),
                ("index", layer if topology is None else topology.layer_index_k_map[layer], index_bytes,
                 index_dim, ratio > 1 and not c.get("attn_index", {}).get("use_kproj", False)),
            ):
                if not ratio or (kind == "index" and not index_dim):
                    continue
                if source != layer:
                    if not 0 <= source < len(layers) or layers[source].ratio != ratio:
                        raise ValueError("reference shared cache source must precede its compatible consumer")
                    names[kind] = getattr(layers[source], kind)
                    if names[kind] is None or getattr(layers[source], kind + "_bytes") != record_bytes:
                        raise ValueError("reference shared cache has incompatible record geometry")
                else:
                    names[kind] = f"layer{layer}.{kind}_page"
                    pages.append((names[kind], (page_tokens // ratio * record_bytes,)))
                    if needs_state:
                        names[kind + "_state"] = f"layer{layer}.{kind}_history"
                        specs.append((names[kind + "_state"],
                                      (ratio + verification_width - 1, 2, state_dim), torch.float32))
            layers.append(ReferenceLayerCache(**names, ratio=ratio,
                latent_bytes=latent_bytes, index_bytes=index_bytes))
        # Partial pages and every history ring participate in continuation copies.
        specs.extend((name, shape, torch.uint8) for name, shape in pages)
        specs.extend((("last_hidden", (cfg.hidden_size,), torch.bfloat16),
                      ("consumed_tokens", (1,), torch.int64), ("engram_history", (3,), torch.int64)))
        layout = StateLayout.from_specs(specs)
        # FlashMLA flattens cache blocks into whole encoded-token rows for TMA.
        stride = lcm(256, V41_FORMAT.bytes_per_token)
        layout = StateLayout(layout.regions, (layout.row_bytes + stride - 1) // stride * stride)
        return cls(layout, tuple(pages), tuple(layers),
                   window_tokens, verification_width, page_tokens)


@dataclass(frozen=True)
class ReferenceCacheStorage(PagedStateStorage):
    geometry: ReferenceCacheLayout

    @classmethod
    def allocate(cls, geometry, slots, pages, device, **kwargs):
        storage = PagedStateStorage.allocate(geometry.continuation, geometry.pages, slots, pages, device, **kwargs)
        return cls(storage.backing, storage.layout, storage.memory_capacity,
                   storage.pages, storage.reserved_backing, geometry)

    def cache_views(self):
        """Return reference operands aliasing complete native-managed storage."""
        from deepinfer.worker.cache_engine import KVCache

        page_views = {}
        layers = []
        for layer in self.geometry.layers:
            fields = {}
            for kind, field in (("latent", "ca"), ("index", "ica")):
                name = getattr(layer, kind)
                if name is not None and name not in page_views:
                    page_views[name] = self.pages[name].view(self.pages[name].shape[0],
                        self.geometry.page_tokens // layer.ratio, 1, getattr(layer, kind + "_bytes"))
                fields[field + "_cache"] = None if name is None else page_views[name]
                state = getattr(layer, kind + "_state")
                fields[field + "_state"] = None if state is None else self.reserved_region(state)
            fields["swa_cache"] = None if layer.window is None else self.reserved_region(layer.window)
            layers.append(KVCache(**fields))
        return tuple(layers)


class ReferenceCacheTransaction:
    """Retain cache writes only for the accepted verification input prefix.

    Snapshot continuation rings before draft execution can overwrite them and
    retain only the compressed page rows that verification may write. Temporary
    snapshots belong to complete-forward storage planning. Native execution owns
    the backing storage and never performs this model computation.
    """

    def __init__(self, pool):
        from deepinfer.model_executor.layers.tilelang_ops.cache_common import (
            get_kv_cache_format, split_kv_cache,
        )

        self.pool = pool
        rings, pages = {}, {}
        for layer in pool.geometry.layers:
            for name in (layer.window, layer.latent_state, layer.index_state):
                if name is not None:
                    rings[name] = pool.reserved_region(name)
            for name, size in ((layer.latent, layer.latent_bytes), (layer.index, layer.index_bytes)):
                if name is not None and name not in pages:
                    backing = pool.pages[name]
                    cache = backing.view(backing.shape[0], pool.geometry.page_tokens // layer.ratio, 1, size)
                    layout = get_kv_cache_format(cache)
                    sections = (cache,) if layout.token_interleaved else tuple(
                        part for part in split_kv_cache(cache, layout)[1] if part is not None)
                    parts = []
                    for section in sections:
                        part = section.view(torch.uint8)
                        width = part.shape[-1]
                        offset = part.storage_offset() - backing.storage_offset()
                        if backing.shape[1] % width or offset % width:
                            raise ValueError("reference cache section must divide its physical page stride")
                        # Index values and scales occupy separate page sections.
                        # Address each section in its own contiguous byte rows.
                        parts.append((backing.view(-1, width), layer.ratio,
                                      backing.shape[1] // width, offset // width))
                    pages[name] = parts
        self.rings = tuple(rings.values())
        self.pages = tuple(part for parts in pages.values() for part in parts)
        device = pool.reserved_backing.device
        ring_names = tuple(rings)
        if ring_names:
            rings_end = max(pool.layout.region(name).offset + pool.layout.region(name).nbytes for name in ring_names)
            if rings_end % 8 or pool.layout.row_bytes % 8:
                raise ValueError("reference continuation ring layout must be 8-byte aligned")
            words = rings_end // 8
            col = torch.full((words,), -1, dtype=torch.int32, device=device)
            mod = torch.ones((words,), dtype=torch.int32, device=device)
            for name in ring_names:
                reg = pool.layout.region(name)
                ring_len = reg.shape[0]
                if reg.offset % 8 or reg.nbytes % (ring_len * 8):
                    raise ValueError(f"reference ring {name} slot bytes must be 8-byte aligned")
                slot_words = reg.nbytes // (ring_len * 8)
                begin_word = reg.offset // 8
                end_word = (reg.offset + reg.nbytes) // 8
                col[begin_word:end_word] = torch.arange(
                    ring_len, dtype=torch.int32, device=device).repeat_interleave(slot_words)
                mod[begin_word:end_word] = ring_len
            self.ring_backing = pool.reserved_backing[:, :rings_end].view(torch.int64)
            self.ring_col = persistent_copy(col)
            self.ring_mod = persistent_copy(mod)
            if device.type == "cuda":
                record_numeric_tensor(self.ring_col)
                record_numeric_tensor(self.ring_mod)
        else:
            self.ring_backing = self.ring_col = self.ring_mod = None

    def begin(self, metadata):
        """Snapshot validated request rows before either draft or target writes."""
        m = metadata
        if self.ring_backing is not None:
            rings = gather_rows(self.ring_backing, m.state_rows, torch.empty(
                (m.state_rows.numel(), self.ring_backing.shape[1]),
                dtype=torch.int64, device=self.ring_backing.device))
        else:
            rings = ()
        pages = []
        address_cache = {}
        position = m.positions.clamp_min(0)
        physical = m.token_page_table.gather(1,
            (position // self.pool.geometry.page_tokens)[:, None]).flatten().long()
        for value, ratio, stride, offset in self.pages:
            key = (ratio, stride, offset)
            if key not in address_cache:
                per_page = self.pool.geometry.page_tokens // ratio
                rows = physical.clamp_min(0) * stride + offset + position // ratio % per_page
                written = (m.positions >= 0) & ((position + 1) % ratio == 0)
                address_cache[key] = (rows, written)
            else:
                rows, written = address_cache[key]
            pages.append((rows, written, value[rows].clone()))
        return rings, tuple(pages)

    def commit(self, metadata, counts, snapshot):
        """Restore rejected writes; inactive requests leave every backing byte intact."""
        m = metadata
        if counts.shape != m.lengths.shape or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("reference cache commit requires one accepted input count per request")
        rings, pages = snapshot
        if self.ring_backing is not None:
            current = gather_rows(self.ring_backing, m.state_rows, torch.empty_like(rings))
            col = self.ring_col[None, :].to(torch.int64)
            mod = self.ring_mod[None, :].to(torch.int64)
            starts = m.starts.to(torch.int64)[:, None]
            accepted = counts.to(torch.int64)[:, None]
            keep = (col < 0) | (((col - starts) % mod) < accepted)
            retained = torch.where(keep, current, rings)
            scatter_rows(retained, m.state_rows, self.ring_backing, m.lengths > 0)
        rejected = m.positions - m.starts[m.requests] >= counts[m.requests]
        for (value, _, _, _), (rows, written, before) in zip(self.pages, pages, strict=True):
            scatter_rows(before, rows, value, written & rejected)
