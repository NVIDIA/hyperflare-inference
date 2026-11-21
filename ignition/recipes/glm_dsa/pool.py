# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Latent and index-key pages of every attention layer, and exact-prefix continuation rows."""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .config import INDEX_RECORD_BYTES, KERNEL_PAGE_RECORDS, LATENT_RECORD_BYTES, PAGE_RECORDS


def page_specs(config, *, draft=False):
    """One rank's record regions in one logical page.

    A page holds the PAGE_RECORDS positions this rank owns among the global
    positions the page covers, PAGE_RECORDS for each context rank. Every
    decoder layer stores the latent records of those positions, and a layer
    that owns an indexer also their index keys, so a page costs the same bytes
    whatever the number of context ranks. With `draft`, the draft layer's
    records and index keys follow the decoder's.
    """
    for layer in range(config.layers):
        yield f"layer{layer}.latent", (PAGE_RECORDS * LATENT_RECORD_BYTES,)
        if config.indexed(layer):
            yield f"layer{layer}.indexer", (PAGE_RECORDS * INDEX_RECORD_BYTES,)
    if draft:
        if not config.draft_layers:
            raise ValueError("this GLM MoE DSA checkpoint has no multi-token prediction layer")
        yield "draft.latent", (PAGE_RECORDS * LATENT_RECORD_BYTES,)
        yield "draft.indexer", (PAGE_RECORDS * INDEX_RECORD_BYTES,)


def page_bytes(config, *, draft=False):
    return sum(shape[0] for _, shape in page_specs(config, draft=draft))


def continuation_layout(config, *, draft=False):
    """One continuation row preserves each cache region's partial page and the final hidden vector."""
    regions = [(name, shape, torch.uint8) for name, shape in page_specs(config, draft=draft)]
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, *, draft=False, **kwargs):
        return cls.allocate(continuation_layout(config, draft=draft), page_specs(config, draft=draft),
                            slots, pages, device, **kwargs)

    def attention(self, layer):
        """The layer's latent records as b12x sparse-MLA kernel pages."""
        return self.pages[f"layer{layer}.latent"].view(-1, KERNEL_PAGE_RECORDS, LATENT_RECORD_BYTES)

    def indexer_pages(self, layer):
        """The layer's index keys as planar 64-record pages: every key's values, then every key's scale."""
        return self.pages[f"layer{layer}.indexer"].view(-1, KERNEL_PAGE_RECORDS * INDEX_RECORD_BYTES)

    def draft_attention(self):
        """The draft layer's latent records as b12x sparse-MLA kernel pages."""
        return self.pages["draft.latent"].view(-1, KERNEL_PAGE_RECORDS, LATENT_RECORD_BYTES)

    def draft_indexer_pages(self):
        """The draft layer's index keys as planar 64-record pages."""
        return self.pages["draft.indexer"].view(-1, KERNEL_PAGE_RECORDS * INDEX_RECORD_BYTES)

    def share_layer_pages(self, storage, group, kind):
        """Every rank's address of each layer's "latent" or "indexer" page pool, [world] int64 keyed by
        decoder layer, or by "draft" for the draft layer."""
        names = [name for name in self.pages if name.endswith("." + kind)]
        return {name.split(".")[0] if name.startswith("draft") else int(name.split(".")[0].removeprefix("layer")):
                pointers for name, pointers in self.share_pages(storage, group, names).items()}
