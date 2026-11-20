# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Attention pages and continuation rows of GLM 5.3 requests on one rank.

Every decoder layer stores one 656-byte latent record per token. Layers with
an indexer also store one 132-byte index key per token. A page holds 256
tokens of every layer. A continuation row holds what a request needs beyond
its complete pages: each region's partial page, the final normalized hidden
state, and the consumed token count.
"""

import torch

from ignition.components.paged_state import PagedStateStorage
from ignition.components.state import StateLayout

from .attention import PAGE_RECORDS, PAGE_TOKENS, RECORD_BYTES
from .indexer import INDEX_RECORD_BYTES


def page_specs(config, *, draft=False):
    """Name and byte shape of every region of one page.

    The draft layer's regions are named `draft.*`: state maintenance of graphs
    that run without the draft layer leaves regions of that prefix untouched.
    """
    for index in range(config.layers + int(draft)):
        owner = "draft" if index == config.draft_layer else f"layer{index}"
        yield f"{owner}.latent", (PAGE_TOKENS * RECORD_BYTES,)
        if config.selects(index):
            yield f"{owner}.indexer", (PAGE_TOKENS * INDEX_RECORD_BYTES,)


def page_bytes(config, *, draft=False):
    return sum(shape[0] for _, shape in page_specs(config, draft=draft))


def continuation_layout(config, *, draft=False):
    regions = [(name, shape, torch.uint8) for name, shape in page_specs(config, draft=draft)]
    regions.extend((("last_hidden", (config.hidden,), torch.bfloat16), ("consumed_tokens", (1,), torch.int64)))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(cls, config, slots, pages, device, *, draft=False, **storage):
        return cls.allocate(continuation_layout(config, draft=draft), page_specs(config, draft=draft), slots, pages,
                            device, **storage)

    def _owner(self, layer):
        """Region prefix of a layer; the draft layer follows the target layers."""
        targets = sum(name.startswith("layer") and name.endswith(".latent") for name in self.pages)
        return "draft" if layer == targets and "draft.latent" in self.pages else f"layer{layer}"

    def latent(self, layer):
        """Latent records of one layer as four 64-record blocks per page."""
        return self.pages[self._owner(layer) + ".latent"].view(-1, PAGE_RECORDS, RECORD_BYTES)

    def index(self, layer):
        """Index keys of one layer as four planar 64-record blocks per page, or None without an indexer."""
        pages = self.pages.get(self._owner(layer) + ".indexer")
        return None if pages is None else pages.view(-1, PAGE_RECORDS * INDEX_RECORD_BYTES)
