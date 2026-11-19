"""Forward objects of the exported gather-core fixture (helios/tests/gather_core_fixture.py)."""

import torch


class GatherForward:
    def __init__(self, operation, source):
        self.operation, self.source = operation, source

    def __call__(self):
        return self.operation(self.source)


class RecordForward:
    """Pack, gather and scatter one window sequence of latent records, as a prefill layer's ring does."""

    def __init__(self, windows, shards, cache, table, output, window):
        self.windows, self.shards, self.cache = windows, shards, cache
        self.table, self.output, self.window = table, output, window

    def __call__(self):
        from ignition.recipes.glm53.dcp import pack_records, scatter_records

        slab = torch.zeros_like(self.output)
        packed = torch.full((4, self.window * 528), 0xA5, dtype=torch.uint8, device=self.cache.device)
        end = torch.full((1,), 128, dtype=torch.int64, device=self.cache.device)
        for first, gather in self.windows:
            pack_records(self.cache.view(torch.int32).view(-1, 132), self.table, end,
                packed, self.shards, first)
            # Counts deliberately have no data dependency on the packed cache records.
            counts = torch.full((4,), min(max(32 - first, 0), self.window) * 528,
                dtype=torch.int64, device=self.cache.device)
            gather(output=packed, counts=counts)
            scatter_records(packed, end, slab.view(torch.int32), self.shards, first)
        self.output.copy_(slab)
