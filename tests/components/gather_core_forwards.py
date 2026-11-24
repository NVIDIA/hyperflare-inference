"""Forward objects of the exported gather-core fixture (helios/tests/gather_core_fixture.py)."""

import torch


class GatherForward:
    def __init__(self, operation, source):
        self.operation, self.source = operation, source

    def __call__(self):
        return self.operation(self.source)


class RecordForward:
    """Pack, gather and scatter one window sequence of latent records, as a prefill layer's ring does."""

    def __init__(self, windows, rank_index, world, cache, table, output, window):
        self.windows, self.rank_index, self.world = windows, rank_index, world
        self.cache, self.table, self.output, self.window = cache, table, output, window

    def __call__(self):
        slab = torch.zeros_like(self.output)
        packed = torch.full((self.world, self.window * 528), 0xA5, dtype=torch.uint8, device=self.cache.device)
        cache_records = self.cache.view(-1, 528)
        end = 128
        span = 4 * self.world
        local_offsets = torch.arange(self.window, dtype=torch.int64, device=self.cache.device)
        owners = torch.arange(self.world, dtype=torch.int64, device=self.cache.device)
        for first, gather in self.windows:
            local = first + local_offsets
            pages = self.table[(local // 256).clamp(max=self.table.shape[0] - 1)].long()
            physical = pages * 256 + (local % 256)
            rank_records = cache_records.index_select(0, physical)
            count = (end // span) * 4 + (end % span - self.rank_index * 4).clamp(0, 4)
            live = (local < count).unsqueeze(-1)
            packed_view = packed.view(self.world, self.window, 528)
            existing = packed_view.index_select(0, self.rank_index).squeeze(0)
            packed_view.index_copy_(0, self.rank_index, torch.where(live, rank_records, existing).unsqueeze(0))
            # Counts deliberately have no data dependency on the packed cache records.
            counts = torch.full((self.world,), min(max(32 - first, 0), self.window) * 528,
                dtype=torch.int64, device=self.cache.device)
            gather(output=packed, counts=counts)
            gathered = packed.view(self.world, self.window, 528)
            tokens = (local[None, :] // 4) * span + owners[:, None] * 4 + (local[None, :] % 4)
            flat_tokens = tokens.reshape(-1)
            valid = flat_tokens < end
            safe_tokens = flat_tokens.clamp(0, slab.shape[0] - 1)
            slab_target = slab.index_select(0, safe_tokens)
            slab.index_copy_(0, safe_tokens, torch.where(valid.unsqueeze(-1), gathered.reshape(-1, 528), slab_target))
        self.output.copy_(slab)
