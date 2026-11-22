# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""In-process runtime-sized storage keeps its full shape while only a prefix is backed."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_core_reserves_persistent_state_and_cache_extents_without_full_backing():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    subprocess.run([sys.executable, "-c", """
from ignition import runtime_patches
runtime_patches.apply()
import torch
from ignition.preparation.core_storage import CoreStorage

torch.empty(1, device="cuda:0")
with CoreStorage(device=0) as storage:
    weight = storage._persistent_empty((16,), (1,), torch.float32, 64, 256)
    unit = storage.granularity
    maximum = torch.cuda.get_device_properties(0).total_memory // unit
    state = storage.device_allocate("state", (maximum, unit), torch.uint8,
                                    group="STATE_ROWS", maximum=maximum, units=1)
    pages = storage.device_allocate("pages", (maximum, unit), torch.uint8,
                                    group="KV_PAGES", maximum=maximum, units=1)
    regions = storage.persistent_regions + storage.dynamic_regions
    bounds = [(r.address, r.address + (storage._persistent_arena.capacity if r in storage.persistent_regions else r.extent))
              for r in regions]
    assert all(a[1] <= b[0] for a, b in zip(bounds, bounds[1:]))
    assert bounds[-1][1] <= storage.DEVICE_BASE + storage.RESERVATION_BYTES
    assert sum(r.nbytes for r in regions) < 32 << 20
    weight.fill_(3)
    state[:1].fill_(7)
    pages[:1].fill_(11)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            weight.add_(1)
            state[:1].add_(1)
            pages[:1].add_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert bool((weight == 4).all())
    assert bool((state[:1] == 8).all()) and bool((pages[:1] == 12).all())
    del graph, weight, state, pages
"""], check=True, timeout=90)


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_virtual_pages_grow_in_place_after_compilation():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    subprocess.run([sys.executable, "-c", """
import torch
from ignition.preparation.virtual_storage import VirtualStorage

torch.empty(1, device="cuda:0")
storage = VirtualStorage("cuda:0")
unit = storage.granularity // 2
pages = storage.device_allocate("kv", (1024, unit), torch.uint8, group="KV_PAGES", maximum=1024, units=4)
index = storage.device_allocate("index", (1024, 8), torch.uint8, group="KV_PAGES", maximum=1024, units=4)
other = storage.device_allocate("other", (16, unit), torch.uint8, group="OTHER", maximum=16, units=16)
assert pages.shape == (1024, unit) and storage.nbytes == 2 * storage.granularity + storage.granularity + 8 * storage.granularity
pages[:4].fill_(7)
index[:4].fill_(3)
address = pages.data_ptr()
storage.grow("KV_PAGES", 100)
torch.cuda.synchronize()
assert pages.data_ptr() == address and storage.nbytes == 50 * storage.granularity + storage.granularity + 8 * storage.granularity
assert bool((pages[:4] == 7).all()) and bool((pages[4:100] == 0).all())
assert bool((index[:4] == 3).all()) and bool((index[4:100] == 0).all())
pages[:100].add_(1)
torch.cuda.synchronize()
storage.close()
storage.close()
assert not storage.dynamic_regions
"""], check=True)
