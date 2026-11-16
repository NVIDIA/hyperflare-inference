# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded checkpoint packing fills physical tier rows without persistent temporaries."""

import os

import pytest
import torch

from ignition.components.megamoe import MegaMoeGeometry
from ignition.components.megamoe_tiers import MegaMoeHostTier
from ignition.components.megamoe_weights import (
    allocate_mxfp4_weights, load_mxfp4_planes, load_mxfp4_weights_into, pack_mxfp4_planes,
)
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from tests.components.test_megamoe_weights_cuda import source

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned SM103 GPU")


def test_bounded_csf_loading_preserves_rows_and_retained_memory(source, megamoe_core_storage):
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires SM103")
    root, _ = source
    manifest = inspect_indexed_checkpoint(root)
    prefixes = ("experts.1", "experts.0")
    storage = megamoe_core_storage
    hbm = allocate_mxfp4_weights(2, 128, 256, device="cuda:0")
    tier = MegaMoeHostTier(MegaMoeGeometry(4, 8, 1, 128, 256, 0), rows=2,
                          storage=storage, name="loaded")
    retained = storage._persistent_arena.used
    try:
        with DirectWeightLoader(manifest, "cuda:0") as loader:
            expected = pack_mxfp4_planes(load_mxfp4_planes(loader, prefixes,
                hidden_size=128, intermediate_size=256), temporary=True)
            for batch in (1, 3):
                for target in (tier.cpu_weights, hbm):
                    for value in (*target.gate_up, *target.down):
                        value.zero_()
                    actual = load_mxfp4_weights_into(loader, prefixes, target,
                        hidden_size=128, intermediate_size=256, batch_experts=batch)
                    torch.cuda.synchronize()
                    assert actual is target
                    assert storage._persistent_arena.used == retained
                    for a, b in zip((*target.gate_up, *target.down), (*expected.gate_up, *expected.down)):
                        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
            with pytest.raises(ValueError, match="positive batch size"):
                load_mxfp4_weights_into(loader, prefixes, hbm, hidden_size=128,
                                        intermediate_size=256, batch_experts=0)
            with pytest.raises(ValueError, match="unique expert prefixes"):
                load_mxfp4_weights_into(loader, (prefixes[0], prefixes[0]), hbm,
                                        hidden_size=128, intermediate_size=256)
            with pytest.raises(ValueError, match="destination geometry"):
                load_mxfp4_weights_into(loader, prefixes, hbm, hidden_size=128,
                                        intermediate_size=256, intermediate_range=(0, 128))
    finally:
        torch.cuda.synchronize()
        for region in tier.regions:
            region.close()
