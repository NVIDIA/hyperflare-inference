# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native row exchanges preserve captured MegaMoE arithmetic and canonical routing.

Uses eight checkpoint experts with generated inputs, not a serving workload.
The qualification extension and exclusive SM103 ownership are required.
"""

import json
import os

import pytest
import torch

from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
from ignition.components.megamoe_tiers import MegaMoeHostTier
from ignition.preparation.compilation import compile_forward
from ignition.preparation.tuning import RecordedPreparationSession
from tests.components.test_megamoe_cuda import experts
from tests.components.test_megamoe_tiers_cuda import select

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned SM103 GPU",
)


@pytest.mark.parametrize("staged", [False, True], ids=["direct-host", "staged-host"])
def test_native_row_exchanges_preserve_checkpoint_moe(experts, tmp_path, megamoe_core_storage, staged):
    from flare_heliopause import ExpertResidencyTestStorage
    from ignition.components.megamoe_staging import PreparedExpertStage

    routed, shared = experts
    host_ids, hbm_ids = list(range(2, 8)), list(range(6))
    geometry = MegaMoeGeometry(12, 256, 6, 5120, 2304, 1)
    workspace = MegaMoeWorkspace(geometry, device="cuda:0", kernel_cache=tmp_path)
    geometry = workspace.geometry
    tier = MegaMoeHostTier(geometry, rows=6, storage=megamoe_core_storage,
                          name="row_exchange", host_lane_clusters=4)
    tier.copy_weights(select(routed, host_ids))
    hbm = select(routed, hbm_ids)
    reference = select(routed, list(range(8)) + list(range(4)))
    mapping = torch.tensor([6, 7, 8, 9, 10, 11, 4, 5], dtype=torch.int32, device="cuda")
    hbm_planes = (*hbm.gate_up, *hbm.down)
    reference_planes = (*routed.gate_up, *routed.down)
    planes = [(host.data_ptr(), device.data_ptr(), device.stride(0) * device.element_size())
              for host, device in zip(tier.device_views, hbm_planes)]
    addresses = [mapping.data_ptr(), *(p.data_ptr() for p in hbm_planes)]
    owner = (megamoe_core_storage, tier, hbm, mapping)
    storage = ExpertResidencyTestStorage.from_rows(owner, [(0, 8, host_ids, hbm_ids, mapping.data_ptr(), planes)])
    graphs, functions, catalog, transactions = [], [], [], []
    try:
        storage.exchange([], [(0, 5)])
        assert mapping.cpu().tolist() == [6, 7, 8, 9, 10, 3, 4, 5]
        with pytest.raises(RuntimeError, match="backing"):
            storage.exchange([], [(0, 0)])
        generator = torch.Generator(device="cuda").manual_seed(1741)
        with RecordedPreparationSession(device="cuda:0", checkpoint="deepseek41-eight-experts",
                workload_schema="native-row-exchanges", numerical_policy="reference-bitwise",
                compile_workers=2, autotune_effort="off") as session:
            stager = (PreparedExpertStage(tier, torch.cuda.Stream(), mapping, host_ids,
                session=session, name="row-stage", min_cold=2) if staged else None)
            for rows in (1, 16, 64, 256):
                x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16, generator=generator) * .1
                ids = (torch.arange(rows * 6, device="cuda") % 8).reshape(rows, 6).int()
                probabilities = torch.softmax(torch.randn(rows, 6, device="cuda", generator=generator), -1)
                padding = torch.zeros(rows, device="cuda", dtype=torch.bool)
                actual, expected = torch.empty_like(x), torch.empty_like(x)
                args = x, ids, probabilities, padding
                op = PreparedMegaMoe(*args, hbm, shared, workspace=workspace,
                                    host_tier=tier, clamp=10., session=session, name=f"moving-{rows}")
                ref = PreparedMegaMoe(*args, reference, shared, workspace=workspace,
                                     clamp=10., session=session, name=f"reference-{rows}")
                catalog.append((args, op, ref, actual, expected))
            session.tune()
            for args, op, ref, actual, expected in catalog:
                def forward(args=args, op=op, ref=ref, actual=actual, expected=expected):
                    x, ids, probabilities, padding = args
                    pending = None if stager is None else (stager, stager.start(x))
                    actual.copy_(op(x, mapping[ids.long()], probabilities, padding, staged=pending))
                    expected.copy_(ref(*args))
                fn = compile_forward(forward)
                fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    fn()
                functions.append(fn)
                graphs.append(graph)
            session.freeze()
            for step in range(16):
                incoming, outgoing = ((6, 2), (7, 3), (2, 6), (3, 7))[step % 4]
                torch.cuda.synchronize()
                free_before = torch.cuda.mem_get_info()[0]
                elapsed = storage.exchange([(0, incoming)], [(0, outgoing)])
                assert torch.cuda.mem_get_info()[0] == free_before, "movement allocated CUDA storage"
                transactions.append(elapsed)
                placement = mapping.cpu().tolist()
                assert placement[incoming] >= len(host_ids) and placement[outgoing] < len(host_ids)
                assert [mapping.data_ptr(), *(p.data_ptr() for p in hbm_planes)] == addresses
                for expert, row in enumerate(placement):
                    if row >= len(host_ids):
                        for source, destination in zip(reference_planes, hbm_planes):
                            assert torch.equal(destination[row - len(host_ids)].cpu(), source[expert].cpu())
                for source, backing in zip(reference_planes, tier.host_views):
                    assert torch.equal(backing, source[host_ids].cpu())
                for (args, _, _, actual, expected), graph in zip(catalog, graphs):
                    x, ids, _, padding = args
                    ids.copy_((torch.arange(ids.numel(), device="cuda").reshape_as(ids) + step) % 8)
                    padding.copy_(torch.arange(x.shape[0], device="cuda") % 3 == step % 3)
                    before = tier.control[16:].clone()
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    assert torch.isfinite(actual).all()
                    physical = mapping[ids[~padding].long()].long().flatten()
                    counts = torch.bincount(physical, minlength=geometry.experts)
                    if staged:
                        torch.testing.assert_close(tier.control[16:], before, rtol=0, atol=0)
                        assert stager.decision.item() == (counts[:tier.rows] > 0).sum().item()
                    else:
                        torch.testing.assert_close((tier.control[16:] - before).long(), counts, rtol=0, atol=0)
            print(json.dumps({"checkpoint_experts": 8, "shapes": [1, 16, 64, 256],
                "exchanges": len(transactions), "exact_output_parity": True,
                "transaction_wall_ns": transactions}))
    finally:
        torch.cuda.synchronize()
        for graph in graphs:
            graph.reset()
        storage.close()
        for region in tier.regions:
            region.close()
