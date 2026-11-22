# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ignition PersistentWorker declaration, validation, and export retention."""

from contextlib import nullcontext
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch

from ignition.components.completion import VECTOR_TOKENS_FIELD
from ignition.preparation.execution import GraphSignature, PersistentWorker, PreparedExecution
from ignition.preparation.runtime_core import Allocation
from ignition.preparation.tensor_metadata import NumericStorage, live_metadata, merge_ranges, subtract_ranges


class _FakeGraph:
    def __init__(self, handle):
        self._handle = handle

    def raw_cuda_graph_exec(self):
        return self._handle


def _base_buffers(max_active=4, token_capacity=8):
    return {
        "frame_index": torch.zeros((1,), dtype=torch.int32),
        "submission_sequence": torch.zeros((1,), dtype=torch.int64),
        "completion_records": torch.zeros((2, max_active, VECTOR_TOKENS_FIELD + token_capacity), dtype=torch.int64),
        "worker_ctrl": torch.zeros((128,), dtype=torch.uint8),
    }


def _patch_bundle(monkeypatch, execution):
    def fake_validate(*, export_graphs=False):
        handles = [100 + i for i in range(len(execution.graphs))]
        spans = [(0x800_0000_0000 + i * 0x10000, t.numel() * t.element_size())
                 for i, t in enumerate(execution.buffers.values())]
        return (0x1234, handles, spans)

    monkeypatch.setattr(execution, "_validate_resources", fake_validate)


def test_persistent_worker_dataclass_validation():
    g = _FakeGraph(1)
    pw = PersistentWorker(
        name="mover",
        graph=g,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=0,
    )
    assert pw.worker_sm_count == 8
    assert not hasattr(pw, "metrics_buffer")

    with pytest.raises(ValueError, match="nonempty graph name"):
        PersistentWorker(name=" ", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl")
    with pytest.raises(ValueError, match="positive uint32"):
        PersistentWorker(name="mover", graph=g, worker_sm_count=0, stop_flag_buffer="worker_ctrl")
    with pytest.raises(ValueError, match="stop_flag_buffer"):
        PersistentWorker(name="mover", graph=g, worker_sm_count=8, stop_flag_buffer=" ")
    with pytest.raises(ValueError, match="4-byte-aligned"):
        PersistentWorker(name="mover", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl", stop_flag_offset=2)
    with pytest.raises(ValueError, match="4-byte-aligned"):
        PersistentWorker(name="mover", graph=g, worker_sm_count=8, stop_flag_buffer="worker_ctrl", stop_flag_offset=-4)


def test_prepared_execution_persistent_worker_manifest_and_rejection(monkeypatch):
    decode_graph = _FakeGraph(10)
    worker_graph = _FakeGraph(20)
    sig = GraphSignature(kind="decode", batch_capacity=4, token_capacity=1)
    pw = PersistentWorker(
        name="mover",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=0,
    )
    exe = PreparedExecution(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
        persistent_worker=pw,
    )
    _patch_bundle(monkeypatch, exe)
    assert "mover" in exe.graphs
    assert "mover" not in exe.serving_graphs

    live = exe.manifest()
    assert live["persistent_worker"] == {
        "worker_sm_count": 8,
        "graph": "mover",
        "stop_flag_buffer": "worker_ctrl",
        "stop_flag_offset": 0,
        "handle": 101,
    }

    exported = exe.export_manifest()
    assert exported["persistent_worker"] == {
        "worker_sm_count": 8,
        "graph": "mover",
        "stop_flag_buffer": "worker_ctrl",
        "stop_flag_offset": 0,
    }

    # Name collision with serving graph rejected
    with pytest.raises(ValueError, match="distinct from serving and tuning"):
        PreparedExecution(
            device=0,
            graphs={"mover": decode_graph},
            signatures={"mover": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_worker=pw,
        )

    # Unregistered stop_flag_buffer rejected
    bad_missing = PersistentWorker(
        name="mover", graph=worker_graph, worker_sm_count=8,
        stop_flag_buffer="missing_ctrl", stop_flag_offset=0,
    )
    with pytest.raises(ValueError, match="registered execution buffer"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_worker=bad_missing,
        )

    # Out-of-bounds stop_flag_offset rejected
    bad_oob = PersistentWorker(
        name="mover", graph=worker_graph, worker_sm_count=8,
        stop_flag_buffer="worker_ctrl", stop_flag_offset=128,
    )
    with pytest.raises(ValueError, match="exceeds its registered buffer"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_worker=bad_oob,
        )

    # Host-upload buffer rejected for stop flag
    bad_upload = PersistentWorker(
        name="mover", graph=worker_graph, worker_sm_count=8,
        stop_flag_buffer="frame_index", stop_flag_offset=0,
    )
    with pytest.raises(ValueError, match="cannot receive host uploads"):
        PreparedExecution(
            device=0,
            graphs={"decode": decode_graph},
            signatures={"decode": sig},
            buffers=_base_buffers(),
            max_active=4,
            persistent_worker=bad_upload,
        )


def test_prepared_execution_in_process_rejects_persistent_worker():
    decode_graph = _FakeGraph(10)
    worker_graph = _FakeGraph(20)
    sig = GraphSignature(kind="decode", batch_capacity=4, token_capacity=1)
    pw = PersistentWorker(
        name="mover",
        graph=worker_graph,
        worker_sm_count=8,
        stop_flag_buffer="worker_ctrl",
        stop_flag_offset=0,
    )
    exe = PreparedExecution(
        device=0,
        graphs={"decode": decode_graph},
        signatures={"decode": sig},
        buffers=_base_buffers(),
        max_active=4,
        persistent_worker=pw,
    )
    with pytest.raises(ValueError, match="green-context runtime control"):
        exe.start_executor()
    with pytest.raises(ValueError, match="green-context runtime control"):
        exe.serve_rank(SimpleNamespace(fileno=lambda: 0), rank=0, world=1)


def test_export_metadata_retains_mapped_host_weight_planes_via_descriptor_scan(monkeypatch):
    """Export numeric classification must omit control/layer_descs so mapped-host weight planes stay reachable."""
    import ignition.components.megamoe_tiers as megamoe_tiers
    import ignition.components.sm103_expert_mover as expert_mover
    import ignition.preparation.tensor_metadata as tensor_metadata
    from ignition.components.megamoe import MegaMoeGeometry
    from ignition.components.megamoe_weights import allocate_mxfp4_weights

    class _FakeHostRegion:
        def __init__(self, name, shape, dtype):
            self.name = name
            self.tensor = torch.zeros(shape, dtype=dtype)
            self.host_view = self.tensor
            self.device_view = self.tensor
            self.address = self.tensor.data_ptr()
            self.extent = self.tensor.nbytes
            self.closed = False

        def close(self):
            self.closed = True

    class _FakeStorage:
        device = 0

        def __init__(self):
            self.host_regions = []

        def host_allocate(self, name, shape, dtype):
            region = _FakeHostRegion(name, shape, dtype)
            self.host_regions.append(region)
            return region

    class _FakeMoverBinding:
        def configure_persistent_mover(self):
            pass

        def layer_desc_bytes(self):
            return 192

        def queue_header_bytes(self):
            return 64

        def queue_request_bytes(self):
            return 16

    real_device = torch.device
    monkeypatch.setattr(
        torch, "device",
        lambda *args, **kwargs: real_device("cpu") if args and args[0] == "cuda" else real_device(*args, **kwargs),
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _dev: SimpleNamespace(major=10, minor=3, multi_processor_count=152),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _dev: nullcontext())
    monkeypatch.setattr(megamoe_tiers, "persistent_copy", lambda tensor: tensor.clone())
    monkeypatch.setattr(megamoe_tiers, "persistent_zeros", lambda shape, dtype, device=None: torch.zeros(shape, dtype=dtype))
    monkeypatch.setattr(expert_mover, "persistent_copy", lambda tensor: tensor.clone())
    monkeypatch.setattr(expert_mover, "persistent_zeros", lambda shape, dtype, device=None: torch.zeros(shape, dtype=dtype))
    monkeypatch.setattr(expert_mover, "prepare_mover", lambda _cache: _FakeMoverBinding())

    storage = _FakeStorage()
    geom = MegaMoeGeometry(experts=16, capacity=64, top_k=2, hidden=256, intermediate=128, shared_experts=1)
    # Include pinned HBM-only experts (12..15) so host_experts < experts is also covered.
    hbm_pool = allocate_mxfp4_weights(8, geom.hidden, geom.intermediate, device="cpu")
    with NumericStorage() as numeric_storage:
        tier = megamoe_tiers.MegaMoeDynamicHostTier(
            geom,
            hbm_slots=8,
            initial_hbm_experts=(0, 1, 2, 12, 13, 14, 15),
            storage=storage,
            name="layer0",
            host_experts=12,
        )
        mover = expert_mover.SM103ExpertMoverRuntime(
            [(tier, hbm_pool)],
            storage=storage,
            kernel_cache=Path("/tmp/unused"),
            worker_sm_count=8,
            queue_capacity=32,
        )

    assert tier.rows == 12
    assert all(view.shape[0] == 12 for view in tier.host_views)

    host_allocs = [
        Allocation(f"host-{idx}", region.address, region.extent, alignment=4096,
                   preserve_address=True, memory="mapped_host")
        for idx, region in enumerate(storage.host_regions)
    ]
    device_tensors = {
        "control": tier.control,
        "layer_descs": mover.layer_descs,
        "live_slot_map": tier.live_slot_map,
        "slot_owner": tier.slot_owner,
        "layer_epoch": tier.layer_epoch,
        "descriptors": tier.descriptors,
        "queue_header": mover.queue_header,
        "queue_ring": mover.queue_ring,
    }
    device_allocs = [
        Allocation(name, tensor.data_ptr(), tensor.nbytes, alignment=256, preserve_address=True)
        for name, tensor in device_tensors.items()
    ]
    all_allocs = [*host_allocs, *device_allocs]
    tracked_objects = [*device_tensors.values(), *tier.device_views, mover.control_region.device_view]

    # Collect export numeric ranges using the same live_metadata pass as export_core.
    monkeypatch.setattr(tensor_metadata.gc, "get_objects", lambda: tracked_objects)
    monkeypatch.setattr(torch.Tensor, "device", property(lambda _: SimpleNamespace(type="cuda")))
    numeric_ranges, _ = live_metadata(all_allocs, numeric_storage, {"sm103_mover_control": mover.control_region.device_view})
    monkeypatch.undo()

    def scan_reached_mapped_host(tensor, excluded_ranges):
        base = tensor.data_ptr()
        raw = bytes(tensor.view(torch.uint8).reshape(-1).tolist())
        reached = set()
        for span_start, span_end in subtract_ranges([(base, base + len(raw))], excluded_ranges):
            chunk = raw[span_start - base:span_end - base]
            for off in range(0, max(0, len(chunk) - 7)):
                ptr = struct.unpack_from("<Q", chunk, off)[0]
                for alloc in host_allocs:
                    if alloc.address <= ptr < alloc.address + alloc.nbytes:
                        reached.add(alloc.name)
        return reached

    expected_weight_planes = {f"host-{idx}" for idx in range(4)}
    # Pointer-bearing descriptors remain unscrubbed so both serving and worker graphs retain all 4 host weight planes.
    assert expected_weight_planes <= scan_reached_mapped_host(tier.control, numeric_ranges)
    assert expected_weight_planes <= scan_reached_mapped_host(mover.layer_descs, numeric_ranges)
    # Numeric leaves (host weight planes and scalar tables) remain classified as numeric.
    for tensor in (*tier.device_views, tier.live_slot_map, tier.slot_owner, tier.layer_epoch,
                   tier.descriptors, mover.queue_header, mover.queue_ring, mover.control_region.device_view):
        start, end = tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes
        assert any(lo <= start and end <= hi for lo, hi in numeric_ranges)

    # Confirm the retention check fails if descriptor tensors are erroneously classified as numeric.
    for desc in (tier.control, mover.layer_descs):
        poisoned = merge_ranges(numeric_ranges + [(desc.data_ptr(), desc.data_ptr() + desc.nbytes)])
        assert not (expected_weight_planes & scan_reached_mapped_host(desc, poisoned))
