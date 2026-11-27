# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Placeholders for process-local resources survive the pickling of compiled-forward guards."""

import os
import pickle

import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from ignition.preparation import forward_metadata
from ignition.preparation.forward_metadata import _Unavailable


def test_placeholder_round_trips_through_pickle():
    restored = pickle.loads(pickle.dumps(_Unavailable("RoCE proxy")))
    assert restored.description == "RoCE proxy"
    with pytest.raises(RuntimeError, match="RoCE proxy.post"):
        restored.post


def test_forward_metadata_replaces_cute_runtime_tensors_with_placeholders():
    # b12x bindings such as the TMEM KDA prefill binding keep CuTe DLPack descriptors of their scratch views.
    descriptor = from_dlpack(torch.zeros(4, 128, dtype=torch.bfloat16), assumed_align=16)
    restored = forward_metadata.loads(forward_metadata.dumps({"tmem_tensors": {"qd": descriptor}}))
    placeholder = restored["tmem_tensors"]["qd"]
    assert isinstance(placeholder, _Unavailable)
    with pytest.raises(RuntimeError, match="cutlass.cute.runtime._Tensor.shape"):
        placeholder.shape


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires an explicitly assigned CUDA device")
def test_collective_transport_metadata_leaves_cuda_resources_on_the_owning_rank():
    from b12x.comm.pcie._cuda_ipc import CudaRTLibrary
    from b12x.comm.pcie._island_copy import PacketTransport

    assert os.environ.get("CUDA_VISIBLE_DEVICES")
    values = torch.zeros(128, device="cuda:0")
    transport = PacketTransport(values.device, CudaRTLibrary())
    try:
        restored = forward_metadata.loads(forward_metadata.dumps(transport))
        for resource in (restored.stream, restored.started, restored.finished):
            with pytest.raises(RuntimeError, match="process-local resource"):
                resource.query()
        transport.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(transport.stream):
            values.fill_(3)
        transport.stream.synchronize()
        torch.testing.assert_close(values, torch.full_like(values, 3))
    finally:
        torch.cuda.synchronize(values.device)
        transport.close()
