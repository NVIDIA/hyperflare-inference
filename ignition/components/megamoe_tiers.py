# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mapped host weight storage and per-layer control for the MegaMoE Grace tier."""

import torch

from ignition.components.megamoe_weights import MegaMoeWeights
from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor


class MegaMoeHostTier:
    """Own physical host rows; routing supplies host-first logical row identities.

    Rows below ``rows`` read registered host memory. Subsequent rows address the
    HBM matrices supplied to PreparedMegaMoe. Each layer owns its control block
    and tensor-map clone table, including when layers share protocol workspace.
    This storage does not choose residency or promote experts.
    """

    def __init__(self, geometry, *, rows, storage, name, host_lane_clusters=4):
        device = torch.device("cuda", storage.device)
        if (type(rows) is not int or not 0 < rows < geometry.experts
                or type(host_lane_clusters) is not int or host_lane_clusters < 0):
            raise ValueError("MegaMoE host storage requires nonempty host and HBM tiers and a nonnegative lane size")
        properties = torch.cuda.get_device_properties(device)
        if (properties.major, properties.minor) != (10, 3):
            raise ValueError("the MegaMoE Grace tier requires SM103")
        if host_lane_clusters >= properties.multi_processor_count // 2:
            raise ValueError("MegaMoE host lane must leave at least one cluster for HBM work")
        self.geometry, self.rows = geometry, rows
        self.host_lane_clusters = host_lane_clusters
        self.regions, self.host_views, self.device_views = [], [], []
        for label, width, columns, scale in (
            ("gate_up", 2 * geometry.intermediate, geometry.hidden, False),
            ("gate_up_scales", 2 * geometry.intermediate, geometry.hidden, True),
            ("down", geometry.hidden, geometry.intermediate, False),
            ("down_scales", geometry.hidden, geometry.intermediate, True),
        ):
            cols = columns // (128 if scale else 2)
            stride = (width * cols, 1, width) if scale else (width * cols, cols, 1)
            region = storage.host_allocate(f"{name}.{label}", (rows * width * cols,),
                                           torch.int32 if scale else torch.int8)
            self.regions.append(region)
            shape = (rows, width, cols)
            self.host_views.append(region.host_view.as_strided(shape, stride))
            self.device_views.append(region.device_view.as_strided(shape, stride))
        self.weights = MegaMoeWeights(tuple(self.device_views[:2]), tuple(self.device_views[2:]))
        for value in self.device_views:
            record_numeric_tensor(value)
        # The pinned kernel's table contains four 128-byte maps per CTA, at most 160 CTAs.
        self.descriptors = persistent_zeros((160 * 4 * 128,), dtype=torch.uint8, device=device)
        header = torch.zeros(16 + geometry.experts, dtype=torch.int32)
        header[0], header[1], header[10] = 0x53505244, rows, host_lane_clusters
        header[:16].view(torch.int64)[1:5] = torch.tensor([t.data_ptr() for t in self.device_views])
        header[:16].view(torch.int64)[7] = self.descriptors.data_ptr()
        self.control = persistent_copy(header.to(device))

    def copy_weights(self, weights, *, start=0):
        """Fill host rows during loading from a bounded batch of transformed experts."""
        source = (*weights.gate_up, *weights.down)
        count = source[0].shape[0]
        if type(start) is not int or count <= 0 or start < 0 or start + count > self.rows:
            raise ValueError("MegaMoE host weight copy exceeds the allocated expert rows")
        for src, dst in zip(source, self.host_views):
            if src.shape != (count, *dst.shape[1:]) or src.dtype != dst.dtype or src.stride() != dst.stride():
                raise ValueError("MegaMoE host weight copy requires the prepared physical layout")
        for src, dst in zip(source, self.host_views):
            dst[start:start + count].copy_(src)
