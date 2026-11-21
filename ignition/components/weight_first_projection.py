# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared mixed-output projections that can stage weights before their input is ready."""

import torch
from b12x import gemm
from b12x.preparation import PreparedCall

from ignition.preparation.tuning import declare


class PreparedWeightFirstProjection:
    """Keep the first dot product in FP32 and round the second once to BF16."""

    @staticmethod
    def fits(source, first, second):
        """The kernel reads complete, aligned BF16 rows from both resident matrices."""
        matrices = (source, first, second)
        if any(not isinstance(t, torch.Tensor) or t.ndim != 2 or t.device.type != "cuda" or t.device != source.device
               or t.dtype != torch.bfloat16 or not t.is_contiguous() or t.data_ptr() % 16 for t in matrices):
            return False
        width = source.shape[1]
        return (1 <= source.shape[0] <= 16 and width > 0 and width % 256 == 0
                and all(w.shape[0] > 0 and w.shape[1] == width for w in (first, second)))

    def __init__(self, source, first, second, *, session, name):
        query = gemm.weight_first_projection.Query(source.shape[0], source.shape[1],
                                                   first.shape[0], second.shape[0], staging_depth=1, pdl=True)
        self.plan = gemm.weight_first_projection.plan(query)
        self.first, self.second = first, second
        self.partial_shape = query.partial_shape
        # The last split-K block restores each counter to zero before returning.
        # Keeping it across replays avoids launching a clear between the two kernels.
        self.counters = torch.zeros(query.counter_shape, dtype=torch.int32, device=source.device)

        def prepare(state):
            first_out, second_out, partials = self._outputs(source)
            return PreparedCall(run=lambda: state.run(source, first, second, first_out,
                                                      second_out, partials, self.counters),
                                output=(first_out, second_out))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))

    def _outputs(self, source):
        first = torch.empty((source.shape[0], self.first.shape[0]), device=source.device, dtype=torch.float32)
        second = torch.empty((source.shape[0], self.second.shape[0]), device=source.device, dtype=torch.bfloat16)
        partials = torch.empty(self.partial_shape, device=source.device, dtype=torch.float32)
        return first, second, partials

    def __call__(self, source):
        first, second, partials = self._outputs(source)
        gemm.weight_first_projection.project(source, self.first, self.second, first, second,
                                             partials, self.counters, self.plan.handle)
        return first, second
