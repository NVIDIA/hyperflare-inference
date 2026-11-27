# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Layout constraints materialize a shared opaque-call operand once."""

import sympy
import torch
import pytest
from torch._inductor import ir, lowering
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V, ops as _ops

from ignition import runtime_patches
from ignition.preparation import compilation


def _graph(opaque=True, *, mutable=False):
    fx = torch.fx.Graph()
    source, output = fx.placeholder('source'), fx.placeholder('output')
    arguments = ([], [source], 0) if mutable else ([source], [output], 0)
    operation = (fx.call_function(torch.ops.flare.storage_call.default, arguments) if opaque
                 else fx.call_function(torch.ops.aten.sin.default, (source,)))
    fx.output(())
    graph = GraphLowering(torch.fx.GraphModule({}, fx), example_inputs=[], is_inference=True)
    graph.current_node = operation
    return graph


def _pointwise():
    return ir.Pointwise.create(device=torch.device('cpu'), dtype=torch.bfloat16,
        inner_fn=lambda index: _ops.constant(1., torch.bfloat16), ranges=[sympy.Integer(4), sympy.Integer(8)])


@pytest.mark.parametrize('mutable', [False, True])
def test_repeated_opaque_consumers_materialize_pointwise_once(mutable):
    runtime_patches.apply()
    graph = _graph(mutable=mutable)
    fake = torch.empty_strided((4, 8), (8, 1), dtype=torch.bfloat16, device='meta')
    with V.set_graph_handler(graph):
        source = _pointwise()
        results = [lowering.constrain_to_fake_tensor(source, fake) for _ in range(3)]
        assert len(graph.buffers) == 1
        assert len({result.get_name() for result in results}) == 1
        assert isinstance(source.data.data, ir.ComputedBuffer)
        assert all(tuple(result.get_stride()) == fake.stride() for result in results)


def test_nonopaque_pointwise_keeps_existing_constraint_behavior():
    runtime_patches.apply()
    graph = _graph(opaque=False)
    fake = torch.empty_strided((4, 8), (8, 1), dtype=torch.bfloat16, device='meta')
    with V.set_graph_handler(graph):
        source = _pointwise()
        results = [lowering.constrain_to_fake_tensor(source, fake) for _ in range(3)]
        assert len(graph.buffers) == 3
        assert len({result.get_name() for result in results}) == 3
        assert isinstance(source.data.data, ir.Pointwise)


def test_realized_views_preserve_aliases_and_incompatible_strides_still_copy():
    runtime_patches.apply()
    graph = _graph()
    with V.set_graph_handler(graph):
        source = _pointwise()
        source.realize()
        ir.ExternKernel.require_exact_strides(source, [8, 1])
        view = ir.ReinterpretView(data=source.data,
            layout=ir.FixedLayout(torch.device('cpu'), torch.bfloat16, size=[4, 4], stride=[8, 2], offset=0))
        fake = torch.empty_strided((4, 4), (8, 2), dtype=torch.bfloat16, device='meta')
        same = lowering.constrain_to_fake_tensor(view, fake)
        assert same.get_name() == source.get_name()
        assert tuple(same.get_stride()) == (8, 2)
        assert len(graph.buffers) == 1
        packed = torch.empty_strided((4, 4), (4, 1), dtype=torch.bfloat16, device='meta')
        copied = lowering.constrain_to_fake_tensor(view, packed)
        assert copied.get_name() != source.get_name()
        assert tuple(copied.get_stride()) == (4, 1)
        assert tuple(view.get_stride()) == (8, 2)
        assert len(graph.buffers) == 2


def test_absent_fake_operand_does_not_materialize_pointwise():
    runtime_patches.apply()
    graph = _graph()
    with V.set_graph_handler(graph):
        source = _pointwise()
        assert lowering.constrain_to_fake_tensor(source, None) is source
        assert isinstance(source.data.data, ir.Pointwise)
        assert not graph.buffers
