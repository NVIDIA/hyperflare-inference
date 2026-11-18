# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Binary arithmetic and view metadata preserve native layouts and aliases."""

import os

import pytest
import torch

from ignition import runtime_patches


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="explicit CUDA assignment required")


_LAYOUTS = {
    "nchw": ((2, 4, 3, 5), (60, 15, 5, 1), 0),
    "channels_last": ((2, 4, 3, 5), (60, 1, 20, 4), 0),
    "singleton_batch": ((1, 4, 3, 5), (4, 1, 20, 4), 0),
    "arbitrary_batch": ((1, 4, 3, 5), (7, 1, 20, 4), 0),
    "zero_batch": ((1, 4, 3, 5), (0, 1, 20, 4), 0),
    "slice": ((2, 4, 3, 5), (120, 30, 10, 2), 0),
    "offset": ((1, 4, 3, 5), (4, 1, 20, 4), 9),
}
_CASES = [
    ("div", "singleton_batch", "one"),
    ("mul", "arbitrary_batch", "fraction"),
    ("add", "singleton_batch", "cpu_scalar"),
    ("sub", "arbitrary_batch", "cuda_scalar"),
    ("div", "arbitrary_batch", "broadcast"),
    ("mul", "offset", "one"),
    ("add", "zero_batch", "cpu_scalar"),
    ("sub", "slice", "fraction"),
    ("add", "nchw", "same_layout"),
    ("sub", "channels_last", "nchw_tensor"),
    ("mul", "singleton_batch", "same_layout"),
    ("div", "arbitrary_batch", "nchw_tensor"),
]


def _metadata(value):
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    def concrete(item):
        if isinstance(item, torch.SymInt):
            assert not free_unbacked_symbols(item)
            assert type(item.node.hint) is int
            return item.node.hint
        return int(item)

    return (tuple(concrete(item) for item in value.shape),
            tuple(concrete(item) for item in value.stride()), value.dtype,
            concrete(value.storage_offset()))


def _meta_like(value):
    extent = value.storage_offset() + 1 + sum((size - 1) * stride for size, stride in
                                            zip(value.shape, value.stride(), strict=True))
    return torch.empty(extent, dtype=value.dtype, device="meta").as_strided(
        value.shape, value.stride(), value.storage_offset())


def _operands(layout, other, dtype):
    shape, strides, offset = _LAYOUTS[layout]
    extent = offset + 1 + sum((size - 1) * stride for size, stride in
                            zip(shape, strides, strict=True))
    backing = torch.arange(extent, device="cuda", dtype=torch.float32)
    backing = backing.remainder_(17).mul_(.03125).add_(.5).to(dtype)
    source = backing.as_strided(shape, strides, offset)
    if other == "one":
        right = 1.0
    elif other == "fraction":
        right = 2.5
    elif other in ("cpu_scalar", "cuda_scalar"):
        right = torch.tensor(2.5, dtype=dtype, device=other.split("_")[0])
    elif other == "same_layout":
        right = torch.empty_strided(shape, strides, dtype=dtype, device="cuda").fill_(2.5)
    else:
        right_shape = (1, shape[1], 1, 1) if other == "broadcast" else shape
        right = torch.full(right_shape, 2.5, dtype=dtype, device="cuda")
    return source, right


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("operation,layout,other", _CASES,
                         ids=["-".join(case) for case in _CASES])
@torch.no_grad()
def test_binary_metadata_matches_eager_cuda(dtype, operation, layout, other):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    runtime_patches.apply()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("assigned GPU does not support BF16")
    source, right = _operands(layout, other, dtype)
    function = getattr(torch, operation)
    expected = _metadata(function(source, right))
    meta_right = right
    if isinstance(right, torch.Tensor) and right.device.type != "cpu":
        meta_right = _meta_like(right)
    assert _metadata(function(_meta_like(source), meta_right)) == expected

    mode = FakeTensorMode()
    fake_source = mode.from_tensor(source)
    fake_right = mode.from_tensor(right) if isinstance(right, torch.Tensor) else right
    with mode:
        assert _metadata(function(fake_source, fake_right)) == expected

    if isinstance(right, torch.Tensor):
        traced = make_fx(function, tracing_mode="symbolic")(source, right)
    else:
        traced = make_fx(lambda value: function(value, right), tracing_mode="symbolic")(source)
    result = next(node for node in traced.graph.nodes if node.op == "output").args[0]
    assert isinstance(result, torch.fx.Node)
    assert _metadata(result.meta["val"]) == expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("operation,layout", [
    ("pow3", "singleton_batch"),
    ("pow4", "arbitrary_batch"),
    ("prims_pow2", "singleton_batch"),
    ("prims_pow4", "arbitrary_batch"),
    ("greater", "channels_last"),
    ("remainder", "nchw"),
])
@torch.no_grad()
def test_unrelated_scalar_operators_match_eager_cuda(dtype, operation, layout):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    runtime_patches.apply()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("assigned GPU does not support BF16")
    source, _ = _operands(layout, "one", dtype)
    if operation.startswith("prims_pow"):
        function = lambda value: torch.ops.prims.pow.default(value, int(operation[-1]))
    elif operation.startswith("pow"):
        function = lambda value: torch.pow(value, int(operation[-1]))
    elif operation == "greater":
        function = lambda value: torch.gt(value, 2.5)
    else:
        function = lambda value: torch.remainder(value, 2.5)
    expected = _metadata(function(source))
    assert _metadata(function(_meta_like(source))) == expected
    mode = FakeTensorMode()
    fake = mode.from_tensor(source)
    with mode:
        assert _metadata(function(fake)) == expected
    traced = make_fx(function, tracing_mode="symbolic")(source)
    result = next(node for node in traced.graph.nodes if node.op == "output").args[0]
    assert isinstance(result, torch.fx.Node)
    assert _metadata(result.meta["val"]) == expected


_VIEW_CASES = [
    ((1, 8, 35), (11, 1, 8), 9, (1, 8, 5, 7)),
    ((1, 8, 35), (11, 1, 8), 9, (1, 8, 35)),
    ((2, 3), (6, 2), 4, (6,)),
    ((2, 3), (1, 2), 5, (6,)),
    ((2, 3), (0, 1), 3, (6,)),
    ((1, 8, 35), (280, 1, 8), 9, (1, 8, -1, 7)),
]


@pytest.mark.parametrize("operation", ["view", "reshape"])
@pytest.mark.parametrize("shape,strides,offset,target", _VIEW_CASES,
                         ids=["unused-stride", "same-shape", "gapped-view", "transposed-copy",
                              "expanded-copy", "inferred-dimension"])
@torch.no_grad()
def test_view_metadata_preserves_values_aliases_and_offsets(shape, strides, offset, target, operation):
    import torch._refs as refs
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    runtime_patches.apply()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    extent = offset + 1 + sum((size - 1) * stride for size, stride in zip(shape, strides, strict=True))
    backing = torch.arange(extent, dtype=torch.float32, device="cuda")
    source = backing.as_strided(shape, strides, offset)
    saved = backing.clone()
    function = lambda value: getattr(value, operation)(target)
    reference = lambda value: getattr(refs, operation)(value, target)
    mode = FakeTensorMode()
    fake = mode.from_tensor(source)
    try:
        expected = function(source)
    except RuntimeError:
        with pytest.raises((RuntimeError, ValueError)):
            reference(source)
        with pytest.raises((RuntimeError, ValueError)):
            function(_meta_like(source))
        with mode, pytest.raises((RuntimeError, ValueError)):
            function(fake)
        with pytest.raises((RuntimeError, ValueError)):
            make_fx(function, tracing_mode="symbolic")(source)
    else:
        expected_alias = expected.untyped_storage()._cdata == source.untyped_storage()._cdata
        actual = reference(source)
        assert _metadata(actual) == _metadata(expected)
        assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
        assert (actual.untyped_storage()._cdata == source.untyped_storage()._cdata) == expected_alias
        meta = _meta_like(source)
        meta_result = function(meta)
        assert _metadata(meta_result) == _metadata(expected)
        assert (meta_result.untyped_storage()._cdata == meta.untyped_storage()._cdata) == expected_alias
        with mode:
            fake_result = function(fake)
        assert _metadata(fake_result) == _metadata(expected)
        assert (fake_result.untyped_storage()._cdata == fake.untyped_storage()._cdata) == expected_alias
        traced = make_fx(function, tracing_mode="symbolic")(source)
        output = next(node for node in traced.graph.nodes if node.op == "output").args[0].meta["val"]
        input_value = next(node for node in traced.graph.nodes if node.op == "placeholder").meta["val"]
        assert _metadata(output) == _metadata(expected)
        assert (output.untyped_storage()._cdata == input_value.untyped_storage()._cdata) == expected_alias
    assert torch.equal(backing.view(torch.uint8), saved.view(torch.uint8))
