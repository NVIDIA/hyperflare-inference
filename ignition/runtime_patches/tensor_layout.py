# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve scalar operands and native backed view geometry in metadata."""

from functools import partial
import inspect

from ignition.errors import PreparationError

from . import _replace, identity


def _binary_metadata(permutation, metadata):
    """Clone metadata with a private stride helper for binary operators."""
    corrected_permutation = _replace(permutation, [
        ("    # Filters the tensors to actual tensors\n",
         "    has_scalar_operand = any(\n"
         "        not isinstance(a, TensorLike) or a.ndim == 0 for a in tensors\n"
         "    )\n\n"
         "    # Filters the tensors to actual tensors\n"),
        ("    if is_contiguous and not is_channels_last:\n",
         "    if is_contiguous and not is_channels_last and not has_scalar_operand:\n"),
        ("    if is_channels_last and not is_contiguous:\n",
         "    if is_channels_last and not is_contiguous and not has_scalar_operand:\n"),
    ])
    cloned = _replace(metadata, [
        ("    *args,\n    type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND,\n",
         "    *args,\n    _stride_permutation,\n    type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND,\n"),
        ("    l2p_perm, _ = utils.compute_elementwise_output_logical_to_physical_perm(*args_)\n",
         "    l2p_perm, _ = _stride_permutation(*args_)\n"),
    ])
    return corrected_permutation, cloned


def _view_metadata(core):
    """Use native view geometry while retaining unbacked and copy fallbacks."""
    return _replace(core, [
        ("    # NOTE [Reshape Algorithm]\n",
         "    import builtins\n"
         "    from torch._subclasses.fake_impls import _compute_stride\n"
         "    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols\n"
         "    metadata = (*a.shape, *a.stride(), *shape)\n"
         "    if not builtins.any(free_unbacked_symbols(value) for value in metadata):\n"
         "        native_strides = _compute_stride(a.shape, a.stride(), shape)\n"
         "        if native_strides is not None:\n"
         "            return as_strided(a, shape, native_strides, a.storage_offset())\n\n"
         "    # NOTE [Reshape Algorithm]\n"),
    ])


def replacements():
    import torch
    import torch._prims as prims
    import torch._refs as refs
    import torch._prims_common as common
    from torch._library.custom_ops import CustomOpDef, OPDEFS
    from torch._subclasses.fake_tensor import FakeTensorMode

    wrapper_source = inspect.getsource(CustomOpDef._register_fake_dispatcher_impl)
    if wrapper_source.count("return self._abstract_fn(*args, **kwargs)") != 1:
        raise PreparationError("primitive fake dispatcher no longer reads its metadata attribute")
    primitive_source = inspect.getsource(prims._make_prim)
    if (primitive_source.count("prim_def.register_fake(meta)") != 1
            or primitive_source.count("p.prim_meta_impl = meta") != 1):
        raise PreparationError("functional primitive metadata registration changed")

    metadata = prims._prim_elementwise_meta
    permutation, binary_metadata = _binary_metadata(
        common.compute_elementwise_output_logical_to_physical_perm, metadata)
    result = [(refs, "_reshape_view_helper_core_alg",
               _view_metadata(refs._reshape_view_helper_core_alg))]
    for name in ("add", "sub", "mul", "div"):
        packet = getattr(torch.ops.prims, name)
        overload = packet.default
        opdef = OPDEFS.get("prims::" + name)
        original = packet.prim_meta_impl
        if (opdef is None or opdef._opoverload is not overload
                or overload.prim_meta_impl is not original or opdef._abstract_fn is not original):
            raise PreparationError(f"{name} primitive metadata registrations do not agree")
        if (not isinstance(original, partial) or original.func is not metadata or original.args
                or set(original.keywords) != {"type_promotion"}):
            raise PreparationError(f"{name} primitive does not use the expected elementwise metadata")
        replacement = partial(binary_metadata, _stride_permutation=permutation, **original.keywords)
        # Meta and FakeTensor dispatch resolve these attributes independently.
        result.extend(((packet, "prim_meta_impl", replacement),
                       (overload, "prim_meta_impl", replacement),
                       (opdef, "_abstract_fn", replacement)))

    dispatch = _replace(FakeTensorMode._dispatch_impl, [
        ("    # Try for fastpath\n    if has_symbolic_sizes:\n",
         "    # Try for fastpath\n    if has_symbolic_sizes:\n"
         "        skip_binary_fastpath = False\n"
         "        if func in (torch.ops.aten.add.Tensor, torch.ops.aten.sub.Tensor,\n"
         "                    torch.ops.aten.mul.Tensor, torch.ops.aten.div.Tensor):\n"
         "            from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_eq\n"
         "            left = args[0] if args else kwargs.get(\"self\")\n"
         "            right = args[1] if len(args) > 1 else kwargs.get(\"other\")\n"
         "            skip_binary_fastpath = (\n"
         "                not isinstance(left, torch.Tensor)\n"
         "                or not isinstance(right, torch.Tensor)\n"
         "                or left.ndim != right.ndim\n"
         "                or not guard_or_false(sym_eq(left.shape, right.shape))\n"
         "            )\n"),
        ("        if fast_impl is not None:\n",
         "        if fast_impl is not None and not skip_binary_fastpath:\n"),
    ])
    # One package identity separates both binary and view metadata from existing cache entries.
    cache_key = _replace(FakeTensorMode._cache_key, [
        ("    key_values = [\n        func,\n",
         "    key_values = [\n        func,\n        " + repr(identity()) + ",\n"),
    ])
    result.extend(((FakeTensorMode, "_dispatch_impl", dispatch),
                   (FakeTensorMode, "_cache_key", cache_key)))
    return result
