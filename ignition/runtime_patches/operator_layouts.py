# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse computed opaque operands before enforcing their exact layouts."""

from . import _replace


def replacements():
    from torch._inductor import lowering

    constrain = _replace(lowering.constrain_to_fake_tensor, ((
        "    if isinstance(arg, ir.IRNode):\n"
        "        return ir.ExternKernel.require_exact_strides(arg, fake_arg.stride())\n",
        "    if isinstance(arg, ir.IRNode):\n"
        "        if (isinstance(arg, ir.TensorBox) and isinstance(arg.data, ir.StorageBox)\n"
        "                and isinstance(arg.data.data, ir.Pointwise)):\n"
        "            node = V.graph.current_node\n"
        "            if (node is not None and isinstance(node.target, torch._ops.OpOverload)\n"
        "                    and node.target.name() == 'flare::storage_call'):\n"
        "                arg.realize()\n"
        "        return ir.ExternKernel.require_exact_strides(arg, fake_arg.stride())\n",
    ),), allow_applied=True)
    return [(lowering, 'constrain_to_fake_tensor', constrain)]
