# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Lossless source planes for MXFP4 MegaMoE preparation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Mxfp4Planes:
    """Packed gate/up and down weights with row-major native UE8M0 scales.

    These are preparation intermediates. MegaMoE's weight transform consumes
    gate rows followed by up rows; its final interleaving is a separate step.
    No activation quantization or expert execution occurs here.
    """

    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor


@dataclass(frozen=True)
class MegaMoeWeights:
    """MegaMoE's eight-row gate/up interleave and packed UTCCP scale layout."""

    gate_up: tuple[torch.Tensor, torch.Tensor]
    down: tuple[torch.Tensor, torch.Tensor]


def pack_mxfp4_planes(planes):
    """Apply DeepGEMM's reference MXFP4 transform without requantizing weights.

    Matches vLLM's SM100 MegaMoE preparation: reinterpret native exponent bytes
    as FP32 powers of two, pack the native scale layout, then interleave gate
    and up rows. Call only during preparation; the returned planes own their
    final storage through graph capture and export.
    """
    import deep_gemm
    from ignition.preparation.persistent import persistent_copy

    count, doubled, packed_hidden = planes.w13.shape
    hidden, intermediate = packed_hidden * 2, doubled // 2
    expected = ((count, doubled, hidden // 32), (count, hidden, intermediate // 2),
                (count, hidden, intermediate // 32))
    if (doubled % 2 or intermediate % 128 or hidden % 128
            or any(t.shape != shape for t, shape in zip((planes.s13, planes.w2, planes.s2), expected))
            or any(t.dtype != torch.uint8 or t.device != planes.w13.device or not t.is_contiguous()
                   for t in (planes.w13, planes.s13, planes.w2, planes.s2))):
        raise ValueError("MegaMoE packing requires matched contiguous MXFP4 planes and 128-aligned widths")

    def scales(value, rows, columns):
        fp32 = (value.to(torch.int32) << 23).view(torch.float32)
        return deep_gemm.transform_sf_into_required_layout(fp32, rows, columns, (1, 32), count)

    gate_up, down = deep_gemm.transform_weights_for_mega_moe(
        (planes.w13.view(torch.int8), scales(planes.s13, 2 * intermediate, hidden)),
        (planes.w2.view(torch.int8), scales(planes.s2, hidden, intermediate)),
    )
    return MegaMoeWeights(tuple(persistent_copy(t) for t in gate_up),
                          tuple(persistent_copy(t) for t in down))


def load_mxfp4_planes(loader, prefixes, *, hidden_size, intermediate_size,
                     intermediate_range=None, projection_names=("w1", "w3", "w2"),
                     scale_name="scale"):
    """Expand checkpoint CSF once into exact native scale bytes for preparation.

    ``projection_names`` is gate, up, down. Packed FP4 bytes are only sliced
    and copied. Native and CSF source groups produce the same row-major planes.
    Temporary GPU storage is owned by the result and the caller must retain it
    until the consuming weight transform has completed.
    """
    from b12x._lib.quant._csf_prepare import expand_source_scales
    from ignition.components.weights import _aligned_range

    loader._require_open()
    prefixes = tuple(prefixes)
    if (not prefixes or len(set(prefixes)) != len(prefixes)
            or hidden_size <= 0 or hidden_size % 128):
        raise ValueError("MegaMoE source planes require unique experts and 128-aligned hidden width")
    if len(projection_names) != 3 or len(set(projection_names)) != 3:
        raise ValueError("MegaMoE source projections must name distinct gate, up, and down weights")
    first, last = _aligned_range(intermediate_range, intermediate_size, 32)
    local, count = last - first, len(prefixes)
    gate, up, down = projection_names
    names = [f"{p}.{projection}.{scale_name}" for p in prefixes for projection in projection_names]
    encodings = []
    for name in names:
        native = name in loader._sources
        choices = []
        for suffixes in (("mxfp4_csf_fixed", "mxfp4_csf_exceptions"),
                         ("csf32_bases", "csf32_codes", "csf32_page_bases", "csf32_offsets", "csf32_exceptions")):
            present = [f"{name}.{suffix}" in loader._sources for suffix in suffixes]
            if any(present):
                if not all(present):
                    raise ValueError(f"incomplete MXFP4 scale encoding: {name}")
                choices.append(suffixes[0])
        if len(choices) + int(native) != 1:
            raise ValueError(f"MXFP4 source needs exactly one scale representation: {name}")
        encodings.append(choices[0] if choices else "native")
    if len(set(encodings)) != 1:
        raise ValueError("MegaMoE source planes require one scale encoding per expert group")

    def empty(shape):
        return torch.empty(shape, dtype=torch.uint8, device=loader.device)

    planes = Mxfp4Planes(empty((count, 2 * local, hidden_size // 2)),
                         empty((count, 2 * local, hidden_size // 32)),
                         empty((count, hidden_size, local // 2)),
                         empty((count, hidden_size, local // 32)))
    for index, prefix in enumerate(prefixes):
        for projection, target, scale_target, rows, columns in (
            (gate, planes.w13[index, :local], planes.s13[index, :local], intermediate_size, hidden_size),
            (up, planes.w13[index, local:], planes.s13[index, local:], intermediate_size, hidden_size),
            (down, planes.w2[index], planes.s2[index], hidden_size, intermediate_size),
        ):
            name = f"{prefix}.{projection}"
            source = loader._sources[f"{name}.weight"]
            if source.dtype not in (torch.uint8, torch.int8) or source.shape != (rows, columns // 2):
                raise ValueError(f"invalid packed MXFP4 source geometry: {name}")
            source = source.view(torch.uint8)
            loader._copy(target, source[:, first // 2:last // 2] if projection == down else source[first:last])
            if encodings[0] == "native":
                scales = loader._sources[f"{name}.{scale_name}"]
                if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu) or scales.shape != (rows, columns // 32):
                    raise ValueError(f"invalid native MXFP4 scale geometry: {name}")
                scales = scales.view(torch.uint8)
                loader._copy(scale_target, scales[:, first // 32:last // 32]
                             if projection == down else scales[first:last])
    sources = ()
    if encodings[0] != "native":
        sources = (
            loader._queue_csf_sources(
                [f"{p}.{projection}.{scale_name}" for p in prefixes for projection in (gate, up)],
                rows=intermediate_size, columns=hidden_size // 32, quantization="mxfp4",
                temporary=True, join=2, row_range=(first, last)),
            loader._queue_csf_sources(
                [f"{p}.{down}.{scale_name}" for p in prefixes], rows=hidden_size,
                columns=intermediate_size // 32, quantization="mxfp4", temporary=True,
                column_range=(first // 32, last // 32)),
        )
    loader._session.flush()
    for source, target in zip(sources, (planes.s13, planes.s2)):
        expand_source_scales(source, target, swizzled=False)
    return planes
