# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact latent and patch geometry for preparation of image and video models."""

from dataclasses import dataclass
from math import prod


_UINT64_MAX = (1 << 64) - 1
_AXES = ("frames", "height", "width")


def _positive(value, name):
    if type(value) is not int or not 1 <= value <= _UINT64_MAX:
        raise ValueError(f"{name} must be a positive uint64 integer")
    return value


def _shape(value, name, rank):
    try:
        dimensions = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain {rank} dimensions") from error
    if len(dimensions) != rank:
        raise ValueError(f"{name} must contain {rank} dimensions")
    for dimension in dimensions:
        _positive(dimension, f"{name} dimension")
    return dimensions


def _element_count(shape, name):
    return _positive(prod(shape), f"{name} element count")


@dataclass(frozen=True)
class LatentGeometry:
    """Map output extents to channel-first latents and nonoverlapping patches.

    ``stride``, ``offset``, and ``patch`` have frames, height, width order. Each
    output extent equals ``(latent_extent - 1) * stride + offset``. Dense spatial
    compression uses ``offset == stride``; causal temporal compression uses a
    frame offset of one. Images have one frame with frame stride and offset one.

    Shapes describe one sample without a batch dimension. Patch metadata records
    extents and element counts, not a tensor permutation or within-patch order.
    Every mapping is exact: no padding, cropping, or rounding is performed.
    """

    channels: int
    stride: tuple[int, int, int]
    offset: tuple[int, int, int]
    patch: tuple[int, int, int] = (1, 1, 1)

    def __post_init__(self):
        _positive(self.channels, "latent channels")
        for name in ("stride", "offset", "patch"):
            object.__setattr__(self, name, _shape(getattr(self, name), name, 3))
        if any(offset > stride for offset, stride in zip(self.offset, self.stride)):
            raise ValueError("compression offsets must not exceed their strides")
        _element_count((self.channels, *self.patch), "packed patch")

    def latent_shape(self, output_shape):
        """Return channels, frames, height, width for an exact output extent."""
        output = _shape(output_shape, "output shape", 3)
        _element_count(output, "output")
        latent = []
        for axis, extent, stride, offset in zip(_AXES, output, self.stride, self.offset):
            if extent < offset or (extent - offset) % stride:
                raise ValueError(f"output {axis} must equal (latent extent - 1) * {stride} + {offset}")
            latent.append((extent - offset) // stride + 1)
        shape = (self.channels, *latent)
        _element_count(shape, "latent")
        return shape

    def output_shape(self, latent_shape):
        """Return frames, height, width from a channel-first latent shape."""
        latent = _shape(latent_shape, "latent shape", 4)
        if latent[0] != self.channels:
            raise ValueError("latent channel dimension must match the declared channels")
        _element_count(latent, "latent")
        output = tuple(_positive((extent - 1) * stride + offset, f"output {axis}")
            for axis, extent, stride, offset in zip(_AXES, latent[1:], self.stride, self.offset))
        _element_count(output, "output")
        return output

    def patch_grid(self, output_shape):
        """Return an exact grid of patches in latent frames, height, width."""
        latent = self.latent_shape(output_shape)[1:]
        if any(extent % patch for extent, patch in zip(latent, self.patch)):
            raise ValueError("latent dimensions must be divisible by the declared patch dimensions")
        return tuple(extent // patch for extent, patch in zip(latent, self.patch))

    def packed_shape(self, output_shape):
        """Return token count and elements per token for nonoverlapping patches."""
        return prod(self.patch_grid(output_shape)), self.channels * prod(self.patch)
