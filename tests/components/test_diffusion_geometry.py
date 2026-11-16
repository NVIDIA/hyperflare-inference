# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact compression and patch capacities for image and causal video latents."""

from math import prod

import pytest

from ignition.components.diffusion_geometry import LatentGeometry


@pytest.mark.parametrize("geometry,output,latent,grid,packed", [
    (LatentGeometry(4, (1, 8, 8), (1, 8, 8)),
     (1, 1024, 1024), (4, 1, 128, 128), (1, 128, 128), (16384, 4)),
    (LatentGeometry(16, (1, 8, 8), (1, 8, 8), (1, 2, 2)),
     (1, 768, 1024), (16, 1, 96, 128), (1, 48, 64), (3072, 64)),
    (LatentGeometry(16, (4, 8, 8), (1, 8, 8), (1, 2, 2)),
     (81, 480, 832), (16, 21, 60, 104), (21, 30, 52), (32760, 64)),
    (LatentGeometry(48, (4, 16, 16), (1, 16, 16), (1, 2, 2)),
     (121, 704, 1280), (48, 31, 44, 80), (31, 22, 40), (27280, 192)),
    (LatentGeometry(16, (4, 8, 8), (1, 8, 8), (1, 2, 2)),
     (129, 720, 1280), (16, 33, 90, 160), (33, 45, 80), (118800, 64)),
])
def test_image_and_video_latent_geometry(geometry, output, latent, grid, packed):
    assert geometry.latent_shape(output) == latent
    assert geometry.output_shape(latent) == output
    assert geometry.patch_grid(output) == grid
    assert geometry.packed_shape(output) == packed
    assert prod(latent) == prod(packed)


def test_causal_compression_preserves_the_first_frame():
    geometry = LatentGeometry(16, (4, 8, 8), (1, 8, 8))
    assert [geometry.latent_shape((frames, 8, 8))[1] for frames in (1, 5, 9)] == [1, 2, 3]
    for frames in (2, 4, 6, 8):
        with pytest.raises(ValueError, match="output frames"):
            geometry.latent_shape((frames, 8, 8))


def test_dense_temporal_compression_uses_complete_groups():
    geometry = LatentGeometry(8, (4, 8, 8), (4, 8, 8))
    assert geometry.latent_shape((8, 16, 24)) == (8, 2, 2, 3)
    assert geometry.output_shape((8, 2, 2, 3)) == (8, 16, 24)
    with pytest.raises(ValueError, match="output frames"):
        geometry.latent_shape((9, 16, 24))


def test_patch_geometry_retains_temporal_patch_extent():
    geometry = LatentGeometry(16, (4, 8, 8), (1, 8, 8), (2, 2, 2))
    assert geometry.packed_shape((13, 32, 48)) == (12, 128)
    with pytest.raises(ValueError, match="divisible"):
        geometry.packed_shape((9, 32, 48))


def test_compression_and_patch_alignment_are_separate_requirements():
    geometry = LatentGeometry(16, (1, 8, 8), (1, 8, 8), (1, 2, 2))
    with pytest.raises(ValueError, match="output height"):
        geometry.latent_shape((1, 1023, 1024))
    assert geometry.latent_shape((1, 1016, 1024)) == (16, 1, 127, 128)
    with pytest.raises(ValueError, match="divisible"):
        geometry.patch_grid((1, 1016, 1024))


@pytest.mark.parametrize("value", [0, -1, True, 1.0, 1 << 64])
def test_invalid_dimensions_fail_before_tensor_allocation(value):
    with pytest.raises(ValueError, match="positive uint64"):
        LatentGeometry(value, (1, 8, 8), (1, 8, 8))
    for field in ("stride", "offset", "patch"):
        arguments = dict(channels=4, stride=(1, 8, 8), offset=(1, 8, 8))
        arguments[field] = (1, value, 1)
        with pytest.raises(ValueError, match="positive uint64"):
            LatentGeometry(**arguments)
    geometry = LatentGeometry(4, (1, 8, 8), (1, 8, 8))
    with pytest.raises(ValueError, match="positive uint64"):
        geometry.latent_shape((1, value, 8))
    with pytest.raises(ValueError, match="positive uint64"):
        geometry.output_shape((4, 1, value, 1))


@pytest.mark.parametrize("shape", [None, 1, (), (1, 8), (1, 8, 8, 1)])
def test_output_shape_requires_three_axes(shape):
    geometry = LatentGeometry(4, (1, 8, 8), (1, 8, 8))
    with pytest.raises(ValueError, match="3 dimensions"):
        geometry.latent_shape(shape)


def test_geometry_requires_consistent_compression_and_channels():
    with pytest.raises(ValueError, match="offsets"):
        LatentGeometry(4, (1, 8, 8), (1, 9, 8))
    geometry = LatentGeometry(4, (1, 8, 8), (1, 8, 8))
    with pytest.raises(ValueError, match="channel"):
        geometry.output_shape((16, 1, 128, 128))
    with pytest.raises(ValueError, match="4 dimensions"):
        geometry.output_shape((1, 128, 128))


def test_uint64_limits_cover_derived_extents_and_element_counts():
    maximum = (1 << 64) - 1
    identity = LatentGeometry(1, (1, 1, 1), (1, 1, 1))
    assert identity.packed_shape((1, 1, maximum)) == (maximum, 1)
    with pytest.raises(ValueError, match="output element count"):
        identity.latent_shape((1, 2, maximum))
    with pytest.raises(ValueError, match="latent element count"):
        LatentGeometry(2, (1, 1, 1), (1, 1, 1)).latent_shape((1, 1, maximum))
    with pytest.raises(ValueError, match="output width"):
        LatentGeometry(1, (1, 1, 2), (1, 1, 2)).output_shape((1, 1, 1, maximum))
    with pytest.raises(ValueError, match="packed patch element count"):
        LatentGeometry(maximum, (1, 1, 1), (1, 1, 1), (1, 1, 2))


def test_geometry_copies_mutable_dimension_arguments():
    stride, offset, patch = [1, 8, 8], [1, 8, 8], [1, 2, 2]
    geometry = LatentGeometry(16, stride, offset, patch)
    stride[1], offset[1], patch[1] = 16, 16, 4
    assert geometry.packed_shape((1, 1024, 1024)) == (4096, 64)
