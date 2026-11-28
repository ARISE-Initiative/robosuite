"""Tests for sampling camera depth during pixel-to-world transforms."""

import numpy as np

from robosuite.utils.camera_utils import transform_from_pixels_to_world


def test_pixel_to_world_samples_each_image_independently():
    pixels = np.array([[[0.5, 1.5], [1.0, 2.0]], [[0.0, 0.0], [0.0, 2.0]]])
    offsets = np.array([[1.0, 10.0], [20.0, 30.0]])
    grid = np.array([[0.0, 1.0, 2.0], [4.0, 5.0, 6.0]])
    depth_maps = (offsets[..., None, None] + grid)[..., None]

    points = transform_from_pixels_to_world(pixels, depth_maps, np.eye(4))

    np.testing.assert_allclose(
        points,
        [[[6.75, 2.25, 4.5], [32.0, 16.0, 16.0]], [[0.0, 0.0, 20.0], [64.0, 0.0, 32.0]]],
    )


def test_pixel_to_world_samples_last_pixel_in_single_image():
    depth_map = np.arange(1.0, 7.0).reshape(2, 3, 1)
    point = transform_from_pixels_to_world(np.array([1.0, 2.0]), depth_map, np.eye(4))
    np.testing.assert_allclose(point, [12.0, 6.0, 6.0])
