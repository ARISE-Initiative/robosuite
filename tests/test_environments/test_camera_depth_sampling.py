"""Check depth sampling and world-coordinate reconstruction at image boundaries."""

import numpy as np
import pytest

from robosuite.utils.camera_utils import bilinear_interpolate, transform_from_pixels_to_world


@pytest.mark.parametrize("pixel", [(0, 0), (0, 4), (2, 0), (2, 4), (2, 2), (1, 4), (0.5, 1.5)])
def test_bilinear_depth_sampling(pixel):
    """Sampling an affine depth image should reproduce its value, including at edges."""
    rows, columns = np.indices((3, 5))
    depth = 1.0 + rows + 2.0 * columns
    row, column = pixel

    actual = bilinear_interpolate(depth, x=column, y=row)

    np.testing.assert_allclose(actual, 1.0 + row + 2.0 * column)


@pytest.mark.parametrize("pixel", [(0, 0), (0, 4), (2, 0), (2, 4), (2, 2), (1, 4), (0.5, 1.5)])
def test_depth_unprojection_at_image_boundaries(pixel):
    """Valid edge pixels must retain their depth when transformed into world coordinates."""
    rows, columns = np.indices((3, 5))
    depth_map = (1.0 + rows + 2.0 * columns)[..., None]
    camera_to_world = np.array(
        [[0.5, 0.0, -1.0, 2.0], [0.0, 0.25, -0.5, -1.0], [0.0, 0.0, 1.0, 3.0], [0.0, 0.0, 0.0, 1.0]]
    )
    row, column = pixel
    depth = 1.0 + row + 2.0 * column
    expected = [(0.5 * column - 1.0) * depth + 2.0, (0.25 * row - 0.5) * depth - 1.0, depth + 3.0]

    actual = transform_from_pixels_to_world(np.array(pixel), depth_map, camera_to_world)

    np.testing.assert_allclose(actual, expected)
