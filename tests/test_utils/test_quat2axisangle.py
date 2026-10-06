"""Check quaternion logarithms without losing vector information near the identity."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import robosuite.utils.transform_utils as T


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("angle", [0.0, 1e-8, 1e-4, 0.3])
def test_quat2axisangle_agrees_with_scipy_near_identity(dtype, angle):
    rotvec = np.array([1.0, -2.0, 3.0]) * (angle / np.sqrt(14.0))
    quat = Rotation.from_rotvec(rotvec).as_quat().astype(dtype)

    expected = Rotation.from_quat(quat).as_rotvec()
    actual = T.quat2axisangle(quat.copy())

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 1e-7, 1e-3, 0.5])
def test_mat2quat_to_axisangle_preserves_the_native_quaternion(angle):
    rotvec = np.array([1.0, -2.0, 3.0]) * (angle / np.sqrt(14.0))
    quat = T.mat2quat(Rotation.from_rotvec(rotvec).as_matrix())
    assert quat.dtype == np.float32

    # Match the actual controller/device conversion chain. The oracle uses the
    # quaternion returned by mat2quat, so its separate float32 error is excluded.
    expected = Rotation.from_quat(quat).as_rotvec()
    actual = T.quat2axisangle(quat.copy())

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-12)


def test_quat2axisangle_preserves_long_angle_convention():
    rotvec = np.array([0.0, 0.0, 4.2])
    quat = T.axisangle2quat(rotvec)
    assert quat[3] < 0

    np.testing.assert_allclose(T.quat2axisangle(quat), rotvec, rtol=0, atol=1e-12)
