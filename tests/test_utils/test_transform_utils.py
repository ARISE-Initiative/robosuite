"""Check wrench frame changes against point forces and invariant mechanical power."""

import numpy as np
import pytest

import robosuite.utils.transform_utils as T


@pytest.fixture(params=["identity", "rotation", "translation", "rotation_and_translation"])
def pose_A_in_B(request):
    """Use exact quarter turns to distinguish the forward and inverse transforms."""
    rotation = np.eye(3)
    translation = np.zeros(3)
    if "rotation" in request.param:
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if "translation" in request.param:
        translation = np.array([1.0, -0.5, 0.25])
    return T.make_pose(translation, rotation)


def test_force_transform_matches_force_applied_at_transformed_point(pose_A_in_B):
    point_A = np.array([0.3, -0.4, 0.8])
    force_A = np.array([1.5, -2.0, 0.75])
    torque_A = np.cross(point_A, force_A)

    # Transform the force's application point and arrow tip with the public pose API.
    point_B = T.pose_in_A_to_pose_in_B(T.make_pose(point_A, np.eye(3)), pose_A_in_B)[:3, 3]
    tip_B = T.pose_in_A_to_pose_in_B(T.make_pose(point_A + force_A, np.eye(3)), pose_A_in_B)[:3, 3]
    expected_force_B = tip_B - point_B

    force_B, torque_B = T.force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B)

    np.testing.assert_allclose(force_B, expected_force_B, rtol=0, atol=1e-12)
    np.testing.assert_allclose(torque_B, np.cross(point_B, expected_force_B), rtol=0, atol=1e-12)


def test_force_and_velocity_transform_preserve_mechanical_power(pose_A_in_B):
    force_A = np.array([1.5, -2.0, 0.75])
    torque_A = np.array([0.25, 0.8, -0.6])
    vel_A = np.array([0.4, -0.1, 0.3])
    ang_vel_A = np.array([-0.2, 0.5, 0.7])

    force_B, torque_B = T.force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B)
    vel_B, ang_vel_B = T.vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B)

    power_A = np.dot(force_A, vel_A) + np.dot(torque_A, ang_vel_A)
    power_B = np.dot(force_B, vel_B) + np.dot(torque_B, ang_vel_B)
    np.testing.assert_allclose(power_B, power_A, rtol=0, atol=1e-12)
