"""Regression tests for the deployed GR00T / SONIC G1 default pose."""

import numpy as np
import pytest

from robosuite.models.grippers.sonic_dex3_gripper import SonicDex3LeftGripper, SonicDex3RightGripper
from robosuite.models.robots.manipulators.sonic_g1_robot import SonicG1

GR00T_DEPLOY_DEFAULTS = (
    ("l_leg_hip_pitch_joint", -0.312),
    ("l_leg_hip_roll_joint", 0.0),
    ("l_leg_hip_yaw_joint", 0.0),
    ("l_leg_knee_joint", 0.669),
    ("l_leg_ankle_pitch_joint", -0.363),
    ("l_leg_ankle_roll_joint", 0.0),
    ("r_leg_hip_pitch_joint", -0.312),
    ("r_leg_hip_roll_joint", 0.0),
    ("r_leg_hip_yaw_joint", 0.0),
    ("r_leg_knee_joint", 0.669),
    ("r_leg_ankle_pitch_joint", -0.363),
    ("r_leg_ankle_roll_joint", 0.0),
    ("torso_waist_yaw_joint", 0.0),
    ("torso_waist_roll_joint", 0.0),
    ("torso_waist_pitch_joint", 0.0),
    ("l_shoulder_pitch_joint", 0.2),
    ("l_shoulder_roll_joint", 0.2),
    ("l_shoulder_yaw_joint", 0.0),
    ("l_elbow_joint", 0.6),
    ("l_wrist_roll_joint", 0.0),
    ("l_wrist_pitch_joint", 0.0),
    ("l_wrist_yaw_joint", 0.0),
    ("r_shoulder_pitch_joint", 0.2),
    ("r_shoulder_roll_joint", -0.2),
    ("r_shoulder_yaw_joint", 0.0),
    ("r_elbow_joint", 0.6),
    ("r_wrist_roll_joint", 0.0),
    ("r_wrist_pitch_joint", 0.0),
    ("r_wrist_yaw_joint", 0.0),
)


def test_init_qpos_matches_groot_deploy_default():
    robot = SonicG1(idn=0)
    expected_names = [name for name, _ in GR00T_DEPLOY_DEFAULTS]
    expected_qpos = np.array([qpos for _, qpos in GR00T_DEPLOY_DEFAULTS])

    assert [name.removeprefix("robot0_") for name in robot.joints] == expected_names
    np.testing.assert_array_equal(robot.init_qpos, expected_qpos)


@pytest.mark.parametrize("gripper_cls", [SonicDex3LeftGripper, SonicDex3RightGripper])
def test_dex3_init_qpos_matches_groot_open_default(gripper_cls):
    gripper = gripper_cls(idn=0)

    np.testing.assert_array_equal(gripper.init_qpos, np.zeros(7))
