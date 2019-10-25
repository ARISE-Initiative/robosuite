"""
Gripper with two fingers.
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper


class TwoFingerGripperBase(Gripper):
    """
    Gripper with two fingers.
    """

    def __init__(self):
        super().__init__(xml_path_completion("grippers/two_finger_gripper.xml"))

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def joints(self):
        return ["r_gripper_l_finger_joint", "r_gripper_r_finger_joint"]

    @property
    def sensors(self):
        return ["force_ee", "torque_ee"]

    @property
    def dof(self):
        return 2

    @property
    def visualization_sites(self):
        return ["grip_site", "grip_site_cylinder"]

    def contact_geoms(self):
        return [
            "r_finger_g0",
            "r_finger_g1",
            "l_finger_g0",
            "l_finger_g1",
            "r_fingertip_g0",
            "l_fingertip_g0",
        ]

    @property
    def left_finger_geoms(self):
        return ["l_finger_g0", "l_finger_g1", "l_fingertip_g0"]

    @property
    def right_finger_geoms(self):
        return ["r_finger_g0", "r_finger_g1", "r_fingertip_g0"]


class TwoFingerGripper(TwoFingerGripperBase):
    """
    Modifies two finger base to only take one action.
    """

    def format_action(self, action):
        """
        1 => open, -1 => closed
        """
        assert len(action) == 1
        return np.array([1 * action[0], -1 * action[0]])

    @property
    def dof(self):
        return 1


class LeftTwoFingerGripperBase(Gripper):
    """
    A copy of two finger gripper with non-overlapping names
    to allow two grippers on a same robot.
    """

    def __init__(self):
        super().__init__(xml_path_completion("grippers/left_two_finger_gripper.xml"))

    def format_action(self, action):
        return action
        # return np.array([-1 * action, 1 * action])

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def joints(self):
        return ["l_gripper_l_finger_joint", "l_gripper_r_finger_joint"]

    @property
    def dof(self):
        return 2

    @property
    def visualization_sites(self):
        return ["l_g_grip_site", "l_g_grip_site_cylinder"]

    def contact_geoms(self):
        return [
            "l_g_r_finger_g0",
            "l_g_r_finger_g1",
            "l_g_l_finger_g0",
            "l_g_l_finger_g1",
            "l_g_r_fingertip_g0",
            "l_g_l_fingertip_g0",
        ]

    @property
    def left_finger_geoms(self):
        return ["l_g_l_finger_g0", "l_g_l_finger_g1", "l_g_l_fingertip_g0"]

    @property
    def right_finger_geoms(self):
        return ["l_g_r_finger_g0", "l_g_r_finger_g1", "l_g_r_fingertip_g0"]


class LeftTwoFingerGripper(LeftTwoFingerGripperBase):
    """
    A copy of two finger gripper with non-overlapping names
    to allow two grippers on a same robot.
    """

    def format_action(self, action):
        """
        Args:
            action: 1 => open, -1 => closed
        """
        assert len(action) == 1
        return np.array([1 * action[0], -1 * action[0]])

    @property
    def dof(self):
        return 1
