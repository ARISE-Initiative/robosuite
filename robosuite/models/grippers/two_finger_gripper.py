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

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/two_finger_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 2

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def _joints(self):
        return ["r_finger_joint", "l_finger_joint"]

    @property
    def _actuators(self):
        return ["gripper_r_finger_joint", "gripper_l_finger_joint"]

    @property
    def _contact_geoms(self):
        return [
            "r_finger_g0",
            "r_finger_g1",
            "l_finger_g0",
            "l_finger_g1",
            "r_fingertip_g0",
            "l_fingertip_g0",
        ]

    @property
    def _left_finger_geoms(self):
        return ["l_finger_g0", "l_finger_g1", "l_fingertip_g0"]

    @property
    def _right_finger_geoms(self):
        return ["r_finger_g0", "r_finger_g1", "r_fingertip_g0"]


class TwoFingerGripper(TwoFingerGripperBase):
    """
    Modifies two finger base to only take one action.
    """

    def format_action(self, action):
        """
        Args:
            action: 1 => open, -1 => closed
        """
        assert len(action) == 1
        return np.array([-1 * action[0], 1 * action[0]])

    @property
    def dof(self):
        return 1
