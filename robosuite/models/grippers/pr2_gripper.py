"""
    4 dof gripper with two fingers and its open/close variant
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper


class PR2GripperBase(Gripper):
    """
    A 4 dof gripper with two fingers.
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/pr2_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 4

    @property
    def init_qpos(self):
        return np.zeros(4)

    @property
    def _joints(self):
        return [
            "r_finger_joint",
            "l_finger_joint",
            "r_finger_tip_joint",
            "l_finger_tip_joint",
        ]

    @property
    def _actuators(self):
        return [
            "gripper_r_finger_joint",
            "gripper_l_finger_joint",
            "gripper_r_finger_tip_joint",
            "gripper_l_finger_tip_joint"
        ]

    @property
    def _contact_geoms(self):
        return [
            "l_finger",
            "l_finger_tip",
            "r_finger",
            "r_finger_tip",
        ]

    @property
    def _left_finger_geoms(self):
        return ["l_finger", "l_finger_tip"]

    @property
    def right_finger_geoms(self):
        return ["r_finger", "r_finger_tip"]


class PR2Gripper(PR2GripperBase):
    """
    Open/close variant of PR2 gripper.
    """

    def format_action(self, action):
        """
        Args:
            action: -1 => open, 1 => closed
        """
        assert len(action) == 1
        return -np.ones(4) * action

    @property
    def dof(self):
        return 1
