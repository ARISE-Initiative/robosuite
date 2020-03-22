"""
    4 dof gripper with two fingers and its open/close variant
"""
import numpy as np

from robosuite.models.grippers import Gripper
from robosuite.utils.mjcf_utils import xml_path_completion


class PR2GripperBase(Gripper):
    """
    A 4 dof gripper with two fingers.
    """

    def __init__(self):
        super().__init__(xml_path_completion("grippers/pr2_gripper.xml"))

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.zeros(4)

    @property
    def joints(self):
        return [
            "r_gripper_r_finger_joint",
            "r_gripper_l_finger_joint",
            "r_gripper_r_finger_tip_joint",
            "r_gripper_l_finger_tip_joint",
        ]

    @property
    def dof(self):
        return 4

    def contact_geoms(self):
        return [
            "r_gripper_l_finger",
            "r_gripper_l_finger_tip",
            "r_gripper_r_finger",
            "r_gripper_r_finger_tip",
        ]

    @property
    def visualization_sites(self):
        return ["grip_site", "grip_site_cylinder"]

    @property
    def left_finger_geoms(self):
        return ["r_gripper_l_finger", "r_gripper_l_finger_tip"]

    @property
    def right_finger_geoms(self):
        return ["r_gripper_r_finger", "r_gripper_r_finger_tip"]


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
