"""
Dexterous hands for GR1 robot.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class G1ThreeFingerLeftGripper(GripperModel):
    """
    Three-finger left gripper  of G1 robot

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/g1_three_finger_left_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action * np.array([0., 1., 1., -1., -1., -1., -1.])

    @property
    def init_qpos(self):
        return np.array([0.0] * 7)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 7


class G1ThreeFingerRightGripper(GripperModel):
    """
    Three-finger right gripper of G1 robot

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/g1_three_finger_right_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action * np.array([0., -1., -1., 1., 1., 1., 1.])

    @property
    def init_qpos(self):
        return np.array([0.0] * 7)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 7