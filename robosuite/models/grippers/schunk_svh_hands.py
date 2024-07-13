"""
Dexterous hands for GR1 robot.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SchunkSvhLeftHand(GripperModel):
    """
    Dexterous left SVH hand from Schunk.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/schunk_svh_left_hand.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0] * 20)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 20


class SchunkSvhRightHand(GripperModel):
    """
    Dexterous right SVH hand from Schunk.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/schunk_svh_right_hand.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0] * 20)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 20
