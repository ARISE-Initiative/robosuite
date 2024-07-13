"""
Dexterous hands for GR1 robot.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class AbilityLeftHand(GripperModel):
    """
    Dexterous left Ability hand.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/ability_left_hand.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.8] * 10)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 10


class AbilityRightHand(GripperModel):
    """
    Dexterous right Ability hand.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/ability_right_hand.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.8] * 10)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 10
