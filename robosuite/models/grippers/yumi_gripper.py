"""
Dexterous hands for GR1 robot.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class YumiRightGripper(GripperModel):
    """
    Right gripper of Yumi Robot.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/yumi_right_gripper.xml"), idn=idn)

    def format_action(self, action):
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, -1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def init_qpos(self):
        return np.array([0.025] * 2)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 1


class YumiLeftGripper(GripperModel):
    """
    Left gripper of Yumi Robot.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/yumi_left_gripper.xml"), idn=idn)

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, -1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def init_qpos(self):
        return np.array([0.025] * 2)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 1
