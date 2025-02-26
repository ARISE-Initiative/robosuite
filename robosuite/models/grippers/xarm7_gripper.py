"""
Gripper for UFactory's XArm7 (has two fingers).
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class XArm7GripperBase(GripperModel):
    """
    Gripper for UFactory's XArm7.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/xarm7_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.02, 0.0, 0.0, -0.02, 0.0, 0.0])

    @property
    def _important_geoms(self):
        return {
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }


class XArm7Gripper(XArm7GripperBase):
    """
    Modifies XArm7 Gripper to only take one action.
    """

    def format_action(self, action):
        """
        -1 => fully open, +1 => fully closed
        """
        assert len(action) == self.dof  # i.e., 1
        # Suppose self.current_action is also shape (1,). Then:
        delta = self.speed * np.sign(action)
        self.current_action = np.clip(self.current_action + delta, -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def dof(self):
        return 1
