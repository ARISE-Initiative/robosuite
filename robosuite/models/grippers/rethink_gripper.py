"""
Gripper with two fingers for Rethink Robots.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class RethinkGripperBase(GripperModel):
    """
    Gripper with long two-fingered parallel jaw.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/rethink_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["l_finger_g0", "l_finger_g1", "l_fingertip_g0", "l_fingerpad_g0"],
            "right_finger": ["r_finger_g0", "r_finger_g1", "r_fingertip_g0", "r_fingerpad_g0"],
            "left_fingerpad": ["l_fingerpad_g0"],
            "right_fingerpad": ["r_fingerpad_g0"],
        }


class RethinkGripper(RethinkGripperBase):
    """
    Modifies two finger base to only take one action.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action + np.array([1.0, -1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
