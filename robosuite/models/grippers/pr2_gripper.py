"""
6-DoF gripper with its open/close variant
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class PR2Gripper(GripperModel):
    """
    4 DoF PR2 Gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/pr2_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.548, 0.0, 0.548, 0.0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "gripper_l_finger_link_collision",
            ],
            "right_finger": ["gripper_r_finger_link_collision"],
            "left_fingerpad": ["gripper_l_finger_tip_link_collision"],
            "right_fingerpad": ["gripper_r_finger_tip_link_collision"],
        }

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
            self.current_action * np.array([1.0, 1.0, 1.0, 1.0]) - self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.10

    @property
    def dof(self):
        return 1
