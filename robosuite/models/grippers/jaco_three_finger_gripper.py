"""
Gripper for Kinova's Jaco robot arm (has three fingers).
"""

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class JacoThreeFingerGripperBase(GripperModel):
    """
    Gripper for Kinova's Jaco robot arm (has three fingers).

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/jaco_three_finger_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.5, 0, 0.5, 0, 0.5, 0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "index_proximal_collision",
                "index_distal_collision",
                "index_tip_collision",
                "pinky_proximal_collision",
                "pinky_distal_collision",
                "pinky_tip_collision",
                "index_tip_collision",
                "pinky_pad_collision",
            ],
            "right_finger": [
                "thumb_proximal_collision",
                "thumb_distal_collision",
                "thumb_tip_collision",
                "thumb_pad_collision",
            ],
            "left_fingerpad": ["index_pad_collision", "pinky_pad_collision"],
            "right_fingerpad": ["thumb_pad_collision"],
        }


class JacoThreeFingerGripper(JacoThreeFingerGripperBase):
    """
    Modifies JacoThreeFingerGripperBase to only take one action.
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
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action - self.speed * np.sign(action), -1.0, 1.0)
        # NOTE(YL): format 1dof to default 3dof
        return self.current_action * np.array([1, 1, 1])

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1


class JacoThreeFingerDexterousGripper(JacoThreeFingerGripperBase):
    """
    Dexterous variation of the Jaco gripper in which all finger are actuated independently
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        all -1 => open, all 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action - self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 3
