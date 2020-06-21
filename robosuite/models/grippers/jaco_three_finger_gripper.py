"""
Gripper for Kinova's Jaco robot arm (has three fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class JacoThreeFingerGripperBase(GripperModel):
    """
    Gripper for Kinova's Jaco robot arm (has three fingers).
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/jaco_three_finger_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 3

    @property
    def init_qpos(self):
        return np.array([0.5, 0, 0.5, 0, 0.5, 0])

    @property
    def _joints(self):
        return [
            "joint_thumb", "joint_thumb_distal",
            "joint_index", "joint_index_distal",
            "joint_pinky", "joint_pinky_distal",
        ]

    @property
    def _actuators(self):
        return [
            "thumb",
            "index",
            "pinky",
        ]

    @property
    def _contact_geoms(self):
        return [
            "hand_collision",
            "thumb_proximal_collision", "thumb_distal_collision", "thumb_pad_collision",
            "index_proximal_collision", "index_distal_collision", "index_pad_collision",
            "pinky_proximal_collision", "pinky_distal_collision", "pinky_pad_collision",
        ]

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["index_proximal_collision", "index_distal_collision", "index_pad_collision",
                            "pinky_proximal_collision", "pinky_distal_collision", "pinky_pad_collision"],
            "right_finger": ["thumb_proximal_collision", "thumb_distal_collision", "thumb_pad_collision"]
        }


class JacoThreeFingerGripper(JacoThreeFingerGripperBase):
    """
    Modifies JacoThreeFingerGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Args:
            -1 => open, 1 => closed
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action - self.speed * np.array(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        """
        How quickly the gripper opens / closes
        """
        return 0.005

    @property
    def dof(self):
        return 1


class JacoThreeFingerDexterousGripper(JacoThreeFingerGripperBase):
    """
    Dexterous variation of the Jaco gripper in which all finger are actuated independently
    """
    def format_action(self, action):
        """
        Args:
            all -1 => open, all 1 => closed
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action - self.speed * np.array(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        """
        How quickly the gripper opens / closes
        """
        return 0.005

    @property
    def dof(self):
        return 3
