"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class PandaGripperBase(GripperModel):
    """
    Gripper for Franka's Panda (has two fingers).
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/panda_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 2

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def _joints(self):
        return ["finger_joint1", "finger_joint2"]

    @property
    def _actuators(self):
        return ["gripper_finger_joint1", "gripper_finger_joint2"]

    @property
    def _contact_geoms(self):
        return [
            "hand_collision",
            "finger1_collision",
            "finger2_collision",
            "finger1_tip_collision",
            "finger2_tip_collision",
        ]

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_tip_collision"],
            "right_finger": ["finger2_tip_collision"],
        }


class PandaGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Args:
            Binary action space
            -1 => open, 1 => closed
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        """
        How quickly the gripper opens / closes
        """
        return 0.01

    @property
    def dof(self):
        return 1
