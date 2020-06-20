"""
Gripper with 140mm Jaw width from Robotiq (has two fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class Robotiq140GripperBase(GripperModel):
    """
    Gripper with 140mm Jaw width from Robotiq (has two fingers).
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/robotiq_gripper_140.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 2

    @property
    def init_qpos(self):
        return np.array([0.012, 0.065, 0.065, -0.012, 0.065, 0.065])

    @property
    def _joints(self):
        return ["finger_joint", "left_inner_finger_joint",
                "left_inner_knuckle_joint", "right_outer_knuckle_joint",
                "right_inner_finger_joint", "right_inner_knuckle_joint"]

    @property
    def _actuators(self):
        return [
            "finger_1",
            "finger_2",
        ]

    @property
    def _contact_geoms(self):
        return [
            "hand_collision",
            "left_outer_knuckle_collision",
            "left_outer_finger_collision",
            "left_inner_finger_collision",
            "left_fingertip_collision",
            "left_inner_knuckle_collision",
            "right_outer_knuckle_collision",
            "right_outer_finger_collision",
            "right_inner_finger_collision",
            "right_fingertip_collision",
            "right_inner_knuckle_collision",

        ]

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision"
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision"
            ],
        }


class Robotiq140Gripper(Robotiq140GripperBase):
    """
    Modifies Robotiq140GripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Args:
            -1 => open, 1 => closed
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + np.array([1.0, -1.0]) * self.speed * action, -1.0, 1.0)
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
