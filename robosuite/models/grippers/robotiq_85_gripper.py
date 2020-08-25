"""
6-DoF gripper with its open/close variant
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class Robotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_85.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 2

    @property
    def init_qpos(self):
        return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])

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


class Robotiq85Gripper(Robotiq85GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
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
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
