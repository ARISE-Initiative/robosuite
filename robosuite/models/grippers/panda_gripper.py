"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper


class PandaGripperBase(Gripper):
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
    def _left_finger_geoms(self):
        return ["finger1_tip_collision"]

    @property
    def _right_finger_geoms(self):
        return ["finger2_tip_collision"]


class PandaGripper(PandaGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        Args:
            -1 => open, 1 => closed
        """
        assert len(action) == 1
        return np.array([-1 * action[0], 1 * action[0]])

    @property
    def dof(self):
        return 1
