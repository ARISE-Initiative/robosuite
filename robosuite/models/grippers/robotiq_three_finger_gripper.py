"""
Gripper with 11-DoF controlling three fingers and its open/close variant.
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper


class RobotiqThreeFingerGripperBase(Gripper):
    """
    Gripper with 11 dof controlling three fingers.
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/robotiq_gripper_s.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 11

    @property
    def init_qpos(self):
        return np.zeros(11)

    @property
    def _joints(self):
        return [
            "palm_finger_1_joint",
            "finger_1_joint_1",
            "finger_1_joint_2",
            "finger_1_joint_3",
            "palm_finger_2_joint",
            "finger_2_joint_1",
            "finger_2_joint_2",
            "finger_2_joint_3",
            "finger_middle_joint_1",
            "finger_middle_joint_2",
            "finger_middle_joint_3",
        ]

    @property
    def _actuators(self):
        return [
            "gripper_palm_finger_1_joint",
            "gripper_finger_1_joint_1",
            "gripper_finger_1_joint_2",
            "gripper_finger_1_joint_3",
            "gripper_palm_finger_2_joint",
            "gripper_finger_2_joint_1",
            "gripper_finger_2_joint_2",
            "gripper_finger_2_joint_3",
            "gripper_finger_middle_joint_1",
            "gripper_finger_middle_joint_2",
            "gripper_finger_middle_joint_3"
        ]

    @property
    def _contact_geoms(self):
        return [
            "f1_l0",
            "f1_l1",
            "f1_l2",
            "f1_l3",
            "f2_l0",
            "f2_l1",
            "f2_l2",
            "f2_l3",
            "f3_l0",
            "f3_l1",
            "f3_l2",
            "f3_l3",
        ]


class RobotiqThreeFingerGripper(RobotiqThreeFingerGripperBase):
    """
    1-DoF variant of RobotiqThreeFingerGripperBase.
    """

    def format_action(self, action):
        """
        Args:
            action: 1 => open, -1 => closed
        """
        movement = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
        return -1 * movement * action

    @property
    def dof(self):
        return 1
