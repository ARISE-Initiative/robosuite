"""
Gripper with 11-DoF controlling three fingers and its open/close variant.
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class RobotiqThreeFingerGripperBase(GripperModel):
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
        return 4

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
            "finger_middle_joint_3"
        ]

    @property
    def _actuators(self):
        return [
            "finger_1",
            "finger_2",
            "middle_finger",
            "finger_scissor"
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
            action: -1 => open, 1 => closed
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + self.speed * action, -1.0, 1.0)
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


class RobotiqThreeFingerDexterousGripper(RobotiqThreeFingerGripperBase):
    """
    Dexterous variation of the 3-finger Robotiq gripper in which all finger are actuated independently as well
    as the scissor joint between fingers 1 and 2
    """

    def format_action(self, action):
        """
        Args:
            action: all -1 => open, all 1 => closed
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + self.speed * action, -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        """
        How quickly the gripper opens / closes
        """
        return 0.01

    @property
    def dof(self):
        return 4
