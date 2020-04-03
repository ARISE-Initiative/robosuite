"""
6-DoF gripper with its open/close variant
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class RobotiqGripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/robotiq_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 6

    @property
    def init_qpos(self):
        return [3.3161, 0., 0., 0., 0., 0.]

    @property
    def _joints(self):
        return [
            "robotiq_85_left_knuckle_joint",
            "robotiq_85_left_inner_knuckle_joint",
            "robotiq_85_left_finger_tip_joint",
            "robotiq_85_right_knuckle_joint",
            "robotiq_85_right_inner_knuckle_joint",
            "robotiq_85_right_finger_tip_joint",
        ]

    @property
    def _actuators(self):
        return [
            "gripper_robotiq_85_left_knuckle_joint",
            "gripper_robotiq_85_left_inner_knuckle_joint",
            "gripper_robotiq_85_left_finger_tip_joint",
            "gripper_robotiq_85_right_knuckle_joint",
            "gripper_robotiq_85_right_inner_knuckle_joint",
            "gripper_robotiq_85_right_finger_tip_joint"
        ]

    @property
    def _contact_geoms(self):
        return [
            "robotiq_85_gripper_joint_0_L",
            "robotiq_85_gripper_joint_1_L",
            "robotiq_85_gripper_joint_0_R",
            "robotiq_85_gripper_joint_1_R",
            "robotiq_85_gripper_joint_2_L",
            "robotiq_85_gripper_joint_3_L",
            "robotiq_85_gripper_joint_2_R",
            "robotiq_85_gripper_joint_3_R",
        ]

    @property
    def _left_finger_geoms(self):
        return [
            "robotiq_85_gripper_joint_0_L",
            "robotiq_85_gripper_joint_1_L",
            "robotiq_85_gripper_joint_2_L",
            "robotiq_85_gripper_joint_3_L",
        ]

    @property
    def _right_finger_geoms(self):
        return [
            "robotiq_85_gripper_joint_0_R",
            "robotiq_85_gripper_joint_1_R",
            "robotiq_85_gripper_joint_2_R",
            "robotiq_85_gripper_joint_3_R",
        ]


class RobotiqGripper(RobotiqGripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Args:
            action: -1 => open, 1 => closed
        """
        assert len(action) == 1
        return np.ones(6) * action

    @property
    def dof(self):
        return 1
