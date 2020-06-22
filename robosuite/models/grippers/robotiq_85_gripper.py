"""
6-DoF gripper with its open/close variant
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class Robotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(xml_path_completion("grippers/robotiq_gripper_85.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 2

    @property
    def init_qpos(self):
        return [0., 0., 0., 0., 0., 0.]

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
            "finger_1",
            "finger_2",
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
            "left_finger_pad_collision"
            "right_finger_pad_collision"
        ]

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "robotiq_85_gripper_joint_0_L",
                "robotiq_85_gripper_joint_1_L",
                "robotiq_85_gripper_joint_2_L",
                "robotiq_85_gripper_joint_3_L",
                "left_finger_pad_collision",
            ],
            "right_finger": [
                "robotiq_85_gripper_joint_0_R",
                "robotiq_85_gripper_joint_1_R",
                "robotiq_85_gripper_joint_2_R",
                "robotiq_85_gripper_joint_3_R",
                "right_finger_pad_collision",
            ],
        }


class Robotiq85Gripper(Robotiq85GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Args:
            action: -1 => open, 1 => closed
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.array(action), -1.0, 1.0)
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
