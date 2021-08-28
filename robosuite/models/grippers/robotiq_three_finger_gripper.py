"""
Gripper with 11-DoF controlling three fingers and its open/close variant.
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class RobotiqThreeFingerGripperBase(GripperModel):
    """
    Gripper with 11 dof controlling three fingers.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_s.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.zeros(11)

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "f1_l0",
                "f1_l1",
                "f1_l2",
                "f1_l3",
                "f2_l0",
                "f2_l1",
                "f2_l2",
                "f2_l3",
                "f1_tip_collision",
                "f2_tip_collision",
                "f1_pad_collision",
                "f2_pad_collision",
            ],
            "right_finger": [
                "f3_l0",
                "f3_l1",
                "f3_l2",
                "f3_l3",
                "finger_middle_tip_collision",
                "finger_middle_pad_collision",
            ],
            "left_fingerpad": ["f1_pad_collision", "f2_pad_collision"],
            "right_fingerpad": ["finger_middle_pad_collision"],
        }


class RobotiqThreeFingerGripper(RobotiqThreeFingerGripperBase):
    """
    1-DoF variant of RobotiqThreeFingerGripperBase.
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
        self.current_action = np.clip(
            self.current_action + self.speed * np.array(action), -1.0, 1.0
        )
        # Automatically set the scissor joint to "closed" position by default
        return np.concatenate([self.current_action * np.ones(3), [-1]])

    @property
    def speed(self):
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
        Maps continuous action into binary output
        all -1 => open, all 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 4
