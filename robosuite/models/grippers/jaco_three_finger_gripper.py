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
        return np.array([0.5,0.5,0.5,0.5,0.5,0.5])

    @property
    def _joints(self):
        return ["j2s7s300_joint_finger_1", "j2s7s300_joint_finger_tip_1",
                "j2s7s300_joint_finger_2", "j2s7s300_joint_finger_tip_2",
                "j2s7s300_joint_finger_3", "j2s7s300_joint_finger_tip_3"]

    @property
    def _actuators(self):
        return [
            "finger_1",
            "finger_2",
            "middle_finger"
        ]

    @property
    def _contact_geoms(self):
        return [
            "hand_collision",
            "finger1_collision",
            "finger2_collision",
            "finger3_collision",
            "fingertip1_collision",
            "fingertip2_collision",
            "fingertip3_collision",
            "fingertip1_pad_collision",
            "fingertip2_pad_collision",
            "fingertip3_pad_collision"
        ]

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_collision", "fingertip1_collision",
                            "finger3_collision", "fingertip3_collision",
                            "fingertip1_pad_collision", "fingertip3_pad_collision"],
            "right_finger": ["finger2_collision", "fingertip2_collision", "fingertip2_pad_collision"]
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
        self.current_action = np.clip(self.current_action + self.speed * np.array(action), -1.0, 1.0)
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
        self.current_action = np.clip(self.current_action + self.speed * np.array(action), -1.0, 1.0)
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
