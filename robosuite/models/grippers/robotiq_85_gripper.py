"""
Robotiq 2F-85 gripper with an open/close variant.

The MJCF is based on the MuJoCo Menagerie / OGBench 4-bar linkage model.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Robotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_2f85_menagerie/2f85_robosuite.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.zeros(8)

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_pad1",
                "left_pad2",
            ],
            "right_finger": [
                "right_pad1",
                "right_pad2",
            ],
            "left_fingerpad": ["left_pad1", "left_pad2"],
            "right_fingerpad": ["right_pad1", "right_pad2"],
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
        return 0.20

    @property
    def dof(self):
        return 1
