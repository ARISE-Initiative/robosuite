"""
Gripper with adhesion mechanism for Suction and Particle Jamming
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class AdhesivePalm(GripperModel):
    """
    Gripper with adhesion mechanism for Suction and Particle Jamming

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/adhesion_palm.xml"), idn=idn)

    @property
    def init_qpos(self):
        return np.zeros(1)

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["gripper_base"],
            "right_finger": ["gripper_base"],
            "left_fingerpad": ["eef"],
            "right_fingerpad": ["eef"],
        }

    @property
    def _important_sensors(self):
        return {
            "force_ee": "force_ee",
            "torque_ee": "torque_ee",
        }

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
            self.current_action + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def dof(self):
        return 1

    @property
    def speed(self):
        return 10.0
