import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Z1Gripper(GripperModel):
    """
    Gripper for Franka's Panda (has two fingers).
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/z1_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.0])

    @property
    def _important_geoms(self):
        return {"stator": ["stator_col"], "mover": ["mover_col"]}
