"""Dex3 three-finger grippers of the SonicG1 (7 DOF per side), generated from SONIC's G1 model by
build_sonic_g1_assets.py."""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion

_FINGER_SIGN = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])


class SonicDex3LeftGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/sonic_dex3_left.xml"), idn=idn)

    def format_action(self, action):
        # 1-D open/close convenience for robosuite's gripper controller; the SONIC
        # composite controller drives all 7 finger torques directly.
        return np.sign(action) * _FINGER_SIGN

    @property
    def init_qpos(self):
        return np.zeros(7)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 7


class SonicDex3RightGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/sonic_dex3_right.xml"), idn=idn)

    def format_action(self, action):
        return np.sign(action) * -_FINGER_SIGN

    @property
    def init_qpos(self):
        return np.zeros(7)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 7
