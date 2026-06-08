"""Dex3 three-finger grippers for the SonicG1 (7 DOF/side). Split from SONIC's
validated model so joints/meshes/ranges match the integrated model the C++ dex3
command stream maps to."""
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper
from robosuite.utils.mjcf_utils import xml_path_completion

_FINGER_SIGN = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])


@register_gripper
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


@register_gripper
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
