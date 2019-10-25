"""
Gripper without fingers to wipe a surface
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper


class WipingGripper(Gripper):
    def __init__(self):
        super().__init__(xml_path_completion('grippers/wiping_gripper.xml'))

    def format_action(self, action):
        return action
 #       return np.ones(4) * action

    @property
    def init_qpos(self):
        return []

    @property
    def joints(self):
        return []

    @property
    def dof(self):
        return 0

    def contact_geoms(self):
        return ["wiping_surface", "wiper_col1", "wiper_col2"]
