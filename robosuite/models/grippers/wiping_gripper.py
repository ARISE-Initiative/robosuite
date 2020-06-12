"""
Gripper without fingers to wipe a surface
"""
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class WipingGripper(GripperModel):
    def __init__(self, idn=0):
        super().__init__(xml_path_completion('grippers/wiping_gripper.xml'), idn=idn)

    def format_action(self, action):
        return action

    @property
    def dof(self):
        return 0

    @property
    def init_qpos(self):
        return None

    @property
    def _joints(self):
        return []

    @property
    def _actuators(self):
        return []

    @property
    def _contact_geoms(self):
        return ["wiping_surface", "wiper_col1", "wiper_col2"]

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["finger1_tip_collision"],
            "right_finger": ["finger2_tip_collision"],
            "corners": ["wiping_corner1", "wiping_corner2", "wiping_corner3", "wiping_corner4"]
        }
