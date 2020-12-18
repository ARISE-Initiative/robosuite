"""
Gripper without fingers to wipe a surface
"""
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper_model import GripperModel


class WipingGripper(GripperModel):
    """
    A Wiping Gripper with no actuation and enabled with sensors to detect contact forces

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion('grippers/wiping_gripper.xml'), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "right_fingerpad": [],
            "corners": ["wiping_corner1", "wiping_corner2", "wiping_corner3", "wiping_corner4"]
        }
