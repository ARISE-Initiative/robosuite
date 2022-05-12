"""
Gripper without fingers to reach into spaces
"""
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class ReachGripper(GripperModel):
    """
    A Reach Gripper with no actuation and enabled with sensors to detect contact forces

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/reach_gripper.xml"), idn=idn)

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
            "corners": ["sensor_corner1", "sensor_corner2", "sensor_corner3", "sensor_corner4"],
        }
