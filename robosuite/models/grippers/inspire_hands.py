"""
Dexterous hands for GR1 robot.
"""
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class InspireLeftHand(GripperModel):
    """
    Dexterous left hand of GR1 robot

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/inspire_left_hand.xml"), idn=idn)

    def format_action(self, action):
        # the more correct way is to add <equality> tag in the xml
        # however the tag makes finger movement laggy, so manually copy the value for finger joints
        # 0 is thumb rot, no copying. Thumb bend has 3 joints, so copy 3 times. Other fingers has 2 joints, so copy 2 times.
        assert len(action) == self.dof
        action = np.array(action)
        indices = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        return action[indices]

    @property
    def init_qpos(self):
        return np.array([0.0] * 12)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 6

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "l_thumb_proximal_col",
                "l_thumb_proximal_2_col",
                "l_thumb_middle_col",
                "l_thumb_distal_col",
            ],
            "right_finger": [
                "l_index_proximal_col",
                "l_index_distal_col",
                "l_middle_proximal_col",
                "l_middle_distal_col",
                "l_ring_proximal_col",
                "l_ring_distal_col",
                "l_pinky_proximal_col",
                "l_pinky_distal_col",
            ],
            "left_fingerpad": [
                "l_thumb_proximal_col",
                "l_thumb_proximal_2_col",
                "l_thumb_middle_col",
                "l_thumb_distal_col",
            ],
            "right_fingerpad": [
                "l_index_proximal_col",
                "l_index_distal_col",
                "l_middle_proximal_col",
                "l_middle_distal_col",
                "l_ring_proximal_col",
                "l_ring_distal_col",
                "l_pinky_proximal_col",
                "l_pinky_distal_col",
            ],
        }


class InspireRightHand(GripperModel):
    """
    Dexterous right hand of GR1 robot

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/inspire_right_hand.xml"), idn=idn)

    def format_action(self, action):
        # the more correct way is to add <equality> tag in the xml
        # however the tag makes finger movement laggy, so manually copy the value for finger joints
        # 0 is thumb rot, no copying. Thumb bend has 3 joints, so copy 3 times. Other fingers has 2 joints, so copy 2 times.
        assert len(action) == self.dof
        action = np.array(action)
        indices = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        return action[indices]

    @property
    def init_qpos(self):
        return np.array([0.0] * 12)

    @property
    def speed(self):
        return 0.15

    @property
    def dof(self):
        return 6

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "r_thumb_proximal_col",
                "r_thumb_proximal_2_col",
                "r_thumb_middle_col",
                "r_thumb_distal_col",
            ],
            "right_finger": [
                "r_index_proximal_col",
                "r_index_distal_col",
                "r_middle_proximal_col",
                "r_middle_distal_col",
                "r_ring_proximal_col",
                "r_ring_distal_col",
                "r_pinky_proximal_col",
                "r_pinky_distal_col",
            ],
            "left_fingerpad": [
                "r_thumb_proximal_col",
                "r_thumb_proximal_2_col",
                "r_thumb_middle_col",
                "r_thumb_distal_col",
            ],
            "right_fingerpad": [
                "r_index_proximal_col",
                "r_index_distal_col",
                "r_middle_proximal_col",
                "r_middle_distal_col",
                "r_ring_proximal_col",
                "r_ring_distal_col",
                "r_pinky_proximal_col",
                "r_pinky_distal_col",
            ],
        }
