"""
Gripper for Franka's Panda (has two fingers).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper


class PandaVICESGripperBase(Gripper):
    """
    Gripper for Franka's Panda (has two fingers).
    """

    def __init__(self):
        super().__init__(xml_path_completion("grippers/panda_vices_gripper.xml"))

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def joints(self):
        return ["finger_joint1", "finger_joint2"]

    @property
    def dof(self):
        return 2

    @property
    def visualization_sites(self):
        return ["grip_site", "grip_site_cylinder"]

    def contact_geoms(self):
        return [
            "hand_collision",
            "finger1_collisiona",
            "finger1_collisionb",
            "finger1_collisionc",
            "finger2_collisiona",
            "finger2_collisionb",
            "finger2_collisionc"
        ]

    @property
    def left_finger_geoms(self):
        return [
            "finger1_visuala",
            "finger1_visualb",
            "finger1_visualc"
        ]

    @property
    def right_finger_geoms(self):
        return [
            "finger2_visuala",
            "finger2_visualb",
            "finger2_visualc"
        ]


class PandaVICESGripper(PandaVICESGripperBase):
    """
    Modifies PandaGripperBase to only take one action.
    """

    def format_action(self, action):
        """
        1 => closed, -1 => open
        """
        assert len(action) == 1
        return np.array([-1 * action[0], 1 * action[0]])

    @property
    def dof(self):
        return 1
