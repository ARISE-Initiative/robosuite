import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion


class Panda(Robot):
    """Panda is a sensitive single-arm robot designed by Franka."""

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this robot instance
            bottom_offset (3-list/tuple): x,y,z offset desired from initial coordinates
        """
        super().__init__(xml_path_completion("robots/panda/robot.xml"), idn=idn, bottom_offset=bottom_offset)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

    @property
    def dof(self):
        return 7

    @property
    def gripper(self):
        return "PandaGripper"

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi/4])

    @property
    def arm_type(self):
        return "single"

    @property
    def _joints(self):
        return ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    @property
    def _eef_name(self):
        return "right_hand"

    @property
    def _robot_base(self):
        return "base"

    @property
    def _actuators(self):
        return {
            "pos": [],  # No position actuators for panda
            "vel": [],  # No velocity actuators for panda
            "torq": ["torq_j1", "torq_j2", "torq_j3", "torq_j4", "torq_j5", "torq_j6", "torq_j7"]
        }

    @property
    def _contact_geoms(self):
        return ["link1_collision", "link2_collision", "link3_collision", "link4_collision",
                "link5_collision", "link6_collision", "link7_collision"]

    @property
    def _root(self):
        return "link0"

    @property
    def _links(self):
        return ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
