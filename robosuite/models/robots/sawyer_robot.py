import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Sawyer(RobotModel):
    """Sawyer is a witty single-arm robot designed by Rethink Robotics."""

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this robot instance
            bottom_offset (3-list/tuple): x,y,z offset desired from initial coordinates
        """
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def dof(self):
        return 7

    @property
    def gripper(self):
        return "TwoFingerGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        # TODO: Determine which start is better
        #return np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])

    @property
    def arm_type(self):
        return "single"

    @property
    def _joints(self):
        return ["right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6"]

    @property
    def _eef_name(self):
        return "right_hand"

    @property
    def _robot_base(self):
        return "base"

    @property
    def _actuators(self):
        return {
            "pos": [],  # No position actuators for sawyer
            "vel": [],  # No velocity actuators for sawyer
            "torq": ["torq_right_j0", "torq_right_j1", "torq_right_j2", "torq_right_j3",
                     "torq_right_j4", "torq_right_j5", "torq_right_j6"]
        }

    @property
    def _contact_geoms(self):
        return ["link0_collision", "link1_collision", "link2_collision", "link3_collision", "link4_collision",
                "link5_collision", "link6_collision"]

    @property
    def _root(self):
        return 'base'

    @property
    def _links(self):
        return ["right_l0", "right_l1", "right_l2", "right_l3", "right_l4", "right_l5", "right_l6"]
