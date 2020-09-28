import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Jaco(RobotModel):
    """
    Jaco is a kind and assistive robot created by Kinova

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
        bottom_offset (3-array): (x,y,z) offset desired from initial coordinates
    """

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        super().__init__(xml_path_completion("robots/jaco/robot.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def dof(self):
        return 7

    @property
    def gripper(self):
        return "JacoThreeFingerGripper"

    @property
    def default_controller_config(self):
        return "default_jaco"

    @property
    def init_qpos(self):
        return np.array([3.192, 3.680, -0.000, 1.170, 0.050, 3.760, -1.510])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }

    @property
    def arm_type(self):
        return "single"

    @property
    def _joints(self):
        return ["j2s7s300_joint_1", "j2s7s300_joint_2", "j2s7s300_joint_3", "j2s7s300_joint_4",
                "j2s7s300_joint_5", "j2s7s300_joint_6", "j2s7s300_joint_7"]

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
            "torq": ["torq_j1", "torq_j2", "torq_j3",
                     "torq_j4", "torq_j5", "torq_j6", "torq_j7"]
        }

    @property
    def _contact_geoms(self):
        return ["s_collision", "ah1_collision", "ah2_collision", "f_collision",
                "ws1_collision", "ws2_collision"]

    @property
    def _root(self):
        return 'base'

    @property
    def _links(self):
        return ["j2s7s300_link_1", "j2s7s300_link_2", "j2s7s300_link_3", "j2s7s300_link_4",
                "j2s7s300_link_5", "j2s7s300_link_6", "j2s7s300_link_7"]
