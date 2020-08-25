import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class UR5e(RobotModel):
    """
    UR5e is a sleek and elegant new robot created by Universal Robots

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
        bottom_offset (3-array): (x,y,z) offset desired from initial coordinates
    """

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        super().__init__(xml_path_completion("robots/ur5e/robot.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def dof(self):
        return 6

    @property
    def gripper(self):
        return "Robotiq85Gripper"

    @property
    def default_controller_config(self):
        return "default_ur5e"

    @property
    def init_qpos(self):
        return np.array([-0.470, -1.735, 2.480, -2.275, -1.590, -0.420])

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
        return ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

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
                     "torq_j4", "torq_j5", "torq_j6"]
        }

    @property
    def _contact_geoms(self):
        return ["shoulder_col", "upperarm_col", "forearm_col",
                "wrist1_col", "wrist2_col", "wrist3_col"]

    @property
    def _root(self):
        return 'base'

    @property
    def _links(self):
        return ["shoulder_link", "upper_arm_link", "forearm_link",
                "wrist_1_link", "wrist_2_link", "wrist_3_link"]
