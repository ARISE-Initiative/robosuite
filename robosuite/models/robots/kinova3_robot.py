import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Kinova3(RobotModel):
    """
    The Gen3 robot is sparkly newest addition to the Kinova line

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
        bottom_offset (3-array): (x,y,z) offset desired from initial coordinates
    """

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        super().__init__(xml_path_completion("robots/kinova3/robot.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def dof(self):
        return 7

    @property
    def gripper(self):
        return "Robotiq85Gripper"

    @property
    def default_controller_config(self):
        return "default_kinova3"

    @property
    def init_qpos(self):
        return np.array([0.000, 0.650, 0.000, 1.890, 0.000, 0.600, -np.pi / 2])

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
        return ["Actuator1", "Actuator2", "Actuator3", "Actuator4",
                "Actuator5", "Actuator6", "Actuator7"]

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
        return ["s_collision", "ha1_collision", "ha2_collision", "f_collision",
                "w1_collision", "w2_collision", "b_collision"]

    @property
    def _root(self):
        return 'base'

    @property
    def _links(self):
        return ["Shoulder_Link", "HalfArm1_Link", "HalfArm2_Link", "ForeArm_Link",
                "SphericalWrist1_Link", "SphericalWrist2_Link", "Bracelet_Link"]
