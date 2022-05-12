import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Kinova3(ManipulatorModel):
    """
    The Gen3 robot is the sparkly newest addition to the Kinova line

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/kinova3/robot.xml"), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "ReachKinovaGripper"

    @property
    def default_controller_config(self):
        return "default_kinova3"

    @property
    def init_qpos(self):
        #return np.array([0.000, 0.650, 0.000, 1.890, 0.000, 0.600, -np.pi / 2])
        #return np.array([-0.142, 1.374, -0.160, 2.291, -0.249, -2.104, -1.497])
        #return np.array([-0.098,1.445,0.042,2.271,-0.098,-2.132,-1.632])
        return np.array([-0.087, 1.618, -0.028, 2.050, -0.147, -2.125, -1.567])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
