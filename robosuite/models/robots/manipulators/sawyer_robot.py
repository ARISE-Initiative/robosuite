import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Sawyer(ManipulatorModel):
    """
    Sawyer is a witty single-arm robot designed by Rethink Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
        bottom_offset (3-array): (x,y,z) offset desired from initial coordinates
    """

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        super().__init__(xml_path_completion("robots/sawyer/robot.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def default_gripper(self):
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])

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
