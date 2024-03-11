import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class GoogleRobot(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/google_robot/robot.xml"), idn=idn)

        # Set joint damping
        # self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

    @property
    def default_base(self):
        return "NullMobileBase"

    @property
    def default_gripper(self):
        return "GoogleGripper"

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 3 * np.pi / 8, 0.0, np.pi / 2, 0.0])

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
