import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Yam(ManipulatorModel):
    """
    Yam robot model.
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/yam/yam.xml"), idn=idn)

        # Set joint damping (copied from Panda, may need tuning)
        # self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

    @property
    def default_base(self):
        return None

    @property
    def default_gripper(self):
        return None  # Integrated gripper

    @property
    def default_controller_config(self):
        return "default_panda"

    @property
    def init_qpos(self):
        # 6 joints
        return np.array([0, 0, 0, 0, 0, 0])

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




