import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class BDGripper(GripperModel):
    """
    Gripper for the Spot Arm.
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/bd_gripper.xml"), idn=idn)

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([1.]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action
    @property
    def init_qpos(self):
        return np.array([-1.57])

    @property
    def speed(self):
        return 0.2

    @property
    def _important_geoms(self):
        return {"arm_link_fngr": ["arm_link_fngr_0", "arm_link_fngr_1", "left_finger_coll", "right_finger_coll"]}
