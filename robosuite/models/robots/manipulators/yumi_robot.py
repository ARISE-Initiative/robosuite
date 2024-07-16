import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Yumi(ManipulatorModel):
    """
    Yummi is a bimanual robot designed by ABB Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/yumi/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "YumiRightGripper", "left": "YumiLeftGripper"}

    @property
    def default_controller_config(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific default controller config names
        """
        return {"right": "default_yumi", "left": "default_yumi"}

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Note that this is a pose such that the arms are half extended

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        # [right, left]
        # Arms half extended
        return np.array(
            [
                # -0.802, 0.76, -0.17, -0.134, 0.137, 0.241, 1.57,
                # 0.745, 0.76, 0.202, -0.138, -0.0221, 0.374, 1.3,
                -0.67643,
                -0.432233,
                -0.35292,
                0.11764,
                -0.25305,
                0.831,
                0.99925,
                0.79407,
                -0.432233,
                0.14705,
                0.312945,
                3.34026,
                -0.608925,
                -1.23907,
            ]
        )

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0.05),
            "empty": (-0.29, 0, 0.05),
            "table": lambda table_length: (-0.26 - table_length / 2, 0, 0.05),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific eef names
        """
        return {"right": "right_hand", "left": "left_hand"}
