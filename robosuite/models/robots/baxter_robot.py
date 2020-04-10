import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Baxter(RobotModel):
    """Baxter is a hunky bimanual robot designed by Rethink Robotics."""

    def __init__(self, idn=0, bottom_offset=(0, 0, -0.913)):
        """
        Args:
            idn (int or str): Number or some other unique identification string for this robot instance
            bottom_offset (3-list/tuple): x,y,z offset desired from initial coordinates
        """
        super().__init__(xml_path_completion("robots/baxter/robot.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def dof(self):
        return 14

    @property
    def gripper(self):
        return {"right": "RethinkGripper",
                "left": "RethinkGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_baxter",
                "left": "default_baxter"}

    @property
    def init_qpos(self):
        # [right, left]
        # Arms half extended
        return np.array([0.403, -0.636, 0.114, 1.432, 0.735, 1.205, -0.269,
                        -0.403, -0.636, -0.114, 1.432, -0.735, 1.205, 0.269])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, 0.3, 0),
            "empty": (-0.29, 0, 0),
            "pegs": (-0.5, 0.15, 0),
            "table": lambda table_length: (-0.26 - table_length / 2, 0, 0)
        }

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _joints(self):
        return ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2",
                "left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]

    @property
    def _eef_name(self):
        return {"right": "right_hand",
                "left": "left_hand"}

    @property
    def _robot_base(self):
        return "base"

    @property
    def _actuators(self):
        return {
            "pos": [],  # No position actuators for baxter
            "vel": [],  # No velocity actuators for baxter
            "torq": ["torq_right_j0", "torq_right_j1", "torq_right_j2", "torq_right_j3",
                     "torq_right_j4", "torq_right_j5", "torq_right_j6",
                     "torq_left_j0", "torq_left_j1", "torq_left_j2", "torq_left_j3",
                     "torq_left_j4", "torq_left_j5", "torq_left_j6"]
        }

    @property
    def _contact_geoms(self):
        return ["right_s0_collision", "right_s1_collision", "right_e0_collision", "right_e1_collision",
                "right_w0_collision", "right_w1_collision", "right_w2_collision",
                "left_s0_collision", "left_s1_collision", "left_e0_collision", "left_e1_collision",
                "left_w0_collision", "left_w1_collision", "left_w2_collision"]

    @property
    def _root(self):
        return "base"

    @property
    def _links(self):
        return ["right_upper_shoulder", "right_lower_shoulder", "right_upper_elbow", "right_lower_elbow",
                "right_upper_forearm", "right_lower_forearm", "right_wrist",
                "left_upper_shoulder", "left_lower_shoulder", "left_upper_elbow", "left_lower_elbow",
                "left_upper_forearm", "left_lower_forearm", "left_wrist"]