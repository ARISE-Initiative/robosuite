import numpy as np

from robosuite.models.robots.manipulators.legged_manipulator_model import LeggedManipulatorModel
from robosuite.utils.mjcf_utils import find_parent, xml_path_completion


class GR1(LeggedManipulatorModel):
    """
    Tiago is a mobile manipulator robot created by PAL Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right", "left"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/gr1/robot.xml"), idn=idn)

    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def default_gripper(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm-specific gripper names
        """
        return {"right": "FourierRightHand", "left": "FourierLeftHand"}

    @property
    def default_controller_config(self):
        """
        Since this is bimanual robot, returns dict with `'right'`, `'left'`, `'head'`, `'torso'` keywords corresponding to their respective
        values

        Returns:
            dict: Dictionary containing arm,head,torso-specific default controller config names
        """
        return {
            "right": "default_gr1",
            "left": "default_gr1",
            "head": "default_gr1_head",
            "torso": "default_gr1_torso",
            "right_leg": "default_gr1",
            "left_leg": "default_gr1",
        }

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Note that this is a pose such that the arms are half extended

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        init_qpos = np.array([0.0] * 32)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[6:13] = right_arm_init
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[13:20] = left_arm_init
        return init_qpos

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.95),
            "empty": (-0.29, 0, 0.95),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.95),
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
        return {"right": "right_eef", "left": "left_eef"}


class GR1FixedLowerBody(GR1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

        # fix lower body
        self._remove_joint_actuation("leg")
        self._remove_free_joint()

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Note that this is a pose such that the arms are half extended

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        init_qpos = np.array([0.0] * 20)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[6:13] = right_arm_init
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[13:20] = left_arm_init
        return init_qpos

    @property
    def default_base(self):
        return "NoActuationBase"


class GR1FloatingBody(GR1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

        # fix lower body
        self._remove_joint_actuation("leg")
        self._remove_free_joint()

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Note that this is a pose such that the arms are half extended

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        init_qpos = np.array([0.0] * 20)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[6:13] = right_arm_init
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[13:20] = left_arm_init
        return init_qpos

    @property
    def default_base(self):
        return "FloatingLeggedBase"

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.97),
            "empty": (-0.29, 0, 0.97),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.97),
        }


class GR1ArmsOnly(GR1):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

        # fix lower body
        self._remove_joint_actuation("leg")
        self._remove_joint_actuation("head")
        self._remove_joint_actuation("torso")
        self._remove_free_joint()

    @property
    def init_qpos(self):
        """
        Since this is bimanual robot, returns [right, left] array corresponding to respective values

        Note that this is a pose such that the arms are half extended

        Returns:
            np.array: default initial qpos for the right, left arms
        """
        init_qpos = np.array([0.0] * 14)
        right_arm_init = np.array([0.0, -0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        left_arm_init = np.array([0.0, 0.1, 0.0, -1.57, 0.0, 0.0, 0.0])
        init_qpos[0:7] = right_arm_init
        init_qpos[7:14] = left_arm_init
        return init_qpos
