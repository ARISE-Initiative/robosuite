import numpy as np

from robosuite.models.robots import *


class PandaOmron(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}

    @property
    def init_qpos(self):
        return np.array([0, np.pi / 16.0 - 0.2, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.4, np.pi / 4])

    @property
    def init_torso_qpos(self):
        return np.array([0.2])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }


class SpotWithArm(SpotArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-1.05, -0.1, 0.3),
            "empty": (-1.1, 0, 0.3),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, 0.46),
        }


class SpotWithArmFloating(SpotArm):
    def __init__(self, idn=0):
        super().__init__(idn=idn)

    @property
    def init_qpos(self):
        return np.array([0.0, -2, 1.26, -0.335, 0.862, 0.0])

    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.7, -0.1, 0.7),
            "empty": (-0.6, 0, 0.7),
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, 0.7),
        }


class PandaDexRH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireRightHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, 0.5, 0.5]}


class PandaDexLH(Panda):
    @property
    def default_gripper(self):
        return {"right": "InspireLeftHand"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.5, -0.5, 0.5, 0.5]}


class GR1Rethink(GR1ArmsOnly):
    @property
    def default_gripper(self):
        return {"right": "RethinkGripper", "left": "RethinkGripper"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.0, 0.0, 0.0], "left": [0.0, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0, 0, 1, 0], "left": [0, 0, 1, 0]}


class SpotArmRethink(SpotWithArm):
    @property
    def default_gripper(self):
        return {"right": "RethinkGripper"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.05, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.707107, 0, 0.707107, 0]}


class SpotArmRobotiqGripper(SpotWithArm):
    @property
    def default_gripper(self):
        return {"right": "Robotiq85Gripper"}

    @property
    def gripper_mount_pos_offset(self):
        return {"right": [0.05, 0.0, 0.0]}

    @property
    def gripper_mount_quat_offset(self):
        return {"right": [0.707107, 0, 0.707107, 0]}
