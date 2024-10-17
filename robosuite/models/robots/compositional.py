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
            "table": lambda table_length: (-0.5 - table_length / 2, 0.0, 0.3),
        }


class SpotWithArmFloating(SpotArm):
    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}
