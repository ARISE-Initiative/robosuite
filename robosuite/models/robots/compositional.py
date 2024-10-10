from robosuite.models.robots import *


class PandaOmron(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.6, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }


class SpotArm(BDArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "BDArm"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-1.05, -0.1, 0.7),
            "empty": (-1.1, 0, 0.7),
            "table": lambda table_length: (-0.8 - table_length / 2, 0.0, 0.7),
        }


class SpotArmFloating(SpotArm):
    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "BDArm"}
