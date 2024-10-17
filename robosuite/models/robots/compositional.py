from robosuite.models.robots import *


class PandaOmron(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}


class SpotWithArm(SpotArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}


class SpotWithArmFloating(SpotArm):
    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "SpotArm"}
