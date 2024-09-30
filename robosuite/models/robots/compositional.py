from robosuite.models.robots import *


class PandaMobile(Panda):
    """
    Variant of Panda robot with mobile base. Currently serves as placeholder class.
    """

    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}


class SpotArm(BDArm):
    @property
    def default_base(self):
        return "Spot"

    @property
    def default_arms(self):
        return {"right": "BDArm"}


class SpotArmFloating(SpotArm):
    @property
    def default_base(self):
        return "SpotFloating"

    @property
    def default_arms(self):
        return {"right": "BDArm"}
