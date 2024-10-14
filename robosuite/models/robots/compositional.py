from robosuite.models.robots import *


class PandaOmron(Panda):
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

class PandaDexRH(Panda):
    @property
    def default_gripper(self):
        return "Panda"

    @property
    def default_gripper(self):
        return {"right": "InspireRightHandForPanda"}

class PandaDexLH(Panda):
    @property
    def default_gripper(self):
        return "Panda"

    @property
    def default_gripper(self):
        return {"right": "InspireLeftHandForPanda"}