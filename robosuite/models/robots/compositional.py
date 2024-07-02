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


class VX300SMobile(VX300S):
    """
    Variant of VX300S robot with mobile base. Currently serves as placeholder class.
    """

    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "VX300S"}


class B1Z1(Z1):
    """
    Variant of VX300S robot with mobile base. Currently serves as placeholder class.
    """

    @property
    def default_base(self):
        return "B1"

    @property
    def default_arms(self):
        return {"right": "Z1"}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0.65),
            "empty": (-0.6, 0, 0.65),
            "table": lambda table_length: (-0.55 - table_length / 2, 0.9, 0.65),
        }


class B1Z1Floating(B1Z1):
    """
    Variant of VX300S robot with mobile base. Currently serves as placeholder class.
    """

    @property
    def default_base(self):
        return "B1Floating"


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