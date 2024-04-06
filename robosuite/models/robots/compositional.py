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



# class Compositional(ManipulatorModel):

# class B1Z1(Compositional):
#     @property
#     def default_base(self):
#         return "B1"
#     @property
#     def default_arms(self):
#         return {"right": "Z1"}
    
# class B1VX300S(Compositional):
#     @property
#     def default_base(self):
#         return "B1"
#     @property
#     def default_arms(self):
#         return {"right": "BX300S"}    
    
    
# class SpotArm(Compositional):
#     @property
#     def default_base(self):
#         return "Spot"
    
#     @property
#     def default_arms(self):
#         return {"right": "BDArm"}
