"""
Gripper with multiple control dimensions.
For example not only for the grasping action, but also the adhesion action.
"""
from robosuite.models.grippers.gripper_model import GripperModel


class FlexGripperModel(GripperModel):
    def __init__(self, fname, idn=0):
        super().__init__(fname, idn=idn)
