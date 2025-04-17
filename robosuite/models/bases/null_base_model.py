"""
Defines the null base model
"""

from robosuite.models.bases.robot_base_model import RobotBaseModel


class NullBaseModel(RobotBaseModel):
    @property
    def naming_prefix(self):
        return "nullbase{}_".format(self.idn)
