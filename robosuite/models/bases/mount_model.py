"""
Defines the MountModel class (Fixed Base that is mounted to the robot)
"""

from robosuite.models.bases.base_model import BaseModel


class MountModel(BaseModel):
    @property
    def naming_prefix(self) -> str:
        return "fixed_mount{}_".format(self.idn)
