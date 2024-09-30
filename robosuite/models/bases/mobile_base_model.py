"""
Defines the mobile base model
"""

from robosuite.models.bases.base_model import BaseModel


class MobileBaseModel(BaseModel):
    @property
    def naming_prefix(self):
        return "mobilebase{}_".format(self.idn)
