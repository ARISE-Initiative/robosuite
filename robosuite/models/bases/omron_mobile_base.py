"""
Omron LD-60 Mobile Base.
"""
import numpy as np

from robosuite.models.bases.mobile_base_model import MobileBaseModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OmronMobileBase(MobileBaseModel):
    """
    Omron LD-60 Mobile Base.

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("bases/omron_mobile_base.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        # TODO: This may be inaccurate; just a placeholder for now
        return 0.25

    @property
    def actuators(self):
        pf = self.naming_prefix
        return [
            f"{pf}actuator_x",
            f"{pf}actuator_y",
            f"{pf}actuator_rot",
        ]

    @property
    def height_actuator(self):
        return "{}actuator_z".format(self.naming_prefix)
