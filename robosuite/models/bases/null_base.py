"""
Rethink's Generic Mount (Officially used on Sawyer).
"""
import numpy as np

from robosuite.models.bases.null_base_model import NullBaseModel
from robosuite.utils.mjcf_utils import xml_path_completion


class NullBase(NullBaseModel):
    """
    Dummy mobile base to signify no mount.

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("bases/null_base.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return 0
