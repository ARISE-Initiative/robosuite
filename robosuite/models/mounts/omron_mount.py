"""
Omron LD-60 Mobile Base.
"""
import numpy as np

from robosuite.models.mounts.mount_model import MountModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OmronMount(MountModel):
    """
    Omron LD-60 Mobile Base.

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("mounts/omron_mount.xml"), idn=idn)

    @property
    def top_offset(self):
        # return np.array((0, 0, 0.922 - 0.10))
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        # TODO: This may be inaccurate; just a placeholder for now
        return 0.25

    @property
    def is_mobile(self):
        return True
