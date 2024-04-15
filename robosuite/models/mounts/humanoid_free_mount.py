"""
Free mount used for humanoids.
"""
import numpy as np

from robosuite.models.mounts.mount_model import MountModel
from robosuite.utils.mjcf_utils import xml_path_completion


class HumanoidFreeMount(MountModel):
    """
    Dummy Mount to signify no mount for humanoids.

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("mounts/null_mount.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0.03))

    @property
    def horizontal_radius(self):
        return 0
