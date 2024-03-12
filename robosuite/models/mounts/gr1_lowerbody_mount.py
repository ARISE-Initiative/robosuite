"""
GR1's lower body as mount.
"""
import numpy as np

from robosuite.models.mounts.mount_model import MountModel
from robosuite.utils.mjcf_utils import xml_path_completion


class GR1LowerBodyMount(MountModel):
    """
    Mount (GR1's lower body) used for GR1.

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("mounts/gr1_lowerbody_mount.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        # TODO: This may be inaccurate; just a placeholder for now
        return 0.25
