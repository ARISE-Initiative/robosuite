"""
Rethink's Alternative Mount (Officially used on Baxter).
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.mounts.mount_model import MountModel


class RethinkMinimalMount(MountModel):
    """
    Mount officially used for Rethink's Baxter Robot. Includes only a wheeled pedestal.

    Args:
        idn (int or str): Number or some other unique identification string for this mount instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("mounts/rethink_minimal_mount.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, -0.062))

    @property
    def horizontal_radius(self):
        # TODO: This may be inaccurate; just a placeholder for now
        return 0.25
