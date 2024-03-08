"""
Defines the base class of all mounts
"""
import numpy as np

from robosuite.models.base import MujocoXMLModel
from robosuite.utils.mjcf_utils import MOUNT_COLLISION_COLOR


class MobileBaseModel(MujocoXMLModel):
    """
    Base class for mounts that will be attached to robots. Note that this model's root body will be directly
    appended to the robot's root body, so all offsets should be taken relative to that.

    Args:
        fname (str): Path to relevant xml file to create this mount instance
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, fname, idn=0):
        super().__init__(fname, idn=idn)

        # Grab mount offset (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
        self.rotation_offset = np.fromstring(
            self.worldbody[0].attrib.get("quat", "1 0 0 0"), dtype=np.float64, sep=" "
        )[[1, 2, 3, 0]]

    # -------------------------------------------------------------------------------------- #
    # Properties: In general, these are the name-adjusted versions from the private          #
    #             subclass implementations pulled from their respective raw xml files        #
    # -------------------------------------------------------------------------------------- #

    @property
    def naming_prefix(self):
        return "mount{}_".format(self.idn)

    @property
    def _important_sites(self):
        """
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def _important_geoms(self):
        """
        Returns:
             dict: (Default is no important geoms; i.e.: empty dict)
        """
        return {}

    @property
    def _important_sensors(self):
        """
        Returns:
            dict: (Default is no sensors; i.e.: empty dict)
        """
        return {}

    @property
    def contact_geom_rgba(self):
        return MOUNT_COLLISION_COLOR

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def top_offset(self):
        """
        Returns vector from model root body to model top.
        This should correspond to the distance from the root body to the actual mounting surface
        location of this mount.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model.

        Helps us put models programmatically without them flying away due to a huge initial contact force.
        Must be defined by subclass.

        Returns:
            float: radius
        """
        raise NotImplementedError
