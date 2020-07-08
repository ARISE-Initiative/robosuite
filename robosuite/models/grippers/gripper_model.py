"""
Defines the base class of all grippers
"""
from robosuite.models.base import MujocoXML
import numpy as np


class GripperModel(MujocoXML):
    """Base class for grippers"""

    def __init__(self, fname, idn=0):
        """
        Args:
            fname (str): Path to relevant xml file to create this gripper instance
            idn (int or str): Number or some other unique identification string for this gripper instance
        """
        super().__init__(fname)

        # Set id number and add prefixes to all body names to prevent naming clashes
        self.idn = idn

        # Set variable to hold current action being outputted
        self.current_action = np.zeros(self.dof)

        # Update all xml element prefixes
        self.add_prefix(self.naming_prefix)

        # Set public attributes with prefixes appended to values
        self.joints = [self.naming_prefix + joint for joint in self._joints]
        self.actuators = [self.naming_prefix + actuator for actuator in self._actuators]
        self.contact_geoms = [self.naming_prefix + geom for geom in self._contact_geoms]
        self.visualization_geoms = [self.naming_prefix + geom for geom in self._visualization_geoms]

        # Grab gripper offset (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
        self.rotation_offset = np.fromstring(self.worldbody[0].attrib.get("quat", "1 0 0 0"),
                                             dtype=np.float64, sep=" ")[[1, 2, 3, 0]]

        # Loop through dict of remaining miscellaneous geoms
        self.important_geoms = {}
        for k, v in self._important_geoms.items():
            self.important_geoms[k] = [self.naming_prefix + vv for vv in v]

    def hide_visualization(self):
        """
        Hides all visualization geoms and sites.
        This should be called before rendering to agents
        """
        for site_name in self.visualization_sites.values():
            site = self.worldbody.find(".//site[@name='{}']".format(site_name))
            site.set("rgba", "0 0 0 0")
        for geom_name in self.visualization_geoms:
            geom = self.worldbody.find(".//geom[@name='{}']".format(geom_name))
            geom.set("rgba", "0 0 0 0")

    def format_action(self, action):
        """
        Given (-1,1) abstract control as np-array
        returns the (-1,1) control signals
        for underlying actuators as 1-d np array
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------------------- #
    # Properties: In general, these are the name-adjusted versions from the private          #
    #             subclass implementations pulled from their respective raw xml files        #
    # -------------------------------------------------------------------------------------- #
    @property
    def naming_prefix(self):
        """Returns a prefix to append to all xml names to prevent naming collisions"""
        return "gripper{}_".format(self.idn)

    @property
    def visualization_sites(self):
        """
        Returns a dict of sites corresponding to the geoms
        used to aid visualization by human. (usually "site" and "cylinder")
        (and should be hidden from robots)
        """
        return {"grip_site": self.naming_prefix + "grip_site",
                "grip_cylinder": self.naming_prefix + "grip_site_cylinder"}

    @property
    def sensors(self):
        """
        Returns a dict of sensor names for each gripper (usually "force_ee" and "torque_ee"
        """
        return {"force_ee": self.naming_prefix + "force_ee",
                "torque_ee": self.naming_prefix + "torque_ee"}

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties based on their respective xml's #
    # (note: only if they exist)                                                             #
    # -------------------------------------------------------------------------------------- #

    @property
    def dof(self):
        """
        Returns the number of DOF of the gripper
        """
        raise NotImplementedError

    @property
    def init_qpos(self):
        """
        Returns rest(open) qpos of the gripper
        """
        raise NotImplementedError

    @property
    def _joints(self):
        """
        Returns a list of joint names of the gripper
        """
        raise NotImplementedError

    @property
    def _actuators(self):
        """
        Returns a list of actuator names of the gripper
        """
        raise NotImplementedError

    @property
    def _contact_geoms(self):
        """
        Returns a list of names corresponding to the geoms
        used to determine contact with the gripper.
        """
        return []

    @property
    def _visualization_geoms(self):
        """
        Returns a list of sites corresponding to the geoms
        used to aid visualization by human.
        (and should be hidden from robots)
        """
        return []

    @property
    def _important_geoms(self):
        """
        Geoms corresponding to important components of the gripper

        Note that this should be a dict of lists
        """
        return {
            "left_finger": [],
            "right_finger": []
        }

