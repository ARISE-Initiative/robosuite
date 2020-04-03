"""
Defines the base class of all grippers
"""
from robosuite.models.base import MujocoXML


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

        # Update all xml element prefixes
        self.add_prefix(self.naming_prefix)

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
    # Properties: In general, these are the name-adjusted versions from the private   #
    #             subclass implementations pulled from their respective raw xml files #
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
    def joints(self):
        return [self.naming_prefix + joint for joint in self._joints]

    @property
    def actuators(self):
        return [self.naming_prefix + actuator for actuator in self._actuators]

    @property
    def contact_geoms(self):
        return [self.naming_prefix + geom for geom in self._contact_geoms]

    @property
    def visualization_geoms(self):
        return [self.naming_prefix + geom for geom in self._visualization_geoms]

    @property
    def left_finger_geoms(self):
        return [self.naming_prefix + geom for geom in self._left_finger_geoms]

    @property
    def right_finger_geoms(self):
        return [self.naming_prefix + geom for geom in self._right_finger_geoms]

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
    def _left_finger_geoms(self):
        """
        Geoms corresponding to left finger of a gripper
        """
        return []

    @property
    def _right_finger_geoms(self):
        """
        Geoms corresponding to right finger of a gripper
        """
        return []

