"""
Defines the base class of all grippers
"""
from robosuite.models.base import MujocoXML


class Gripper(MujocoXML):
    """Base class for grippers"""

    def __init__(self, fname):
        super().__init__(fname)

    def format_action(self, action):
        """
        Given (-1,1) abstract control as np-array
        returns the (-1,1) control signals
        for underlying actuators as 1-d np array
        """
        raise NotImplementedError

    @property
    def init_qpos(self):
        """
        Returns rest(open) qpos of the gripper
        """
        raise NotImplementedError

    @property
    def dof(self):
        """
        Returns the number of DOF of the gripper
        """
        raise NotImplementedError

    @property
    def joints(self):
        """
        Returns a list of joint names of the gripper
        """
        raise NotImplementedError

    def contact_geoms(self):
        """
        Returns a list of names corresponding to the geoms
        used to determine contact with the gripper.
        """
        return []

    @property
    def visualization_sites(self):
        """
        Returns a list of sites corresponding to the geoms
        used to aid visualization by human.
        (and should be hidden from robots)
        """
        return []

    @property
    def visualization_geoms(self):
        """
        Returns a list of sites corresponding to the geoms
        used to aid visualization by human.
        (and should be hidden from robots)
        """
        return []

    @property
    def left_finger_geoms(self):
        """
        Geoms corresponding to left finger of a gripper
        """
        raise NotImplementedError

    @property
    def right_finger_geoms(self):
        """
        Geoms corresponding to raise finger of a gripper
        """
        raise NotImplementedError

    def hide_visualization(self):
        """
        Hides all visualization geoms and sites.
        This should be called before rendering to agents
        """
        for site_name in self.visualization_sites:
            site = self.worldbody.find(".//site[@name='{}']".format(site_name))
            site.set("rgba", "0 0 0 0")
        for geom_name in self.visualization_geoms:
            geom = self.worldbody.find(".//geom[@name='{}']".format(geom_name))
            geom.set("rgba", "0 0 0 0")
