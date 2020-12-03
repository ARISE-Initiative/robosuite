"""
Defines the base class of all grippers
"""
from robosuite.models.base import MujocoXML, MujocoModel
from robosuite.utils.mjcf_utils import GRIPPER_COLLISION_COLOR, sort_elements
import numpy as np


class GripperModel(MujocoXML, MujocoModel):
    """
    Base class for grippers

    Args:
        fname (str): Path to relevant xml file to create this gripper instance
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, fname, idn=0):
        super().__init__(fname)

        # Set id number and add prefixes to all body names to prevent naming clashes
        self.idn = idn

        # Parse element tree to get all relevant bodies, joints, actuators, and geom groups
        self._elements = sort_elements(root=self.root)
        assert len(self._elements["root_body"]) == 1, "Invalid number of root bodies found for robot model. Expected 1," \
                                                      "got {}".format(len(self._elements["root_body"]))
        self._elements["root_body"] = self._elements["root_body"][0]
        self._elements["bodies"] = [self._elements["root_body"]] + self._elements["bodies"] if \
            "bodies" in self._elements else [self._elements["root_body"]]
        self._root_body = self._elements["root_body"].get("name")
        self._bodies = [e.get("name") for e in self._elements.get("bodies", [])]
        self._joints = [e.get("name") for e in self._elements.get("joints", [])]
        self._actuators = [e.get("name") for e in self._elements.get("actuators", [])]
        self._sites = [e.get("name") for e in self._elements.get("sites", [])]
        self._contact_geoms = [e.get("name") for e in self._elements.get("contact_geoms", [])]
        self._visual_geoms = [e.get("name") for e in self._elements.get("visual_geoms", [])]

        # Set variable to hold current action being outputted
        self.current_action = np.zeros(self.dof)

        # Update all xml element prefixes
        self.add_prefix(self.naming_prefix)

        # Update collision geom colors
        self.recolor_collision_geoms(GRIPPER_COLLISION_COLOR)

        # Grab gripper offset (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
        self.rotation_offset = np.fromstring(self.worldbody[0].attrib.get("quat", "1 0 0 0"),
                                             dtype=np.float64, sep=" ")[[1, 2, 3, 0]]

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
        return "gripper{}_".format(self.idn)

    @property
    def root_body(self):
        return self.correct_naming(self._root_body)

    @property
    def bodies(self):
        return self.correct_naming(self._bodies)

    @property
    def joints(self):
        return self.correct_naming(self._joints)

    @property
    def actuators(self):
        return self.correct_naming(self._actuators)

    @property
    def sites(self):
        return self.correct_naming(self._sites)

    @property
    def contact_geoms(self):
        return self.correct_naming(self._contact_geoms)

    @property
    def visual_geoms(self):
        return self.correct_naming(self._visual_geoms)

    @property
    def important_geoms(self):
        return self.correct_naming(self._important_geoms)

    @property
    def important_sites(self):
        """
        Sites used to aid visualization by human. (usually "grip_site" and "grip_cylinder")
        (and should be hidden from robots)

        Returns:
            dict:

                :`'grip_site'`: Name of grip actuation intersection location site
                :`'grip_cylinder'`: Name of grip actuation z-axis location site
        """
        return {"grip_site": self.naming_prefix + "grip_site",
                "grip_cylinder": self.naming_prefix + "grip_site_cylinder"}

    @property
    def sensors(self):
        """
        Sensor names for each gripper (usually "force_ee" and "torque_ee")

        Returns:
            dict:

                :`'force_ee'`: Name of force eef sensor for this gripper
                :`'torque_ee'`: Name of torque eef sensor for this gripper
        """
        return {"force_ee": self.naming_prefix + "force_ee",
                "torque_ee": self.naming_prefix + "torque_ee"}

    @property
    def speed(self):
        """
        How quickly the gripper opens / closes

        Returns:
            float: Speed of the gripper
        """
        return 0.0

    @property
    def dof(self):
        """
        Defines the number of DOF of the gripper

        Returns:
            int: gripper DOF
        """
        return len(self._actuators)

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def init_qpos(self):
        """
        Defines the default rest (open) qpos of the gripper

        Returns:
            np.array: Default init qpos of this gripper
        """
        raise NotImplementedError

    @property
    def _important_geoms(self):
        """
        Geoms corresponding to important components of the gripper (by default, left_finger, right_finger,
        left_fingerpad, right_fingerpad).
        Note that these are the raw string names directly pulled from a gripper's corresponding XML file,
        NOT the adjusted name with an auto-generated naming prefix

        Note that this should be a dict of lists.

        Returns:
            dict of list: Raw XML important geoms, where each set of geoms are grouped as a list and are
            organized by keyword string entries into a dict
        """
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "right_fingerpad": [],
        }
