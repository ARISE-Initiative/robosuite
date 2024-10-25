"""
Defines the base class of all mobile bases
"""
from xml.etree import ElementTree as ET

import numpy as np

from robosuite.models.base import MujocoXMLModel
from robosuite.utils.mjcf_utils import MOUNT_COLLISION_COLOR, find_elements, find_parent


class LegBaseModel(MujocoXMLModel):
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

    @property
    def init_qpos(self):
        raise NotImplementedError

    # -------------------------------------------------------------------------------------- #
    # Properties: In general, these are the name-adjusted versions from the private          #
    #             subclass implementations pulled from their respective raw xml files        #
    # -------------------------------------------------------------------------------------- #

    @property
    def naming_prefix(self):
        return "leg{}_".format(self.idn)

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

    def _remove_joint_actuation(self, part_name):
        for joint in self.worldbody.findall(".//joint"):
            if part_name in joint.get("name"):
                parent_body = find_parent(self.worldbody, joint)
                parent_body.remove(joint)
                self._joints.remove(joint.get("name").replace(self.naming_prefix, ""))
        for motor in self.actuator.findall(".//motor"):
            if part_name in motor.get("name"):
                parent_body = find_parent(self.actuator, motor)
                parent_body.remove(motor)
                self._actuators.remove(motor.get("name").replace(self.naming_prefix, ""))

    def _remove_free_joint(self):
        # remove freejoint
        for freejoint in self.worldbody.findall(".//freejoint"):
            find_parent(self.worldbody, freejoint).remove(freejoint)

    def _add_mobile_joint(self):
        """This is a special processing of leg base, which converts leg bases into floating ones without users to manually modify the xml file."""

        # Define the dictionary representing the attributes of the tag
        forward_joint_attributes = {
            "name": self.naming_prefix + "joint_mobile_forward",
            "pos": "0 0 0",
            "axis": "0 1 0",
            "type": "slide",
            "limited": "false",
            "damping": "0",
            "armature": "0.0",
            "frictionloss": "250",
        }

        side_joint_attributes = {
            "name": self.naming_prefix + "joint_mobile_side",
            "pos": "0 0 0",
            "axis": "1 0 0",
            "type": "slide",
            "limited": "false",
            "damping": "0",
            "armature": "0.0",
            "frictionloss": "250",
        }

        yaw_joint_attributes = {
            "name": self.naming_prefix + "joint_mobile_yaw",
            "pos": "0 0 0",
            "axis": "0 0 1",
            "type": "hinge",
            "limited": "false",
            "damping": "0",
            "armature": "0.0",
            "frictionloss": "250",
        }

        # Create the root element
        forward_joint = ET.Element("joint")
        side_joint = ET.Element("joint")
        yaw_joint = ET.Element("joint")

        # Set the attributes of the root element from the dictionary
        for key, value in forward_joint_attributes.items():
            forward_joint.set(key, value)
        for key, value in side_joint_attributes.items():
            side_joint.set(key, value)
        for key, value in yaw_joint_attributes.items():
            yaw_joint.set(key, value)


        forward_actuation_attributes = {
            "ctrllimited": "true",
            "ctrlrange": "-1.00 1.00",
            "joint": self.naming_prefix + "joint_mobile_forward",
            "kv": "1000",
            "name": self.naming_prefix + "actuator_mobile_forward",
            "forcelimited": "true",
            "forcerange": "-600 600"
        }
        side_actuation_attributes = {
            "ctrllimited": "true",
            "ctrlrange": "-1.00 1.00",
            "joint": self.naming_prefix + "joint_mobile_side",
            "kv": "1000",
            "name": self.naming_prefix + "actuator_mobile_side",
            "forcelimited": "true",
            "forcerange": "-600 600"
        }
        yaw_actuation_attributes = {
            "ctrllimited": "true",
            "ctrlrange": "-1.50 1.50",
            "joint": self.naming_prefix + "joint_mobile_yaw",
            "kv": "1500",
            "name": self.naming_prefix + "actuator_mobile_yaw",
            "forcelimited": "true",
            "forcerange": "-600 600"
        }

        forward_actuation = ET.Element("velocity")
        side_actuation = ET.Element("velocity")
        yaw_actuation = ET.Element("velocity")

        for key, value in forward_actuation_attributes.items():
            forward_actuation.set(key, str(value))
        for key, value in side_actuation_attributes.items():
            side_actuation.set(key, str(value))
        for key, value in yaw_actuation_attributes.items():
            yaw_actuation.set(key, str(value))

        root = find_elements(self.worldbody, tags="body", return_first=True)

        for joint in [forward_joint, side_joint, yaw_joint]:
            root.append(joint)
            self._joints.append(joint.get("name").replace(self.naming_prefix, ""))

        actuators = find_elements(self.actuator, tags="actuator", return_first=True)
        for actuator in [forward_actuation, side_actuation, yaw_actuation]:
            actuators.append(actuator)
            self._actuators.append(actuator.get("name").replace(self.naming_prefix, ""))

    @property
    def default_base(self):
        return "NullMobileBase"
