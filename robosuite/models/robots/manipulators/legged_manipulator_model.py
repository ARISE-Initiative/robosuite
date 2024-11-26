from collections import OrderedDict

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import find_parent


class LeggedManipulatorModel(ManipulatorModel):
    """
    Base class for all manipulator models (robot arm(s) with gripper(s)).

    Args:
        fname (str): Path to relevant xml file from which to create this robot instance
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, fname, idn=0):
        # Always run super init first
        super().__init__(fname, idn=idn)

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
        for motor in self.actuator.findall(".//position"):
            if part_name in motor.get("name"):
                parent_body = find_parent(self.actuator, motor)
                parent_body.remove(motor)
                self._actuators.remove(motor.get("name").replace(self.naming_prefix, ""))
        for fixed in self.tendon.findall(".//fixed"):
            for joint in fixed.findall(".//joint"):
                if part_name in joint.get("joint"):
                    parent_body = find_parent(self.tendon, fixed)
                    parent_body.remove(fixed)
                    break

    def _remove_free_joint(self):
        # remove freejoint
        for freejoint in self.worldbody.findall(".//freejoint"):
            find_parent(self.worldbody, freejoint).remove(freejoint)

    @property
    def legs_joints(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no joints; i.e.: empty dict)
        """
        return self._legs_joints
