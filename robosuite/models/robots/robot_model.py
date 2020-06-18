from collections import OrderedDict

from robosuite.models.base import MujocoXML
from robosuite.utils import XMLError
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.transform_utils import euler2mat, mat2quat

import numpy as np

# List of bimanaul robots -- must be maintained manually
BIMANUAL_ROBOTS = {"Baxter"}

REGISTERED_ROBOTS = {}


def register_robot(target_class):
    REGISTERED_ROBOTS[target_class.__name__] = target_class


def create_robot(robot_name, *args, **kwargs):
    """Try to get the equivalent functionality of gym.make in a sloppy way."""
    if robot_name not in REGISTERED_ROBOTS:
        raise Exception(
            "Robot {} not found. Make sure it is a registered robot among: {}".format(
                robot_name, ", ".join(REGISTERED_ROBOTS)
            )
        )
    return REGISTERED_ROBOTS[robot_name](*args, **kwargs)


def check_bimanual(robot_name):
    """Utility function that returns whether the inputted robot_name is a bimanual robot or not"""
    return robot_name in BIMANUAL_ROBOTS


class RobotModelMeta(type):
    """Metaclass for registering robot arms"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["RobotModel"]

        if cls.__name__ not in _unregistered_envs:
            register_robot(cls)
        return cls


class RobotModel(MujocoXML, metaclass=RobotModelMeta):
    """Base class for all robot models."""

    def __init__(self, fname, idn=0, bottom_offset=(0, 0, 0)):
        """
        Initializes robot xml from file @fname.

        Args:
            fname (str): Path to relevant xml file to create this robot instance
            idn (int or str): Number or some other unique identification string for this robot instance
            bottom_offset (3-list/tuple): x,y,z offset desired from initial coordinates
        """
        super().__init__(fname)

        # Set id and add prefixes to all body names to prevent naming clashes
        self.idn = idn

        # Set offset
        self.bottom_offset = np.array(bottom_offset)

        # Update all xml element prefixes
        self.add_prefix(self.naming_prefix)

        # key: gripper name and value: gripper model
        self.grippers = OrderedDict()

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def add_gripper(self, gripper, arm_name=None):
        """
        Mounts gripper to arm.

        Throws error if robot already has a gripper or gripper type is incorrect.

        Args:
            gripper (MujocoGripper instance): gripper MJCF model
            arm_name (str): (Optional) name of arm mount -- defaults to eef_name if not specified
        """
        if arm_name is None:
            arm_name = self.eef_name
        if arm_name in self.grippers:
            raise ValueError("Attempts to add multiple grippers to one body")

        arm_subtree = self.worldbody.find(".//body[@name='{}']".format(arm_name))

        for actuator in gripper.actuator:

            if actuator.get("name") is None:
                raise XMLError("Actuator has no name")

            if not actuator.get("name").startswith("gripper"):
                raise XMLError(
                    "Actuator name {} does not have prefix 'gripper'".format(
                        actuator.get("name")
                    )
                )

        for body in gripper.worldbody:
            arm_subtree.append(body)

        self.merge(gripper, merge_body=False)
        self.grippers[arm_name] = gripper

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='{}']".format(self._root_))
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_ori(self, rot):
        """Rotates robot by rotation @rot (r,p,y) from its original orientation."""
        node = self.worldbody.find("./body[@name='{}']".format(self._root_))
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        rot = mat2quat(euler2mat(rot))[[3,0,1,2]]
        node.set("quat", array_to_string(rot))

    def set_joint_attribute(self, attrib, values):
        """
        Sets joint attributes, e.g.: friction loss, damping, etc.

        Args:
            attrib (str): Attribute to set for all joints
            values (ndarray): Values to set for each joint
        """
        assert values.size == self.dof, "Error setting joint attributes: " + \
            "Values must be same size as joint dimension. Got {}, expected {}!".format(values.size, self.dof)
        body = self._root_body_
        for i in range(len(self._links_)):
            body = body.find("./body[@name='{}']".format(self._links_[i]))
            joint = body.find("./joint[@name='{}']".format(self.joints[i]))
            joint.set(attrib, array_to_string(np.array([values[i]])))

    def correct_naming(self, names):
        """
        Corrects all xml names by adding the naming prefix to it and returns the name-corrected values

        @names (str, list, or dict): Name(s) to be corrected
        """
        if type(names) is str:
            return self.naming_prefix + names
        elif type(names) is list:
            return [self.naming_prefix + name for name in names]
        elif type(names) is dict:
            names = names.copy()
            for key, val in names.items():
                names[key] = self.correct_naming(val)
            return names
        else:
            # Assumed to be type error
            raise TypeError("Error: type of 'names' must be str, list, or dict!")

    # -------------------------------------------------------------------------------------- #
    # Public Properties: In general, these are the name-adjusted versions from the private   #
    #                    subclass implementations pulled from their respective raw xml files #
    # -------------------------------------------------------------------------------------- #

    @property
    def naming_prefix(self):
        """Returns a prefix to append to all xml names to prevent naming collisions"""
        return "robot{}_".format(self.idn)

    @property
    def joints(self):
        return self.correct_naming(self._joints)

    @property
    def eef_name(self):
        return self.correct_naming(self._eef_name)

    @property
    def robot_base(self):
        return self.correct_naming(self._robot_base)

    @property
    def actuators(self):
        return self.correct_naming(self._actuators)

    @property
    def contact_geoms(self):
        return self.correct_naming(self._contact_geoms)

    # -------------------------------------------------------------------------------------- #
    # -------------------------- Private Properties ---------------------------------------- #
    # -------------------------------------------------------------------------------------- #

    @property
    def _root_body_(self):
        """Returns xml element of the root body for this robot"""
        node = self.worldbody.find("./body[@name='{}']".format(self._root_))
        return node

    @property
    def _root_(self):
        return self.correct_naming(self._root)

    @property
    def _links_(self):
        return self.correct_naming(self._links)

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties based on their respective xml's #
    # -------------------------------------------------------------------------------------- #

    @property
    def dof(self):
        """Returns number of degrees of freedom for this robot"""
        raise NotImplementedError

    @property
    def gripper(self):
        """Returns default gripper type for this robot that gets added to end effector"""
        raise NotImplementedError

    @property
    def default_controller_config(self):
        """Returns name of default controller config file in the controllers/config directory for this robot."""
        raise NotImplementedError

    @property
    def init_qpos(self):
        """Returns default qpos of this robot"""
        raise NotImplementedError

    @property
    def base_xpos_offset(self):
        """
        Returns dict of various (x,y,z) tuple offsets relative to specific arenas placed at (0,0,0)
        Assumes robot is facing forwards (in the +x direction) when determining offset. Should have entries for each
        arena case; i.e.: "bins", "empty", and "table")
        """
        raise NotImplementedError

    @property
    def arm_type(self):
        """
        Type of robot arm. Should be either "bimanual" or "single" (or something else if it gets added in the future)
        """
        raise NotImplementedError

    @property
    def _joints(self):
        """Returns a list of joint names of the robot."""
        raise NotImplementedError

    @property
    def _eef_name(self):
        """Returns XML eef name for this robot to which grippers can be attached"""
        raise NotImplementedError

    @property
    def _robot_base(self):
        """Returns the base name of the physical base for this robot"""
        raise NotImplementedError

    @property
    def _actuators(self):
        """Returns the xml names for the pos, vel, and torq actuators for this robot. Should be a dict with entries
        for 'pos', 'vel', and 'torq' """
        raise NotImplementedError

    @property
    def _contact_geoms(self):
        """Returns list of contact geom names for this robot"""
        raise NotImplementedError

    @property
    def _root(self):
        """Returns the root name of the mujoco xml body"""
        raise NotImplementedError

    @property
    def _links(self):
        """Returns list of xml link names for this robot"""
        raise NotImplementedError
