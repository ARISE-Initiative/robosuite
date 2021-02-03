from robosuite.models.base import MujocoXMLModel
from robosuite.utils.mjcf_utils import array_to_string, ROBOT_COLLISION_COLOR, string_to_array
from robosuite.utils.transform_utils import euler2mat, mat2quat

import numpy as np

REGISTERED_ROBOTS = {}


def register_robot(target_class):
    REGISTERED_ROBOTS[target_class.__name__] = target_class


def create_robot(robot_name, *args, **kwargs):
    """
    Instantiates a Robot object.

    Args:
        robot_name (str): Name of the robot to initialize
        *args: Additional arguments to pass to the specific Robot class initializer
        **kwargs: Additional arguments to pass to the specific Robot class initializer

    Returns:
        Robot: Desired robot

    Raises:
        Exception: [Invalid robot name]
    """
    if robot_name not in REGISTERED_ROBOTS:
        raise Exception(
            "Robot {} not found. Make sure it is a registered robot among: {}".format(
                robot_name, ", ".join(REGISTERED_ROBOTS)
            )
        )
    return REGISTERED_ROBOTS[robot_name](*args, **kwargs)


class RobotModelMeta(type):
    """Metaclass for registering robot arms"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["RobotModel", "ManipulatorModel"]

        if cls.__name__ not in _unregistered_envs:
            register_robot(cls)
        return cls


class RobotModel(MujocoXMLModel, metaclass=RobotModelMeta):
    """
    Base class for all robot models.

    Args:
        fname (str): Path to relevant xml file from which to create this robot instance
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, fname, idn=0):
        super().__init__(fname, idn=idn)

        # Define other variables that get filled later
        self.mount = None

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

        # By default, set small frictionloss and armature values
        self.set_joint_attribute(attrib="frictionloss", values=0.1 * np.ones(self.dof), force=False)
        self.set_joint_attribute(attrib="damping", values=0.1 * np.ones(self.dof), force=False)
        self.set_joint_attribute(attrib="armature",
                                 values=np.array([5.0 / (i + 1) for i in range(self.dof)]), force=False)

    def set_base_xpos(self, pos):
        """
        Places the robot on position @pos.

        Args:
            pos (3-array): (x,y,z) position to place robot base
        """
        self._elements["root_body"].set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_ori(self, rot):
        """
        Rotates robot by rotation @rot from its original orientation.

        Args:
            rot (3-array): (r,p,y) euler angles specifying the orientation for the robot base
        """
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        rot = mat2quat(euler2mat(rot))[[3,0,1,2]]
        self._elements["root_body"].set("quat", array_to_string(rot))

    def set_joint_attribute(self, attrib, values, force=True):
        """
        Sets joint attributes, e.g.: friction loss, damping, etc.

        Args:
            attrib (str): Attribute to set for all joints
            values (n-array): Values to set for each joint
            force (bool): If True, will automatically override any pre-existing value. Otherwise, if a value already
                exists for this value, it will be skipped

        Raises:
            AssertionError: [Inconsistent dimension sizes]
        """
        assert values.size == len(self._elements["joints"]), "Error setting joint attributes: " + \
            "Values must be same size as joint dimension. Got {}, expected {}!".format(values.size, self.dof)
        for i, joint in enumerate(self._elements["joints"]):
            if force or joint.get(attrib, None) is None:
                joint.set(attrib, array_to_string(np.array([values[i]])))

    def add_mount(self, mount):
        """
        Mounts @mount to arm.

        Throws error if robot already has a mount or if mount type is incorrect.

        Args:
            mount (MountModel): mount MJCF model

        Raises:
            ValueError: [mount already added]
        """
        if self.mount is not None:
            raise ValueError("Mount already added for this robot!")

        # First adjust mount's base position
        offset = self.base_offset - mount.top_offset
        mount._elements["root_body"].set("pos", array_to_string(offset))

        self.merge(mount, merge_body=self.root_body)

        self.mount = mount

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    # -------------------------------------------------------------------------------------- #
    # Public Properties: In general, these are the name-adjusted versions from the private   #
    #                    attributes pulled from their respective raw xml files               #
    # -------------------------------------------------------------------------------------- #

    @property
    def naming_prefix(self):
        return "robot{}_".format(self.idn)

    @property
    def dof(self):
        """
        Defines the number of DOF of the robot

        Returns:
            int: robot DOF
        """
        return len(self._joints)

    @property
    def bottom_offset(self):
        """
        Returns vector from model root body to model bottom.
        By default, this is equivalent to this robot's mount's (bottom_offset - top_offset) + this robot's base offset

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        return (self.mount.bottom_offset - self.mount.top_offset) + self._base_offset if \
            self.mount is not None else self._base_offset

    @property
    def horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model. This method takes into
        account the mount horizontal radius as well

        Returns:
            float: radius
        """
        return max(self._horizontal_radius, self.mount.horizontal_radius)

    @property
    def contact_geom_rgba(self):
        return ROBOT_COLLISION_COLOR

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def default_mount(self):
        """
        Defines the default mount type for this robot that gets added to root body (base)

        Returns:
            str: Default mount name to add to this robot
        """
        raise NotImplementedError

    @property
    def default_controller_config(self):
        """
        Defines the name of default controller config file in the controllers/config directory for this robot.

        Returns:
            str: filename of default controller config for this robot
        """
        raise NotImplementedError

    @property
    def init_qpos(self):
        """
        Defines the default rest qpos of this robot

        Returns:
            np.array: Default init qpos of this robot
        """
        raise NotImplementedError

    @property
    def base_xpos_offset(self):
        """
        Defines the dict of various (x,y,z) tuple offsets relative to specific arenas placed at (0,0,0)
        Assumes robot is facing forwards (in the +x direction) when determining offset. Should have entries for each
        arena case; i.e.: "bins", "empty", and "table")

        Returns:
            dict: Dict mapping arena names to robot offsets from the global origin (dict entries may also be lambdas
                for variable offsets)
        """
        raise NotImplementedError

    @property
    def top_offset(self):
        """
        Returns vector from model root body to model top.
        Useful for, e.g. placing models on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def _horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model.

        Helps us put models programmatically without them flying away due to a huge initial contact force.
        Must be defined by subclass.

        Returns:
            float: radius
        """
        raise NotImplementedError

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
