from robosuite.models.base import MujocoXML, MujocoModel
from robosuite.utils.mjcf_utils import array_to_string, ROBOT_COLLISION_COLOR, sort_elements
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


class RobotModel(MujocoXML, MujocoModel, metaclass=RobotModelMeta):
    """
    Base class for all robot models.

    Args:
        fname (str): Path to relevant xml file from which to create this robot instance
        idn (int or str): Number or some other unique identification string for this robot instance
        bottom_offset (3-array of float): x,y,z offset desired from initial coordinates
    """

    def __init__(self, fname, idn=0, bottom_offset=(0, 0, 0)):
        super().__init__(fname)

        # Set id and add prefixes to all body names to prevent naming clashes
        self.idn = idn

        # Set offset
        self.bottom_offset = np.array(bottom_offset)

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

        # Update all xml element prefixes
        self.add_prefix(self.naming_prefix)

        # Recolor all collision geoms appropriately
        self.recolor_collision_geoms(ROBOT_COLLISION_COLOR)

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def set_base_xpos(self, pos):
        """
        Places the robot on position @pos.

        Args:
            pos (3-array): (x,y,z) position to place robot base
        """
        node = self.worldbody.find("./body[@name='{}']".format(self.root_body))
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_ori(self, rot):
        """
        Rotates robot by rotation @rot from its original orientation.

        Args:
            rot (3-array): (r,p,y) euler angles specifying the orientation for the robot base
        """
        node = self.worldbody.find("./body[@name='{}']".format(self.root_body))
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        rot = mat2quat(euler2mat(rot))[[3,0,1,2]]
        node.set("quat", array_to_string(rot))

    def set_joint_attribute(self, attrib, values):
        """
        Sets joint attributes, e.g.: friction loss, damping, etc.

        Args:
            attrib (str): Attribute to set for all joints
            values (n-array): Values to set for each joint

        Raises:
            AssertionError: [Inconsistent dimension sizes]
        """
        assert values.size == len(self._elements["joints"]), "Error setting joint attributes: " + \
            "Values must be same size as joint dimension. Got {}, expected {}!".format(values.size, self.dof)
        for i, joint in enumerate(self._elements["joints"]):
            joint.set(attrib, array_to_string(np.array([values[i]])))

    # -------------------------------------------------------------------------------------- #
    # Public Properties: In general, these are the name-adjusted versions from the private   #
    #                    attributes pulled from their respective raw xml files               #
    # -------------------------------------------------------------------------------------- #

    @property
    def naming_prefix(self):
        return "robot{}_".format(self.idn)

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
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def sensors(self):
        """
        Returns:
            dict: (Default is no sensors; i.e.: empty dict)
        """
        return {}

    @property
    def dof(self):
        """
        Defines the number of DOF of the robot

        Returns:
            int: robot DOF
        """
        return len(self._joints)

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

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
    def _important_geoms(self):
        """
        Returns:
             dict: (Default is no important geoms; i.e.: empty dict)
        """
        return {}
