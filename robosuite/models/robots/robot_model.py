from copy import deepcopy
from typing import Dict, List, Optional

import mujoco
import numpy as np

from robosuite.models.base import MujocoXMLModel
from robosuite.models.bases import LegBaseModel, MobileBaseModel, MountModel, NullBaseModel, RobotBaseModel
from robosuite.utils.mjcf_utils import ROBOT_COLLISION_COLOR, array_to_string, find_elements, find_parent
from robosuite.utils.transform_utils import euler2mat, mat2quat

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

    def __init__(self, fname: str, idn=0):
        super().__init__(fname, idn=idn)

        # Define other variables that get filled later
        self.base = None

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

        # By default, set small frictionloss and armature values
        self.set_joint_attribute(attrib="frictionloss", values=0.1 * np.ones(self.dof), force=False)
        self.set_joint_attribute(attrib="damping", values=0.1 * np.ones(self.dof), force=False)
        self.set_joint_attribute(
            attrib="armature", values=np.array([5.0 / (i + 1) for i in range(self.dof)]), force=False
        )
        self.mujoco_model: Optional[mujoco.MjModel] = None

    def set_mujoco_model(self, mujoco_model: Optional[mujoco.MjModel] = None):
        if mujoco_model is not None:
            self.mujoco_model = mujoco_model
        else:
            self.mujoco_model = self.get_model()

    def set_base_xpos(self, pos: np.ndarray):
        """
        Places the robot on position @pos.

        Args:
            pos (3-array): (x,y,z) position to place robot base
        """
        self._elements["root_body"].set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_ori(self, rot: np.ndarray):
        """
        Rotates robot by rotation @rot from its original orientation.

        Args:
            rot (3-array): (r,p,y) euler angles specifying the orientation for the robot base
        """
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        rot = mat2quat(euler2mat(rot))[[3, 0, 1, 2]]
        self._elements["root_body"].set("quat", array_to_string(rot))

    def set_joint_attribute(self, attrib: str, values: np.ndarray, force=True):
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
        assert values.size == len(self._elements["joints"]), (
            "Error setting joint attributes: "
            + "Values must be same size as joint dimension. Got {}, expected {}!".format(values.size, self.dof)
        )
        for i, joint in enumerate(self._elements["joints"]):
            if force or joint.get(attrib, None) is None:
                joint.set(attrib, array_to_string(np.array([values[i]])))

    def add_base(self, base: RobotBaseModel):
        """
        Mounts a base to the robot. Bases are defined in robosuite.models.bases
        """
        if isinstance(base, MountModel):
            self.add_mount(base)
        elif isinstance(base, MobileBaseModel):
            self.add_mobile_base(base)
        elif isinstance(base, LegBaseModel):
            self.add_leg_base(base)
        elif isinstance(base, NullBaseModel):
            self.add_null_base(base)
        else:
            raise ValueError("Invalid base type to add to robot!")

    def add_mount(self, mount: MountModel):
        """
        Mounts @mount to arm.

        Throws error if robot already has a mount or if mount type i\s incorrect.

        Args:
            mount (MountModel): mount MJCF model

        Raises:
            ValueError: [mount already added]
        """
        if self.base is not None:
            raise ValueError("Mount already added for this robot!")

        # First adjust mount's base position
        offset = self.base_offset - mount.top_offset
        mount._elements["root_body"].set("pos", array_to_string(offset))

        self.merge(mount, merge_body=self.root_body)

        self.base = mount

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def add_mobile_base(self, mobile_base: MobileBaseModel):
        """
        Mounts @mobile_base to arm.

        Throws error if robot already has a mobile base or if mobile base type is incorrect.

        Args:
            mobile base (MobileBaseModel): mount MJCF model

        Raises:
            ValueError: [mobile base already added]
        """
        if self.base is not None:
            raise ValueError("Mobile base already added for this robot!")

        # First adjust mount's base position
        offset = self.base_offset - mobile_base.top_offset
        mobile_base._elements["root_body"].set("pos", array_to_string(offset))

        # Keep robot0_base as the root, but move all its content (geoms, inertial, arms) to mobile base support
        merge_body = self.root_body
        root = find_elements(root=self.worldbody, tags="body", attribs={"name": merge_body}, return_first=True)
        
        # Store all direct children of robot0_base (arms, geoms, inertial, etc.)
        all_root_children = list(root)  # Make a copy of all children
        
        # Append mobile base bodies to robot0_base (not to worldbody)
        for body in mobile_base.worldbody:
            root.append(body)

        # Find the mount's support body where everything should attach
        mount_support = find_elements(
            root=self.worldbody, tags="body", attribs={"name": mobile_base.correct_naming("support")}, return_first=True
        )
        # Move ALL content from robot0_base to the mobile base support (arms, geoms, inertial)
        for child in all_root_children:
            mount_support.append(deepcopy(child))
            root.remove(child)
        self.merge_assets(mobile_base)
        for one_actuator in mobile_base.actuator:
            self.actuator.append(one_actuator)
        for one_sensor in mobile_base.sensor:
            self.sensor.append(one_sensor)
        for one_tendon in mobile_base.tendon:
            self.tendon.append(one_tendon)
        for one_equality in mobile_base.equality:
            self.equality.append(one_equality)
        for one_contact in mobile_base.contact:
            self.contact.append(one_contact)

        self.base = mobile_base

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def add_leg_base(self, leg_base: LegBaseModel):
        """
        Mounts @mobile_base to arm.

        Throws error if robot already has a mobile base or if mobile base type is incorrect.

        Args:
            mobile base (MobileBaseModel): mount MJCF model

        Raises:
            ValueError: [mobile base already added]
        """
        if self.base is not None:
            raise ValueError("Mobile base already added for this robot!")

        # First adjust mount's base position
        offset = self.base_offset - leg_base.top_offset
        leg_base._elements["root_body"].set("pos", array_to_string(offset))

        # if the mount is mobile, the robot should be "merged" into the mount,
        # so that when the mount moves the robot moves along with it
        merge_body = self.root_body
        root = find_elements(root=self.worldbody, tags="body", return_first=True)
        for body in leg_base.worldbody:
            root.append(body)
        # import pdb; pdb.set_trace()

        arm_root = find_elements(root=self.worldbody, tags="body", return_first=False)[1]

        mount_support = find_elements(
            root=leg_base.worldbody,
            tags="body",
            attribs={"name": leg_base.correct_naming("support")},
            return_first=True,
        )

        free_joint = find_elements(root=leg_base.worldbody, tags="freejoint", return_first=True)
        if free_joint is not None:
            root.append(deepcopy(free_joint))
            free_joint_parent = find_parent(leg_base.worldbody, free_joint)
            free_joint_parent.remove(free_joint)
        mount_support.append(deepcopy(arm_root))
        root.remove(arm_root)
        self.merge_assets(leg_base)
        for one_actuator in leg_base.actuator:
            self.actuator.append(one_actuator)
        for one_sensor in leg_base.sensor:
            self.sensor.append(one_sensor)
        for one_tendon in leg_base.tendon:
            self.tendon.append(one_tendon)
        for one_equality in leg_base.equality:
            self.equality.append(one_equality)
        for one_contact in leg_base.contact:
            self.contact.append(one_contact)

        self.base = leg_base

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def add_null_base(self, base: NullBaseModel):
        """
        Do not add any base to the robot.
        """
        if self.base is not None:
            raise ValueError("Mobile base already added for this robot!")

        self.base = base
        for body in base.worldbody:
            ele = body.find("site")
            if ele is not None:
                self.worldbody.append(ele)

        self.cameras = self.get_element_names(self.worldbody, "camera")

    # -------------------------------------------------------------------------------------- #
    # Public Properties: In general, these are the name-adjusted versions from the private   #
    #                    attributes pulled from their respective raw xml files               #
    # -------------------------------------------------------------------------------------- #

    @property
    def naming_prefix(self) -> str:
        return "robot{}_".format(self.idn)

    @property
    def dof(self) -> int:
        """
        Defines the number of DOF of the robot

        Returns:
            int: robot DOF
        """
        return len(self._joints)

    @property
    def bottom_offset(self) -> np.ndarray:
        """
        Returns vector from model root body to model bottom.
        By default, this is equivalent to this robot's mount's (bottom_offset - top_offset) + this robot's base offset

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        return (
            (self.base.bottom_offset - self.base.top_offset) + self._base_offset
            if self.base is not None
            else self._base_offset
        )

    @property
    def horizontal_radius(self) -> float:
        """
        Returns maximum distance from model root body to any radial point of the model. This method takes into
        account the mount horizontal radius as well

        Returns:
            float: radius
        """
        return max(self._horizontal_radius, self.base.horizontal_radius)

    @property
    def models(self) -> List[MujocoXMLModel]:
        """
        Returns a list of all m(sub-)models owned by this robot model. By default, this includes the mount model,
        if specified

        Returns:
            list: models owned by this object
        """
        return [self.base] if self.base is not None else []

    @property
    def contact_geom_rgba(self):
        return ROBOT_COLLISION_COLOR

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def default_base(self) -> str:
        """
        Defines the default mount type for this robot that gets added to root body (base)

        Returns:
            str: Default base name to add to this robot
        """
        raise NotImplementedError

    @property
    def default_controller_config(self) -> Dict[str, str]:
        """
        Defines the name of default controller config file in the controllers/config directory for this robot.

        Returns:
            dict: Dictionary containing arm-specific default controller config names
                e.g.: {"right": "default_panda", "left": "default_panda"}
        """
        raise NotImplementedError

    @property
    def init_qpos(self) -> np.ndarray:
        """
        Defines the default rest qpos of this robot

        Returns:
            np.array: Default init qpos of this robot
        """
        raise NotImplementedError

    @property
    def base_xpos_offset(self) -> Dict[str, np.ndarray]:
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
    def top_offset(self) -> np.ndarray:
        """
        Returns vector from model root body to model top.
        Useful for, e.g. placing models on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def _horizontal_radius(self) -> float:
        """
        Returns maximum distance from model root body to any radial point of the model.

        Helps us put models programmatically without them flying away due to a huge initial contact force.
        Must be defined by subclass.

        Returns:
            float: radius
        """
        raise NotImplementedError

    @property
    def _important_sites(self) -> Dict[str, str]:
        """
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def _important_geoms(self) -> Dict[str, List[str]]:
        """
        Returns:
             dict: (Default is no important geoms; i.e.: empty dict)
        """
        return {}

    @property
    def _important_sensors(self) -> Dict[str, str]:
        """
        Returns:
            dict: (Default is no sensors; i.e.: empty dict)
        """
        return {}

    @property
    def all_joints(self) -> List:
        """
        Returns:
            list: (Default is no joints; i.e.: empty list)
        """
        all_joints = []
        all_joints += self.joints
        if self.base is not None:
            all_joints += self.base.joints
        return all_joints

    @property
    def all_actuators(self) -> List:
        """
        Returns:
            list: (Default is no actuators; i.e.: empty list)
        """
        all_actuators = []
        all_actuators += self.actuators
        if self.base is not None:
            all_actuators += self.base.actuators
        return all_actuators
