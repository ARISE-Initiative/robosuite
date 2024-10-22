from collections import OrderedDict

import numpy as np

from robosuite.models.robots import RobotModel
from robosuite.utils.mjcf_utils import find_elements, find_elements_by_substring, string_to_array


class ManipulatorModel(RobotModel):
    """
    Base class for all manipulator models (robot arm(s) with gripper(s)).

    Args:
        fname (str): Path to relevant xml file from which to create this robot instance
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, fname, idn=0):
        # Always run super init first
        super().__init__(fname, idn=idn)

        # key: gripper name and value: gripper model
        self.grippers = OrderedDict()

        # # Grab hand's offset from final robot link (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
        # # Different case based on whether we're dealing with single or bimanual armed robot
        # if self.arm_type == "single":
        #     hand_element = find_elements(
        #         root=self.root, tags="body", attribs={"name": self.eef_name}, return_first=True
        #     )
        #     self.hand_rotation_offset = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]
        # else:  # "bimanual" case
        if self.arm_type == "single":
            arms = ["right"]
        elif self.arm_type == "bimanual":
            arms = ["right", "left"]
        self.hand_rotation_offset = {}
        for arm in arms:
            hand_element = find_elements(
                root=self.root, tags="body", attribs={"name": self.eef_name[arm]}, return_first=True
            )
            self.hand_rotation_offset[arm] = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

        self._base_actuators = []
        self._torso_actuators = []
        self._head_actuators = []
        self._legs_actuators = []

        self._arms_actuators = []
        # specify arm actuators by excluding overallpy actuators from previous ones

        self._base_joints = []
        self._torso_joints = []
        self._head_joints = []
        self._legs_joints = []

        self._arms_joints = []

    def add_gripper(self, gripper, arm_name=None):
        """
        Mounts @gripper to arm.

        Throws error if robot already has a gripper or gripper type is incorrect.

        Args:
            gripper (GripperModel): gripper MJCF model
            arm_name (str): name of arm mount -- defaults to self.eef_name if not specified

        Raises:
            ValueError: [Multiple grippers]
        """
        if arm_name is None:
            arm_name = self.eef_name
        if arm_name in self.grippers:
            raise ValueError("Attempts to add multiple grippers to one body")
        self.merge(gripper, merge_body=arm_name)

        self.grippers[arm_name] = gripper

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    def update_joints(self):
        for joint in self.all_joints:
            if "mobile" in joint:
                self.base_joints.append(joint)
            elif "torso" in joint:
                self.torso_joints.append(joint)
            elif "head" in joint:
                self.head_joints.append(joint)
            elif "leg" in joint:
                self.legs_joints.append(joint)

        for joint in self.all_joints:
            if (
                joint not in self._base_joints
                and joint not in self._torso_joints
                and joint not in self._head_joints
                and joint not in self._legs_joints
            ):
                self._arms_joints.append(joint)

    def update_actuators(self):
        for actuator in self.all_actuators:
            if "mobile" in actuator:
                self.base_actuators.append(actuator)
            elif "torso" in actuator:
                self.torso_actuators.append(actuator)
            elif "head" in actuator:
                self.head_actuators.append(actuator)
            elif "leg" in actuator:
                self.legs_actuators.append(actuator)

        for actuator in self.all_actuators:
            if (
                actuator not in self._base_actuators
                and actuator not in self._torso_actuators
                and actuator not in self._head_actuators
                and actuator not in self._legs_actuators
            ):
                self._arms_actuators.append(actuator)

    # -------------------------------------------------------------------------------------- #
    # Public Properties: In general, these are the name-adjusted versions from the private   #
    #                    attributes pulled from their respective raw xml files               #
    # -------------------------------------------------------------------------------------- #

    @property
    def eef_name(self):
        """
        Returns:
            str or dict of str: Prefix-adjusted eef name for this robot. If bimanual robot, returns {"left", "right"}
                keyword-mapped eef names
        """
        return self.correct_naming(self._eef_name)

    @property
    def models(self):
        """
        Returns a list of all m(sub-)models owned by this robot model. By default, this includes the gripper model,
        if specified

        Returns:
            list: models owned by this object
        """
        models = super().models
        return models + list(self.grippers.values())

    # -------------------------------------------------------------------------------------- #
    # -------------------------- Private Properties ---------------------------------------- #
    # -------------------------------------------------------------------------------------- #

    @property
    def _important_sites(self):
        """
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def _eef_name(self):
        """
        XML eef name for this robot to which grippers can be attached. Note that these should be the raw
        string names directly pulled from a robot's corresponding XML file, NOT the adjusted name with an
        auto-generated naming prefix

        Returns:
            str: Raw XML eef name for this robot (default is "right_hand")
        """
        return {"right": "right_hand"}

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def default_gripper(self):
        """
        Defines the default gripper type for this robot that gets added to end effector

        Returns:
            str: Default gripper name to add to this robot
        """
        raise NotImplementedError
    
    @property
    def gripper_mount_pos_offset(self):
        """
        Define the custom offset of the gripper that is different from the one defined in xml.
        The offset will applied to the first body in the gripper definition file.

        Returns:
            Empty dictionary unless specified.
        """
        return {}

    @property
    def gripper_mount_quat_offset(self):
        """
        Define the custom orientation offset of the gripper with respect to the arm. 
        The offset will applied to the first body in the gripper definition file.
        Return empty dict by default unless specified. 
        The quaternion is in the (w, x, y, z) format to match the mjcf format.
        """
        return {}

    @property
    def arm_type(self):
        """
        Type of robot arm. Should be either "bimanual" or "single" (or something else if it gets added in the future)

        Returns:
            str: Type of robot
        """
        raise NotImplementedError

    @property
    def base_xpos_offset(self):
        """
        Defines the dict of various (x,y,z) tuple offsets relative to specific arenas placed at (0,0,0)
        Assumes robot is facing forwards (in the +x direction) when determining offset. Should have entries for each
        manipulator arena case; i.e.: "bins", "empty", and "table")

        Returns:
            dict:

                :`'bins'`: (x,y,z) robot offset if placed in bins arena
                :`'empty'`: (x,y,z) robot offset if placed in the empty arena
                :`'table'`: lambda function that takes in table_length and returns corresponding (x,y,z) offset
                    if placed in the table arena
        """
        raise NotImplementedError

    @property
    def top_offset(self):
        raise NotImplementedError

    @property
    def _horizontal_radius(self):
        raise NotImplementedError

    @property
    def default_base(self):
        raise NotImplementedError

    @property
    def default_controller_config(self):
        raise NotImplementedError

    @property
    def init_qpos(self):
        raise NotImplementedError

    @property
    def arm_actuators(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no actuators; i.e.: empty dict)
        """
        return self._arms_actuators

    @property
    def base_actuators(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no actuators; i.e.: empty dict)
        """
        return self._base_actuators

    @property
    def torso_actuators(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no actuators; i.e.: empty dict)
        """
        return self._torso_actuators

    @property
    def head_actuators(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no actuators; i.e.: empty dict)
        """
        return self._head_actuators

    @property
    def legs_actuators(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no actuators; i.e.: empty dict)
        """
        return self._legs_actuators

    @property
    def arm_joints(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no joints; i.e.: empty dict)
        """
        return self._arms_joints

    @property
    def base_joints(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no joints; i.e.: empty dict)
        """
        return self._base_joints

    @property
    def torso_joints(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no joints; i.e.: empty dict)
        """
        return self._torso_joints

    @property
    def head_joints(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no joints; i.e.: empty dict)
        """
        return self._head_joints

    @property
    def legs_joints(self):
        """
        No need for name correcting because the prefix has been added during creation.

        Returns:
            list: (Default is no joints; i.e.: empty dict)
        """
        return self._legs_joints
