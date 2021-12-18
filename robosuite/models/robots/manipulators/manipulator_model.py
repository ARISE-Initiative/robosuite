from collections import OrderedDict

import numpy as np

from robosuite.models.robots import RobotModel
from robosuite.utils.mjcf_utils import find_elements, string_to_array


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

        # Grab hand's offset from final robot link (string -> np.array -> elements [1, 2, 3, 0] (x, y, z, w))
        # Different case based on whether we're dealing with single or bimanual armed robot
        if self.arm_type == "single":
            hand_element = find_elements(
                root=self.root, tags="body", attribs={"name": self.eef_name}, return_first=True
            )
            self.hand_rotation_offset = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]
        else:  # "bimanual" case
            self.hand_rotation_offset = {}
            for arm in ("right", "left"):
                hand_element = find_elements(
                    root=self.root, tags="body", attribs={"name": self.eef_name[arm]}, return_first=True
                )
                self.hand_rotation_offset[arm] = string_to_array(hand_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

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
        return "right_hand"

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
    def default_mount(self):
        raise NotImplementedError

    @property
    def default_controller_config(self):
        raise NotImplementedError

    @property
    def init_qpos(self):
        raise NotImplementedError
