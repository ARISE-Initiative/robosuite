from collections import OrderedDict

from robosuite.models.base import MujocoXML
from robosuite.utils import XMLError


class Robot(MujocoXML):
    """Base class for all robot models."""

    def __init__(self, fname):
        """Initializes from file @fname."""
        super().__init__(fname)
        # key: gripper name and value: gripper model
        self.grippers = OrderedDict()

    def add_gripper(self, arm_name, gripper):
        """
        Mounts gripper to arm.

        Throws error if robot already has a gripper or gripper type is incorrect.

        Args:
            arm_name (str): name of arm mount
            gripper (MujocoGripper instance): gripper MJCF model
        """
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

    @property
    def dof(self):
        """Returns the number of DOF of the robot, not including gripper."""
        raise NotImplementedError

    @property
    def joints(self):
        """Returns a list of joint names of the robot."""
        raise NotImplementedError

    @property
    def init_qpos(self):
        """Returns default qpos."""
        raise NotImplementedError
