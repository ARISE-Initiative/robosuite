import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class Panda(Robot):
    """Panda is a sensitive single-arm robot designed by Franka."""

    def __init__(self):
        super().__init__(xml_path_completion("robots/panda/robot.xml"))

        self.bottom_offset = np.array([0, 0, -0.913])
        self.set_joint_damping()
        self._model_name = "panda"
        # Careful of init_qpos -- certain init poses cause ik controller to go unstable (e.g: pi/4 instead of -pi/4
        # for the final joint angle)
        self._init_qpos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi/4])

    def set_base_xpos(self, pos):
        """Places the robot on position @pos."""
        node = self.worldbody.find("./body[@name='{}']".format(self._base_name))
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_joint_damping(self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint damping """
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("damping", array_to_string(np.array([damping[i]])))

    def set_joint_frictionloss(self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))):
        """Set joint friction loss (static friction)"""
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("frictionloss", array_to_string(np.array([friction[i]])))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["joint{}".format(x) for x in range(1, 8)]

    @property
    def init_qpos(self):
        return self._init_qpos

    @property
    def contact_geoms(self):
        return ["link{}_collision".format(x) for x in range(1, 8)]

    @property
    def _base_body(self):
        node = self.worldbody.find("./body[@name='{}']".format(self._base_name))
        return node

    @property
    def _base_name(self):
        # Returns the base name of the mujoco xml body
        return 'link0'

    @property
    def _link_body(self):
        return ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]

    @property
    def _joints(self):
        return ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

