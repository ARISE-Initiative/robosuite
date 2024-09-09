import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.robot_utils import check_bimanual
from robosuite.utils.transform_utils import mat2quat


class TwoArmEnv(ManipulationEnv):
    """
    A manipulation environment intended for two robot arms.
    """

    # Dictionary which maps the old configuration names to the new configuration names.
    # This is used to maintain backwards compatibility with past environment configurations.
    _PREV_CONFIG_TO_NEW_CONFIG = {
        "single-arm-opposed": "opposed",
        "single-arm-parallel": "parallel",
    }

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        super()._check_robot_configuration(robots)
        if self.env_configuration in self._PREV_CONFIG_TO_NEW_CONFIG:
            self.env_configuration = self._PREV_CONFIG_TO_NEW_CONFIG[self.env_configuration]
        robots = robots if type(robots) == list or type(robots) == tuple else [robots]

        # Process single robot and 2 robot cases separately
        if len(robots) == 1:
            if not check_bimanual(robots[0]):
                raise ValueError("Error: If inputting a single robot, it must be bimanual.")
            # Automatically set the configuration to single-robot if one robot is given
            self.env_configuration = "single-robot"
        elif len(robots) == 2:
            if self.env_configuration == "default":
                self.env_configuration = "opposed"

            # Check that the configuration is valid
            if not (self.env_configuration == "opposed" or self.env_configuration == "parallel"):
                # This is an unknown env configuration, print error
                raise ValueError(
                    "Error: Unknown environment configuration received. Only 'opposed', 'parallel', "
                    "are supported. Got: {}".format(self.env_configuration)
                )
        else:
            raise ValueError(
                "Error: Invalid number of robots received. Expected either 1 bimanual robot or 2 of any type of robots."
            )

    def _gripper0_to_target(self, target, target_type="body", return_distance=False):
        """
        Returns the distance/distance vector between gripper0 and target. If there is only one robot,
        gripper0 is the right gripper of the first robot. Otherwise, it is the gripper(s) of the
        first robot and the returned distance is the minimum distance between the gripper(s) and the target.

        Args:
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
            return_distance (bool): If set, will return Euclidean distance instead of Cartesian distance

        Returns:
            np.array or float: (Cartesian or Euclidean) distance from gripper to target
        """

        if self.env_configuration == "single-robot":
            return self._gripper_to_target(self.robots[0].gripper["right"], target, target_type, return_distance)
        else:
            return self._gripper_to_target(self.robots[0].gripper, target, target_type, return_distance)

    def _gripper1_to_target(self, target, target_type="body", return_distance=False):
        """
        Returns the distance/distance vector between gripper1 and target. If there is only one robot,
        gripper1 is the left gripper of the first robot. Otherwise, it is the gripper(s) of the
        second robot and the returned distance is the minimum distance between the gripper(s) and the target.

        Args:
            target (MujocoModel or str): Either a site / geom / body name, or a model that serves as the target.
                If a model is given, then the root body will be used as the target.
            target_type (str): One of {"body", "geom", or "site"}, corresponding to the type of element @target
                refers to.
            return_distance (bool): If set, will return Euclidean distance instead of Cartesian distance

        Returns:
            np.array or float: (Cartesian or Euclidean) distance from gripper to target
        """

        if self.env_configuration == "single-robot":
            return self._gripper_to_target(self.robots[0].gripper["left"], target, target_type, return_distance)
        else:
            return self._gripper_to_target(self.robots[1].gripper, target, target_type, return_distance)

    @property
    def _eef0_xpos(self):
        """
        Grab the position of Robot 0's right end effector.

        Returns:
            np.array: (x,y,z) position of EEF0
        """

        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])

    @property
    def _eef1_xpos(self):
        """
        Grab the position of Robot 1's end effector.

        Returns:
            np.array: (x,y,z) position of EEF1
        """
        if self.env_configuration == "single-robot":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["left"]])
        else:
            # right is the default first arm
            return np.array(self.sim.data.site_xpos[self.robots[1].eef_site_id["right"]])

    @property
    def _eef0_xmat(self):
        """
        Right End Effector 0 orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (3,3) orientation matrix for EEF0
        """
        pf = self.robots[0].gripper["right"].naming_prefix
        return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "grip_site")]).reshape(3, 3)

    @property
    def _eef0_xquat(self):
        """
        Right End Effector 0 orientation as a (x,y,z,w) quaternion
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (x,y,z,w) quaternion for EEF0
        """
        return mat2quat(self._eef0_xmat)
