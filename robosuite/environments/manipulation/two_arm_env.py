import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.utils.robot_utils import check_bimanual
from robosuite.utils.transform_utils import mat2quat


class TwoArmEnv(ManipulationEnv):
    """
    A manipulation environment intended for two robot arms.
    """

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        super()._check_robot_configuration(robots)
        robots = robots if type(robots) == list or type(robots) == tuple else [robots]
        # If default config is used, set env_configuration accordingly
        if self.env_configuration == "default":
            self.env_configuration = "bimanual" if check_bimanual(robots[0]) else "single-arm-opposed"

        if self.env_configuration == "single-arm-opposed" or self.env_configuration == "single-arm-parallel":
            # Specifically two robots should be inputted!
            is_bimanual = False
            if type(robots) is not list or len(robots) != 2:
                raise ValueError(
                    "Error: Exactly two single-armed robots should be inputted " "for this task configuration!"
                )
        elif self.env_configuration == "bimanual":
            is_bimanual = True
            # Specifically one robot should be inputted!
            if type(robots) is list and len(robots) != 1:
                raise ValueError("Error: Exactly one bimanual robot should be inputted " "for this task configuration!")
        else:
            # This is an unknown env configuration, print error
            raise ValueError(
                "Error: Unknown environment configuration received. Only 'bimanual',"
                "'single-arm-parallel', and 'single-arm-opposed' are supported. Got: {}".format(self.env_configuration)
            )

        # Lastly, check to make sure all inputted robot names are of their correct type (bimanual / not bimanual)
        for robot in robots:
            if check_bimanual(robot) != is_bimanual:
                raise ValueError(
                    "Error: For {} configuration, expected bimanual check to return {}; "
                    "instead, got {}.".format(self.env_configuration, is_bimanual, check_bimanual(robot))
                )

    @property
    def _eef0_xpos(self):
        """
        Grab the position of Robot 0's end effector.

        Returns:
            np.array: (x,y,z) position of EEF0
        """
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])

    @property
    def _eef1_xpos(self):
        """
        Grab the position of Robot 1's end effector.

        Returns:
            np.array: (x,y,z) position of EEF1
        """
        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["left"]])
        else:
            return np.array(self.sim.data.site_xpos[self.robots[1].eef_site_id])

    @property
    def _eef0_xmat(self):
        """
        End Effector 0 orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (3,3) orientation matrix for EEF0
        """
        pf = self.robots[0].gripper.naming_prefix

        if self.env_configuration == "bimanual":
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "right_grip_site")]).reshape(3, 3)
        
        else:
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "grip_site")]).reshape(3, 3)

    @property
    def _eef1_xmat(self):
        """
        End Effector 1 orientation as a rotation matrix
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (3,3) orientation matrix for EEF1
        """
        if self.env_configuration == "bimanual":
            pf = self.robots[0].gripper.naming_prefix
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "left_grip_site")]).reshape(3, 3)
        else:
            pf = self.robots[1].gripper.naming_prefix
            return np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(pf + "grip_site")]).reshape(3, 3)

    @property
    def _eef0_xquat(self):
        """
        End Effector 0 orientation as a (x,y,z,w) quaternion
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (x,y,z,w) quaternion for EEF0
        """
        return mat2quat(self._eef0_xmat)

    @property
    def _eef1_xquat(self):
        """
        End Effector 1 orientation as a (x,y,z,w) quaternion
        Note that this draws the orientation from the "ee" site, NOT the gripper site, since the gripper
        orientations are inconsistent!

        Returns:
            np.array: (x,y,z,w) quaternion for EEF1
        """
        return mat2quat(self._eef1_xmat)