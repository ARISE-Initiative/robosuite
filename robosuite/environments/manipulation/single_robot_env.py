from robosuite.environments.manipulation.manipulation_env import ManipulationEnv


class SingleRobotEnv(ManipulationEnv):
    """
    A manipulation environment intended for a single robot
    This class is a modified version of the original single_robot_env.py file in the robosuite package.
    """

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable.
        For this class, we only allow for a single robot to be inputted.

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        super()._check_robot_configuration(robots)
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"
