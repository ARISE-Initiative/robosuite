# Utilities functions for working with robots

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS


def check_bimanual(robot_name):
    """
    Utility function that returns whether the inputted robot_name is a bimanual robot or not

    Args:
        robot_name (str): Name of the robot to check

    Returns:
        bool: True if the inputted robot is a bimanual robot
    """
    robot = REGISTERED_ROBOTS[robot_name]()
    return robot.arm_type == "bimanual"