# Utilities functions for working with robots

from robosuite.robots.legacy import BIMANUAL_ROBOTS


def check_bimanual(robot_name):
    """
    Utility function that returns whether the inputted robot_name is a bimanual robot or not

    Args:
        robot_name (str): Name of the robot to check

    Returns:
        bool: True if the inputted robot is a bimanual robot
    """
    return robot_name.lower() in BIMANUAL_ROBOTS
