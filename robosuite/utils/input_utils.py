"""
Utility functions for grabbing user inputs
"""

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.devices import *
from robosuite.models.robots import *
from robosuite.robots import *


def choose_environment():
    """
    Prints out environment options, and returns the selected env_name choice

    Returns:
        str: Chosen environment name
    """
    # get the list of all environments
    envs = sorted(suite.ALL_ENVIRONMENTS)

    # Select environment to run
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    print()
    try:
        s = input("Choose an environment to run " + "(enter a number from 0 to {}): ".format(len(envs) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs))
    except:
        k = 0
        print("Input is not valid. Use {} by default.\n".format(envs[k]))

    # Return the chosen environment name
    return envs[k]


def choose_controller():
    """
    Prints out controller options, and returns the requested controller name

    Returns:
        str: Chosen controller name
    """
    # get the list of all controllers
    controllers = list(suite.ALL_COMPOSITE_CONTROLLERS)

    # Select controller to use
    print("Here is a list of controllers in the suite:\n")

    for k, controller in enumerate(controllers):
        print("[{}] {}".format(k, controller))
    print()
    try:
        s = input("Choose a controller for the robot " + "(enter a number from 0 to {}): ".format(len(controllers) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(controllers) - 1)
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(controllers)[k])

    # Return chosen controller
    return controllers[k]


def choose_part_controller():
    """
    Prints out part controller options, and returns the requested part controller name

    Returns:
        str: Chosen part controller name
    """
    controllers = list(suite.ALL_PART_CONTROLLERS)

    # Select controller to use
    print("Here is a list of part controllers in the suite:\n")

    for k, controller in enumerate(controllers):
        print("[{}] {}".format(k, controller))
    print()
    try:
        s = input(
            "Choose a part controller for the robot " + "(enter a number from 0 to {}): ".format(len(controllers) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(controllers) - 1)
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(controllers)[k])

    # Return chosen controller
    return controllers[k]


def choose_multi_arm_config():
    """
    Prints out multi-arm environment configuration options, and returns the requested config name

    Returns:
        str: Requested multi-arm configuration name
    """
    # Get the list of all multi arm configs
    env_configs = {
        "Opposed": "opposed",
        "Parallel": "parallel",
    }

    # Select environment configuration
    print("A multi-arm environment was chosen. Here is a list of multi-arm environment configurations:\n")

    for k, env_config in enumerate(list(env_configs)):
        print("[{}] {}".format(k, env_config))
    print()
    try:
        s = input(
            "Choose a configuration for this environment "
            + "(enter a number from 0 to {}): ".format(len(env_configs) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(env_configs))
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(list(env_configs)[k]))

    # Return requested configuration
    return list(env_configs.values())[k]


def choose_robots(exclude_bimanual=False, use_humanoids=False):
    """
    Prints out robot options, and returns the requested robot. Restricts options to single-armed robots if
    @exclude_bimanual is set to True (False by default). Restrict options to humanoids if @use_humanoids is set to True (Flase by default).

    Args:
        exclude_bimanual (bool): If set, excludes bimanual robots from the robot options
        use_humanoids (bool): If set, use humanoid robots

    Returns:
        str: Requested robot name
    """
    # Get the list of robots
    robots = {"Sawyer", "Panda", "Jaco", "Kinova3", "IIWA", "UR5e"}

    # Add Baxter if bimanual robots are not excluded
    if not exclude_bimanual:
        robots.add("Baxter")
        robots.add("GR1")
        robots.add("GR1UpperBody")
    if use_humanoids:
        robots = {"GR1", "GR1UpperBody"}

    # Make sure set is deterministically sorted
    robots = sorted(robots)

    # Select robot
    print("Here is a list of available robots:\n")

    for k, robot in enumerate(robots):
        print("[{}] {}".format(k, robot))
    print()
    try:
        s = input("Choose a robot " + "(enter a number from 0 to {}): ".format(len(robots) - 1))
        # parse input into a number within range
        k = min(max(int(s), 0), len(robots))
    except:
        k = 0
        print("Input is not valid. Use {} by default.".format(list(robots)[k]))

    # Return requested robot
    return list(robots)[k]
