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
    controllers_info = suite.controllers.CONTROLLER_INFO
    controllers = list(suite.ALL_CONTROLLERS)

    # Select controller to use
    print("Here is a list of controllers in the suite:\n")

    for k, controller in enumerate(controllers):
        print("[{}] {} - {}".format(k, controller, controllers_info[controller]))
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


def choose_multi_arm_config():
    """
    Prints out multi-arm environment configuration options, and returns the requested config name

    Returns:
        str: Requested multi-arm configuration name
    """
    # Get the list of all multi arm configs
    env_configs = {
        "Single Arms Opposed": "single-arm-opposed",
        "Single Arms Parallel": "single-arm-parallel",
        "Bimanual": "bimanual",
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


def input2action(device, robot, active_arm="right", env_configuration=None):
    """
    Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

    If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action

    Args:
        device (Device): A device from which user inputs can be converted into actions. Can be either a Spacemouse or
            Keyboard device class

        robot (Robot): Which robot we're controlling

        active_arm (str): Only applicable for multi-armed setups (e.g.: multi-arm environments or bimanual robots).
            Allows inputs to be converted correctly if the control type (e.g.: IK) is dependent on arm choice.
            Choices are {right, left}

        env_configuration (str or None): Only applicable for multi-armed environments. Allows inputs to be converted
            correctly if the control type (e.g.: IK) is dependent on the environment setup. Options are:
            {bimanual, single-arm-parallel, single-arm-opposed}

    Returns:
        2-tuple:

            - (None or np.array): Action interpreted from @device including any gripper action(s). None if we get a
                reset signal from the device
            - (None or int): 1 if desired close, -1 if desired open gripper state. None if get a reset signal from the
                device

    """
    state = device.get_controller_state()
    # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
    #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
    #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
    dpos, rotation, raw_drotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["raw_drotation"],
        state["grasp"],
        state["reset"],
    )

    # If we're resetting, immediately return None
    if reset:
        return None, None

    # Get controller reference
    controller = robot.controller[active_arm]
    gripper_dof = robot.gripper[active_arm].dof

    # First process the raw drotation
    drotation = raw_drotation[[1, 0, 2]]
    if controller.name == "IK_POSE":
        # If this is panda, want to swap x and y axis
        if isinstance(robot.robot_model, Panda):
            drotation = drotation[[1, 0, 2]]
        else:
            # Flip x
            drotation[0] = -drotation[0]
        # Scale rotation for teleoperation (tuned for IK)
        drotation *= 10
        dpos *= 5
        # relative rotation of desired from current eef orientation
        # map to quat
        drotation = T.mat2quat(T.euler2mat(drotation))

        # If we're using a non-forward facing configuration, need to adjust relative position / orientation
        if env_configuration == "single-arm-opposed":
            # Swap x and y for pos and flip x,y signs for ori
            dpos = dpos[[1, 0, 2]]
            drotation[0] = -drotation[0]
            drotation[1] = -drotation[1]
            if active_arm == "left":
                # x pos needs to be flipped
                dpos[0] = -dpos[0]
            else:
                # y pos needs to be flipped
                dpos[1] = -dpos[1]

        # Lastly, map to axis angle form
        drotation = T.quat2axisangle(drotation)

    elif controller.name == "OSC_POSE":
        # Flip z
        drotation[2] = -drotation[2]
        # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
        drotation = drotation * 1.5 if isinstance(device, Keyboard) else drotation * 50
        dpos = dpos * 75 if isinstance(device, Keyboard) else dpos * 125
    elif controller.name == "OSC_POSITION":
        dpos = dpos * 75 if isinstance(device, Keyboard) else dpos * 125
    else:
        # No other controllers currently supported
        print("Error: Unsupported controller specified -- Robot must have either an IK or OSC-based controller!")

    # map 0 to -1 (open) and map 1 to 1 (closed)
    grasp = 1 if grasp else -1

    if robot.is_mobile:
        assert controller.name == "OSC_POSE", "Mobile robots only currently supported by OSC_POSE controller"
        base_mode = bool(state["base_mode"])
        if base_mode is True:
            arm_ac = np.zeros(6)
            base_ac = np.array([dpos[0], dpos[1], drotation[2], dpos[2]])
            mode_ac = np.array([1])
        else:
            arm_ac = np.concatenate([dpos, drotation])
            base_ac = np.zeros(4)
            mode_ac = np.array([-1])
        gripper_ac = np.array([grasp] * gripper_dof)
        action = np.concatenate((arm_ac, gripper_ac, base_ac, mode_ac))
        return action, {"grasp": grasp, "mode": base_mode}
    else:
        # Create action based on action space of individual robot
        if controller.name == "OSC_POSITION":
            action = np.concatenate([dpos, [grasp] * gripper_dof])
        else:
            action = np.concatenate([dpos, drotation, [grasp] * gripper_dof])

        # Return the action and grasp
        return action, grasp
