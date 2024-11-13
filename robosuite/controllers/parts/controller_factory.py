"""
Set of functions that streamline controller initialization process
"""
import json
import os
from copy import deepcopy

import numpy as np

from robosuite.controllers.parts import arm as arm_controllers
from robosuite.controllers.parts import generic
from robosuite.controllers.parts import gripper as gripper_controllers
from robosuite.controllers.parts import mobile_base as mobile_base_controllers
from robosuite.utils.traj_utils import LinearInterpolator

# from . import legs as legs_controllers


def load_part_controller_config(custom_fpath=None, default_controller=None):
    """
    Utility function that loads the desired controller and returns the loaded configuration as a dict

    If @default_controller is specified, any value inputted to @custom_fpath is overridden and the default controller
    configuration is automatically loaded. See specific arg description below for available default controllers.

    Args:
        custom_fpath (str): Absolute filepath to the custom controller configuration .json file to be loaded
        default_controller (str): If specified, overrides @custom_fpath and loads a default configuration file for the
            specified controller.
            Choices are: {"JOINT_POSITION", "JOINT_TORQUE", "JOINT_VELOCITY", "OSC_POSITION", "OSC_POSE", "IK_POSE"}

    Returns:
        dict: Controller configuration

    Raises:
        AssertionError: [Unknown default controller name]
        AssertionError: [No controller specified]
    """
    # TODO (YL): this is nolonger usable, need to update
    # First check if default controller is not None; if it
    # is not, load the appropriate controller
    if default_controller is not None:

        # Assert that requested default controller is in the available default controllers
        from robosuite.controllers import ALL_PART_CONTROLLERS

        assert (
            default_controller in ALL_PART_CONTROLLERS
        ), "Error: Unknown default controller specified. Requested {}, " "available controllers: {}".format(
            default_controller, list(ALL_PART_CONTROLLERS)
        )

        # Store the default controller config fpath associated with the requested controller
        custom_fpath = os.path.join(
            os.path.dirname(__file__), "..", "config/default/parts/{}.json".format(default_controller.lower())
        )

    # Assert that the fpath to load the controller is not empty
    assert custom_fpath is not None, "Error: Either custom_fpath or default_controller must be specified!"

    # Attempt to load the controller
    try:
        with open(custom_fpath) as f:
            controller_config = json.load(f)
    except FileNotFoundError:
        print("Error opening controller filepath at: {}. " "Please check filepath and try again.".format(custom_fpath))
        raise FileNotFoundError

    # Return the loaded controller
    return controller_config


def arm_controller_factory(name, params):
    """
    Generator for controllers

    Creates a Controller instance with the provided @name and relevant @params.

    Args:
        name (str): the name of the controller. Must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
            OSC_POSITION, OSC_POSE, IK_POSE}
        params (dict): dict containing the relevant params to pass to the controller
        sim (MjSim): Mujoco sim reference to pass to the controller

    Returns:
        Controller: Controller instance

    Raises:
        ValueError: [unknown controller]
    """

    interpolator = None
    if params["interpolation"] == "linear":
        interpolator = LinearInterpolator(
            ndim=params["ndim"],
            controller_freq=(1 / params["sim"].model.opt.timestep),
            policy_freq=params["policy_freq"],
            ramp_ratio=params["ramp_ratio"],
        )

    if name == "OSC_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos and ori each
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.set_states(ori="euler")
        params["control_ori"] = True
        return arm_controllers.OperationalSpaceController(
            interpolator_pos=interpolator, interpolator_ori=ori_interpolator, **params
        )

    if name == "OSC_POSITION":
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE control uses dim 3 for pos
        params["control_ori"] = False
        return arm_controllers.OperationalSpaceController(interpolator_pos=interpolator, **params)

    if name == "IK_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.set_states(dim=3)  # EE IK control uses dim 3 for pos and dim 4 for ori
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.set_states(dim=4, ori="quat")

        from robosuite.controllers.parts.arm.ik import InverseKinematicsController

        return InverseKinematicsController(
            interpolator_pos=interpolator,
            interpolator_ori=ori_interpolator,
            **params,
        )

    if name == "JOINT_VELOCITY":
        return generic.JointVelocityController(interpolator=interpolator, **params)

    if name == "JOINT_POSITION":
        return generic.JointPositionController(interpolator=interpolator, **params)

    if name == "JOINT_TORQUE":
        return generic.JointTorqueController(interpolator=interpolator, **params)

    raise ValueError("Unknown controller name: {}".format(name))


def controller_factory(part_name, controller_type, controller_params):
    if part_name in ["right", "left"]:
        return arm_controller_factory(controller_type, controller_params)
    elif part_name in ["right_gripper", "left_gripper"]:
        return gripper_controller_factory(controller_type, controller_params)
    elif part_name == "base":
        return mobile_base_controller_factory(controller_type, controller_params)
    elif part_name == "torso":
        return torso_controller_factory(controller_type, controller_params)
    elif part_name == "head":
        return head_controller_factory(controller_type, controller_params)
    elif part_name == "legs":
        return legs_controller_factory(controller_type, controller_params)
    else:
        raise ValueError("Unknown controller part name: {}".format(part_name))


def gripper_controller_factory(name, params):
    interpolator = None
    if name == "GRIP":
        return gripper_controllers.SimpleGripController(interpolator=interpolator, **params)
    elif name == "JOINT_POSITION":
        return generic.JointPositionController(interpolator=interpolator, **params)
    raise ValueError("Unknown controller name: {}".format(name))


def mobile_base_controller_factory(name, params):
    interpolator = None
    if name == "JOINT_VELOCITY":
        return mobile_base_controllers.MobileBaseJointVelocityController(interpolator=interpolator, **params)
    elif name == "JOINT_POSITION":
        raise NotImplementedError
    raise ValueError("Unknown controller name: {}".format(name))


def torso_controller_factory(name, params):
    interpolator = None
    if params["interpolation"] == "linear":
        interpolator = LinearInterpolator(
            ndim=params["ndim"],
            controller_freq=(1 / params["sim"].model.opt.timestep),
            policy_freq=params["policy_freq"],
            ramp_ratio=params["ramp_ratio"],
        )

    if name == "JOINT_VELOCITY":
        return generic.JointVelocityController(interpolator=interpolator, **params)
    elif name == "JOINT_POSITION":
        return generic.JointPositionController(interpolator=interpolator, **params)
    raise ValueError("Unknown controller name: {}".format(name))


def head_controller_factory(name, params):
    interpolator = None
    if params["interpolation"] == "linear":
        interpolator = LinearInterpolator(
            ndim=params["ndim"],
            controller_freq=(1 / params["sim"].model.opt.timestep),
            policy_freq=params["policy_freq"],
            ramp_ratio=params["ramp_ratio"],
        )

    if name == "JOINT_VELOCITY":
        return generic.JointVelocityController(interpolator=interpolator, **params)
    elif name == "JOINT_POSITION":
        return generic.JointPositionController(interpolator=interpolator, **params)
    raise ValueError("Unknown controller name: {}".format(name))


def legs_controller_factory(name, params):
    interpolator = None
    if params["interpolation"] == "linear":
        interpolator = LinearInterpolator(
            ndim=params["ndim"],
            controller_freq=(1 / params["sim"].model.opt.timestep),
            policy_freq=params["policy_freq"],
            ramp_ratio=params["ramp_ratio"],
        )

    if name == "JOINT_POSITION":
        return generic.JointPositionController(interpolator=interpolator, **params)

    if name == "JOINT_TORQUE":
        return generic.JointTorqueController(interpolator=interpolator, **params)

    raise ValueError("Unknown controller name: {}".format(name))
