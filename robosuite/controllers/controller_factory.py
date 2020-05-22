"""
Defines a string based method of initializing controllers
"""
from .osc import OperationalSpaceController
from .joint_vel import JointVelocityController
from .joint_pos import JointPositionController
from .joint_tor import JointTorqueController
from .interpolators.linear_interpolator import LinearInterpolator

import json
import os

from copy import deepcopy

# Global var for linking pybullet server to multiple ik controller instances if necessary
pybullet_server = None


def reset_controllers():
    """Global function for doing one-time clears and restarting of any global controller-related
    specifics before re-initializing each individual controller again.
    """
    global pybullet_server
    # Disconnect and reconnect to pybullet server if it exists
    if pybullet_server:
        pybullet_server.disconnect()
        pybullet_server.connect()


def get_pybullet_server():
    """Getter to return reference to pybullet server module variable"""
    global pybullet_server
    return pybullet_server


def load_controller_config(custom_fpath=None, default_controller=None):
    """
    Utility function that loads the desired controller and returns the loaded configuration as a dict

    If @default_controller is specified, any value inputted to @custom_fpath is overridden and the default controller
        configuration is automatically loaded. See specific arg description below for available default controllers.

    Args:
        @custom_fpath (str): Absolute filepath to the custom controller configuration .json file to be loaded
        @default_controller (str): If specified, overrides @custom_fpath and loads a default configuration file for the
            specified controller.
            Choices are: {"JOINT_POSITION", "JOINT_TORQUE", "JOINT_VELOCITY", "OSC_POSITION", "OSC_POSE", "IK_POSE"}
    """
    # First check if default controller is not None; if it is not, load the appropriate controller
    if default_controller is not None:

        # Assert that requested default controller is in the available default controllers
        from robosuite.controllers import ALL_CONTROLLERS
        assert default_controller in ALL_CONTROLLERS, "Error: Unknown default controller specified. Requested {}, " \
            "available controllers: {}".format(default_controller, list(ALL_CONTROLLERS))

        # Store the default controller config fpath associated with the requested controller
        custom_fpath = os.path.join(os.path.dirname(__file__), '..',
                                    'controllers/config/{}.json'.format(default_controller.lower()))

    # Assert that the fpath to load the controller is not empty
    assert custom_fpath is not None, "Error: Either custom_fpath or default_controller must be specified!"

    # Attempt to load the controller
    try:
        with open(custom_fpath) as f:
            controller_config = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
              "Please check filepath and try again.".format(custom_fpath))

    # Return the loaded controller
    return controller_config


def controller_factory(name, params):
    """
    Generator for controllers

    Creates a Controller instance with the provided name and relevant params.

    Args:
        name: the name of the controller. Must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
            OSC_POSITION, OSC_POSE, IK_POSE}
        params: dict containing the relevant params to pass to the controller
        sim: Mujoco sim reference to pass to the controller

    Returns:
        Controller: Controller instance

    Raises:
        ValueError: [unknown controller]
    """

    interpolator = None
    if params["interpolation"] == "linear":
        interpolator = LinearInterpolator(max_delta=0.5,
                                          ndim=params["ndim"],
                                          controller_freq=(1 / params["sim"].model.opt.timestep),
                                          policy_freq=params["policy_freq"],
                                          ramp_ratio=params["ramp_ratio"])

    if name == "OSC_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.dim = 3                # EE control uses dim 3 for pos and ori each
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.ori_interpolate = True
        params["control_ori"] = True
        return OperationalSpaceController(interpolator_pos=interpolator,
                                          interpolator_ori=ori_interpolator, **params)

    if name == "OSC_POSITION":
        if interpolator is not None:
            interpolator.dim = 3                # EE control uses dim 3 for pos
        params["control_ori"] = False
        return OperationalSpaceController(interpolator_pos=interpolator, **params)

    if name == "IK_POSE":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.dim = 3                # EE IK control uses dim 3 for pos and dim 4 for ori
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.dim = 4
            ori_interpolator.ori_interpolate = True

        # Import pybullet server if necessary
        global pybullet_server
        from .ik import InverseKinematicsController
        if not pybullet_server:
            from robosuite.controllers.ik import PybulletServer
            pybullet_server = PybulletServer()
        return InverseKinematicsController(interpolator_pos=interpolator,
                                           interpolator_ori=ori_interpolator,
                                           bullet_server_id=pybullet_server.server_id,
                                           **params)

    if name == "JOINT_VELOCITY":
        return JointVelocityController(interpolator=interpolator, **params)

    if name == "JOINT_POSITION":
        return JointPositionController(interpolator=interpolator, **params)

    if name == "JOINT_TORQUE":
        return JointTorqueController(interpolator=interpolator, **params)

    raise ValueError("Unknown controller name: {}".format(name))
