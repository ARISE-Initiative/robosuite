"""
Defines a string based method of initializing controllers
"""
from .ee_imp import EndEffectorImpedanceController
from .joint_vel import JointVelocityController
from .joint_imp import JointImpedanceController
from .joint_tor import JointTorqueController
from .interpolators.linear_interpolator import LinearInterpolator

import json
import os

from copy import deepcopy


def load_controller_config(custom_fpath=None, default_controller=None):
    """
    Utility function that loads the desired controller and returns the loaded configuration as a dict

    If @default_controller is specified, any value inputted to @custom_fpath is overridden and the default controller
        configuration is automatically loaded. See specific arg description below for available default controllers.

    Args:
        @custom_fpath (str): Absolute filepath to the custom controller configuration .json file to be loaded
        @default_controller (str): If specified, overrides @custom_fpath and loads a default configuration file for the
            specified controller.
            Choices are: {"JOINT_IMP", "JOINT_TOR", "JOINT_VEL", "EE_POS", "EE_POS_ORI", "EE_IK"}
    """
    # First check if default controller is not None; if it is not, load the appropriate controller
    if default_controller is not None:
        # Map default_controller to lower case so it is case invariant
        default_controller = default_controller.lower()

        # Dict mapping expected inputs to str value for loading the appropriate default
        controllers = {"joint_vel", "joint_tor", "joint_imp", "ee_pos", "ee_pos_ori", "ee_ik"}

        # Assert that requested default controller is in the available default controllers
        assert default_controller in controllers, "Error: Unknown default controller specified. Requested {}," \
                                                  "available controllers: {}".format(default_controller, controllers)

        # Store the default controller config fpath associated with the requested controller
        custom_fpath = os.path.join(os.path.dirname(__file__), '..',
                                    'controllers/config/{}.json'.format(default_controller))

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
        name: the name of the controller. Must be one of: {JOINT_IMP, JOINT_TOR, JOINT_VEL, EE_POS, EE_POS_ORI, EE_IK}
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
                                          controller_freq=params["controller_freq"],
                                          policy_freq=params["policy_freq"],
                                          ramp_ratio=params["ramp_ratio"])

    if name == "EE_POS_ORI":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.dim = 3                # EE control uses dim 3 for pos and ori each
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.ori_interpolate = True
        params["control_ori"] = True
        return EndEffectorImpedanceController(interpolator_pos=interpolator,
                                              interpolator_ori=ori_interpolator, **params)
    if name == "EE_POS":
        if interpolator is not None:
            interpolator.dim = 3                # EE control uses dim 3 for pos
        params["control_ori"] = False
        return EndEffectorImpedanceController(interpolator_pos=interpolator, **params)
    if name == "EE_IK":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.dim = 3                # EE IK control uses dim 3 for pos and dim 4 for ori
            ori_interpolator = deepcopy(interpolator)
            ori_interpolator.dim = 4
            ori_interpolator.ori_interpolate = True
        from .ee_ik import EndEffectorInverseKinematicsController
        return EndEffectorInverseKinematicsController(interpolator_pos=interpolator,
                                                      interpolator_ori=ori_interpolator, **params)
    if name == "JOINT_VEL":
        return JointVelocityController(interpolator=interpolator, **params)
    if name == "JOINT_IMP":
        return JointImpedanceController(interpolator=interpolator, **params)
    if name == "JOINT_TOR":
        return JointTorqueController(interpolator=interpolator, **params)

    raise ValueError("Unknown controller name: {}".format(name))
