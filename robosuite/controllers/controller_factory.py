"""
Defines a string based method of initializing controllers
"""
from .ee_imp import EEImpController
from .joint_vel import JointVelController
from .joint_imp import JointImpController
from .joint_tor import JointTorController
from .interpolators.linear_interpolator import LinearInterpolator

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
        interpolator = LinearInterpolator(max_dx=0.5,
                                          ndim=params["ndim"],
                                          controller_freq=params["sim"].model.opt.timestep,
                                          policy_freq=params["policy_freq"],
                                          ramp_ratio=0.5)

    if name == "EE_POS_ORI":
        ori_interpolator = None
        if interpolator is not None:
            interpolator.dim = 3                # EE control uses dim 3 for pos and ori each
            ori_interpolator = deepcopy(interpolator)
        params["control_ori"] = True
        return EEImpController(interpolator_pos=interpolator, interpolator_ori=ori_interpolator, **params)
    if name == "EE_POS":
        if interpolator is not None:
            interpolator.dim = 3                # EE control uses dim 3 for pos and ori each
        params["control_ori"] = False
        return EEImpController(interpolator_pos=interpolator, **params)
    if name == "EE_IK":
        global pybullet_server
        from .ee_ik import EEIKController
        if not pybullet_server:
            from robosuite.controllers.ee_ik import PybulletServer
            pybullet_server = PybulletServer()
        return EEIKController(interpolator=interpolator, bullet_server_id=pybullet_server.server_id, **params)
    if name == "JOINT_VEL":
        return JointVelController(interpolator=interpolator, **params)
    if name == "JOINT_IMP":
        return JointImpController(interpolator=interpolator, **params)
    if name == "JOINT_TOR":
        return JointTorController(interpolator=interpolator, **params)

    raise ValueError("Unknown controller name: {}".format(name))
