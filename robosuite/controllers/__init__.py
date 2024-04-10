from .controller_factory import controller_factory, load_controller_config, reset_controllers, get_pybullet_server
from .manager import controller_manager_factory
from .arm import *


CONTROLLER_INFO = {
    "JOINT_VELOCITY": "Joint Velocity",
    "JOINT_TORQUE": "Joint Torque",
    "JOINT_POSITION": "Joint Position",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSE": "Operational Space Control (Position + Orientation)",
    "IK_POSE": "Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)",
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()
