from .controller_factory import controller_factory, load_controller_config, reset_controllers, get_pybullet_server
from .osc import OperationalSpaceController
from .joint_pos import JointPositionController
from .joint_vel import JointVelocityController
from .joint_tor import JointTorqueController

ALL_CONTROLLERS_INFO = {
    "JOINT_VELOCITY":   "Joint Velocity",
    "JOINT_TORQUE":     "Joint Torque",
    "JOINT_POSITION":   "Joint Impedance",
    "OSC_POSITION":     "End Effector Position using Operational Space Control",
    "OSC_POSE":         "End Effector Pose (Pos + Ori) using Operational Space Control",
    "IK_POSE":          "End Effector Pose (Pos + Ori) using Inverse Kinematics (note: must have PyBullet installed)",
}

ALL_CONTROLLERS = ALL_CONTROLLERS_INFO.keys()
