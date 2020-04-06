from .controller_factory import controller_factory, load_controller_config
from .ee_imp import EndEffectorImpedanceController
from .joint_imp import JointImpedanceController
from .joint_vel import JointVelocityController
from .joint_tor import JointTorqueController

ALL_CONTROLLERS_INFO = {
    "JOINT_VEL":  "Joint Velocity",
    "JOINT_TOR":  "Joint Torque",
    "JOINT_IMP":  "Joint Impedance",
    "EE_POS":     "End Effector Position",
    "EE_POS_ORI": "End Effector Position Orientation",
    "EE_IK":      "End Effector Inverse Kinematics (note: must have PyBullet installed)",
}

ALL_CONTROLLERS = ALL_CONTROLLERS_INFO.keys()