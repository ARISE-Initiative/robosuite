from .manipulator import Manipulator
from .mobile_manipulator import MobileManipulator
from .single_arm import SingleArm
from .mobile_single_arm import MobileSingleArm
from .bimanual import Bimanual

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

# Robot class mappings -- must be maintained manually
ROBOT_CLASS_MAPPING = {
    "Baxter": Bimanual,
    "IIWA": SingleArm,
    "Jaco": SingleArm,
    "Kinova3": SingleArm,
    "Panda": SingleArm,
    "PandaMobile": MobileSingleArm,
    "Sawyer": SingleArm,
    "UR5e": SingleArm,
    "GoogleRobot": MobileSingleArm,
}

BIMANUAL_ROBOTS = {k.lower() for k, v in ROBOT_CLASS_MAPPING.items() if v == Bimanual}
