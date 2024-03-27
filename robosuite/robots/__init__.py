from .manipulator import Manipulator
from .single_arm import SingleArm
from .bimanual import Bimanual
from .fixed_base_robot import SingleArmFixedBaseRobot, BimanualFixedBaseRobot
from .mobile_base_robot import MobileBaseRobot

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

# Robot class mappings -- must be maintained manually
ROBOT_CLASS_MAPPING = {
    "Baxter": BimanualFixedBaseRobot, # FixedBaseRobot,
    "IIWA": SingleArmFixedBaseRobot, # FixedBaseRobot,
    "Jaco": SingleArmFixedBaseRobot, # FixedBaseRobot,
    "Kinova3": SingleArmFixedBaseRobot, # FixedBaseRobot,
    "Panda": SingleArmFixedBaseRobot, # FixedBaseRobot,
    "PandaMobile": MobileBaseRobot,
    "Sawyer": SingleArmFixedBaseRobot, # FixedBaseRobot,
    "UR5e": SingleArmFixedBaseRobot, # FixedBaseRobot,
    "GoogleRobot": MobileBaseRobot,
    "Z1": SingleArmFixedBaseRobot, # FixedBaseRobot,
    "Aloha": BimanualFixedBaseRobot, # FixedBaseRobot,
    "VX300S": SingleArmFixedBaseRobot, # FixedBaseRobot,
}

BIMANUAL_ROBOTS = {k.lower() for k, v in ROBOT_CLASS_MAPPING.items() if v == Bimanual}
