from .fixed_base_robot import FixedBaseRobot
from .mobile_base_robot import MobileBaseRobot
from .wheeled_robot import WheeledRobot
from .legged_robot import LeggedRobot

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

# Robot class mappings -- must be maintained manually
ROBOT_CLASS_MAPPING = {
    "Baxter": FixedBaseRobot,
    "Yumi": FixedBaseRobot,
    "IIWA": FixedBaseRobot,
    "Jaco": FixedBaseRobot,
    "Kinova3": FixedBaseRobot,
    "Panda": FixedBaseRobot,
    "Sawyer": FixedBaseRobot,
    "UR5e": FixedBaseRobot,
    "VX300S": FixedBaseRobot,
    "Z1": FixedBaseRobot,
    "PandaMobile": WheeledRobot,
    "GoogleRobot": WheeledRobot,
    "VX300SMobile": WheeledRobot,
    "Tiago": WheeledRobot,
    "PR2": WheeledRobot,
    "B1Z1": LeggedRobot,
    "SpotArm": LeggedRobot,
    "Go2Arx5": LeggedRobot,
    "GR1": LeggedRobot,
    "GR1FixedLowerBody": LeggedRobot,
    "GR1ArmsOnly": LeggedRobot,
    "GR1FloatingBody": LeggedRobot,
    "H1": LeggedRobot,
    "H1FixedLowerBody": LeggedRobot,
    "H1FloatingBody": LeggedRobot,
    "H1ArmsOnly": LeggedRobot,
    "G1": LeggedRobot,
    "G1FixedLowerBody": LeggedRobot,
    "G1ArmsOnly": LeggedRobot,
    "G1FloatingBody": LeggedRobot,
}
