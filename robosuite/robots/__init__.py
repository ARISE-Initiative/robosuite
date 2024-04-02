from .fixed_robot import FixedRobot
from .mobile_robot import MobileRobot
from .wheeled_robot import WheeledRobot
from .legged_robot import LeggedRobot

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

# Robot class mappings -- must be maintained manually
ROBOT_CLASS_MAPPING = {
    "Baxter": FixedRobot,
    "IIWA": FixedRobot,
    "Jaco": FixedRobot,
    "Kinova3": FixedRobot,
    "Panda": FixedRobot,
    "PandaMobile": WheeledRobot,
    "Sawyer": FixedRobot,
    "UR5e": FixedRobot,
    "GoogleRobot": WheeledRobot,
    "VX300S": FixedRobot,
    "VX300SMobile": WheeledRobot,
}
