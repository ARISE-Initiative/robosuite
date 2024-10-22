from .fixed_base_robot import FixedBaseRobot
from .mobile_base_robot import MobileBaseRobot
from .wheeled_robot import WheeledRobot
from .legged_robot import LeggedRobot

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

# Robot class mappings -- must be maintained manually
# These are the main robot used. Remaining robots are located in
# https://github.com/ARISE-Initiative/robosuite_models
ROBOT_CLASS_MAPPING = {
    "Baxter": FixedBaseRobot,
    "IIWA": FixedBaseRobot,
    "Jaco": FixedBaseRobot,
    "Kinova3": FixedBaseRobot,
    "Panda": FixedBaseRobot,
    "Sawyer": FixedBaseRobot,
    "UR5e": FixedBaseRobot,
    "PandaMobile": WheeledRobot,
    "Tiago": WheeledRobot,
    "SpotArm": LeggedRobot,
    "GR1": LeggedRobot,
    "GR1FixedLowerBody": LeggedRobot,
    "GR1ArmsOnly": LeggedRobot,
    "GR1FloatingBody": LeggedRobot,
}
