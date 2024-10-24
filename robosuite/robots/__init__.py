from .fixed_base_robot import FixedBaseRobot
from .mobile_robot import MobileRobot
from .wheeled_robot import WheeledRobot
from .legged_robot import LeggedRobot

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

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
    "PandaOmron": WheeledRobot,
    "Tiago": WheeledRobot,
    "SpotWithArm": LeggedRobot,
    "SpotWithArmFloating": LeggedRobot,
    "GR1": LeggedRobot,
    "GR1FixedLowerBody": LeggedRobot,
    "GR1ArmsOnly": LeggedRobot,
    "GR1FloatingBody": LeggedRobot,
    "PandaDexRH": FixedBaseRobot,
    "PandaDexLH": FixedBaseRobot,
    "SpotArmRethink": FixedBaseRobot,
    "SpotArmRobotiqGripper": FixedBaseRobot,
}

target_type_mapping = {
    "FixedBaseRobot": FixedBaseRobot,
    "MobileRobot": MobileRobot,
    "WheeledRobot": WheeledRobot,
    "LeggedRobot": LeggedRobot,
}


def register_robot_class(target_type, **kwargs):
    def decorator(target_class):
        # Store the class in the registry with additional arguments
        ROBOT_CLASS_MAPPING.update({target_class.__name__: target_type_mapping[target_type]})

        return target_class  # Return the class itself

    return decorator
