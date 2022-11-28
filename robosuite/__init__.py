try:
    from robosuite.macros_private import *
except ImportError:
    import robosuite
    from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

    ROBOSUITE_DEFAULT_LOGGER.warn("No private macro file found!")
    ROBOSUITE_DEFAULT_LOGGER.warn("It is recommended to use a private macro file")
    ROBOSUITE_DEFAULT_LOGGER.warn("To setup, run: python {}/scripts/setup_macros.py".format(robosuite.__path__[0]))

from robosuite.environments.base import make

# Manipulation environments
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS

__version__ = "1.4.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
