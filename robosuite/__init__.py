from robosuite.environments.base import make

from robosuite.environments.lift import Lift
from robosuite.environments.stack import Stack
from robosuite.environments.nut_assembly import NutAssembly
from robosuite.environments.pick_place import PickPlace
from robosuite.environments.door import Door
from robosuite.environments.wipe import Wipe
from robosuite.environments.two_arm_lift import TwoArmLift
from robosuite.environments.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.two_arm_handover import TwoArmHandover

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS

__version__ = "1.0.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
