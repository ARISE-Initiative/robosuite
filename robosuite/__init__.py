from robosuite.environments.base import make

from robosuite.environments.lift import Lift
from robosuite.environments.two_arm_lift import TwoArmLift
from robosuite.environments.stack import Stack
from robosuite.environments.nut_assembly import NutAssembly
from robosuite.environments.pick_place import PickPlace
from robosuite.environments.two_arm_peg_in_hole import TwoArmPegInHole

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS, ALL_CONTROLLERS_INFO

__version__ = "0.3.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
