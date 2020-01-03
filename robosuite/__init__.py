import os

from robosuite.environments.base import make

from robosuite.environments.sawyer_lift import SawyerLift
from robosuite.environments.sawyer_stack import SawyerStack
from robosuite.environments.sawyer_pick_place import SawyerPickPlace
from robosuite.environments.sawyer_nut_assembly import SawyerNutAssembly

from robosuite.environments.panda_lift import PandaLift
from robosuite.environments.panda_stack import PandaStack
from robosuite.environments.panda_pick_place import PandaPickPlace
from robosuite.environments.panda_nut_assembly import PandaNutAssembly

from robosuite.environments.baxter_lift import BaxterLift
from robosuite.environments.baxter_peg_in_hole import BaxterPegInHole

__version__ = "0.3.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
