import os

from robosuite.environments.base import make

from robosuite.environments.sawyer_lift import SawyerLift
from robosuite.environments.sawyer_stack import SawyerStack
from robosuite.environments.sawyer_pick_place import SawyerPickPlace
from robosuite.environments.sawyer_nut_assembly import SawyerNutAssembly
from robosuite.environments.sawyer_free_space_traj import SawyerFreeSpaceTraj
from robosuite.environments.sawyer_door import SawyerDoor
from robosuite.environments.sawyer_wipe_force import SawyerWipeForce
from robosuite.environments.sawyer_wipe_pegs import SawyerWipePegs
from robosuite.environments.sawyer_wipe_tactile import SawyerWipeTactile
from robosuite.environments.sawyer_wipe_3d_tactile import SawyerWipe3DTactile

from robosuite.environments.panda_lift import PandaLift
from robosuite.environments.panda_stack import PandaStack
from robosuite.environments.panda_pick_place import PandaPickPlace
from robosuite.environments.panda_nut_assembly import PandaNutAssembly
from robosuite.environments.panda_free_space_traj import PandaFreeSpaceTraj
from robosuite.environments.panda_door import PandaDoor
from robosuite.environments.panda_wipe_force import PandaWipeForce
from robosuite.environments.panda_wipe_pegs import PandaWipePegs
from robosuite.environments.panda_wipe_tactile import PandaWipeTactile
from robosuite.environments.panda_wipe_3d_tactile import PandaWipe3DTactile

from robosuite.environments.baxter_lift import BaxterLift
from robosuite.environments.baxter_peg_in_hole import BaxterPegInHole

__version__ = "0.1.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
