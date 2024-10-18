import numpy as np

import robosuite as suite
import robosuite.utils.test_utils as tu
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.composite_utils import create_composite_robot

if __name__ == "__main__":

    name = "BaxterPanda"
    create_composite_robot(name, base="RethinkMount", robot="Tiago", grippers="PandaGripper")
    controller_config = load_composite_controller_config(controller="BASIC", robot=name)

    tu.create_and_test_env(env="Lift", robots=name, controller_config=controller_config)
