"""
Script to test composite robots:

$ pytest -s tests/test_robots/test_composite_robots.py
"""
import pytest

import robosuite.utils.composite_utils as cu
import robosuite.utils.test_utils as tu
from robosuite.controllers import load_composite_controller_config
from robosuite.models.grippers import GRIPPER_MAPPING

TEST_ROBOTS = ["Baxter", "IIWA", "Jaco", "Kinova3", "Panda", "Sawyer", "UR5e", "Tiago", "SpotArm", "GR1"]
TEST_BASES = [
    "RethinkMount",
    "RethinkMinimalMount",
    "NullMount",
    "OmronMobileBase",
    "NullMobileBase",
    "NoActuationBase",
    "Spot",
    "SpotFloating",
]


@pytest.mark.parametrize("robot", TEST_ROBOTS)
@pytest.mark.parametrize("gripper", GRIPPER_MAPPING.keys())
def test_composite_robot_gripper_combinations(robot, gripper):
    if tu.is_robosuite_robot(robot):
        if robot in ["Tiago"]:
            base = "NullMobileBase"
        elif robot == "GR1":
            base = "NoActuationBase"
        else:
            base = "RethinkMount"

        composite_robot = cu.create_composite_robot(name="CompositeRobot", robot=robot, base=base, grippers=gripper)
        controller_config = load_composite_controller_config(controller="BASIC", robot="CompositeRobot")
        tu.create_and_test_env(env="Lift", robots="CompositeRobot", controller_config=controller_config, render=False)


@pytest.mark.parametrize("robot", TEST_ROBOTS)
@pytest.mark.parametrize("base", TEST_BASES)
def test_composite_robot_base_combinations(robot, base):
    if tu.is_robosuite_robot(robot):
        if robot in ["Tiago", "GR1", "SpotArm"]:
            pytest.skip(f"Skipping {robot} for now since it we typically do not attach it to another base.")
        elif base in ["NullMobileBase", "NoActuationBase", "Spot", "SpotFloating"]:
            pytest.skip(f"Skipping {base} for now since comopsite robots do not use {base}.")
        else:
            print(robot, base)
            composite_robot = cu.create_composite_robot(
                name="CompositeRobot", robot=robot, base=base, grippers="RethinkGripper"
            )
            controller_config = load_composite_controller_config(controller="BASIC", robot="CompositeRobot")
            tu.create_and_test_env(
                env="Lift", robots="CompositeRobot", controller_config=controller_config, render=True
            )
