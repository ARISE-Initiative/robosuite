"""
Tests the basic interface of all robots.

This runs some basic sanity checks on the robots, namely, checking that:
    - Verifies that all single-arm robots have properly defined contact geoms.

Obviously, if an environment crashes during runtime, that is considered a failure as well.
"""
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots import SingleArm


def test_single_arm_robots():
    for name, robot in ROBOT_CLASS_MAPPING.items():
        if robot == SingleArm:
            print(f"Testing {name}")
            _test_contact_geoms(robot(name))


def _test_contact_geoms(robot):
    robot.load_model()
    contact_geoms = robot.robot_model._contact_geoms
    for geom in contact_geoms:
        assert isinstance(
            geom, str
        ), f"The geom {geom} is of type {type(geom)}, but should be {type('placeholder')}"


if __name__ == "__main__":
    test_single_arm_robots()
    print("Robot tests completed.")
