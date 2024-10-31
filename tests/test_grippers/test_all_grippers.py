"""
Tests the basic interface of all grippers.

This runs some basic sanity checks on the environment, namely, checking that:
    - Verifies that the gripper's action, init_qpos exist and are valid

Obviously, if an environment crashes during runtime, that is considered a failure as well.
"""

from robosuite.models.grippers import GRIPPER_MAPPING


def test_all_gripper():
    for name, gripper in GRIPPER_MAPPING.items():
        # Test all grippers except the null gripper
        if name not in {None, "WipingGripper"}:
            print("Testing {}...".format(name))
            _test_gripper(gripper())


def _test_gripper(gripper):
    action = gripper.format_action([1] * gripper.dof)
    assert action is not None

    assert gripper.init_qpos is not None
    assert len(gripper.init_qpos) == len(gripper.joints)


if __name__ == "__main__":
    test_all_gripper()
    print("Gripper tests completed.")
