"""
    Tests the basic interface of all grippers
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

    assert gripper.dof > 0

    assert gripper.joints is not None

    assert gripper.contact_geoms is not None

    assert gripper.visualization_sites is not None

    assert gripper.visualization_geoms is not None


if __name__ == "__main__":
    test_all_gripper()
    print("Gripper tests completed.")
