"""
    Tests the basic interface of all grippers
"""
from robosuite.models.grippers import GRIPPER_MAPPING


def test_all_gripper():
    for _, gripper in GRIPPER_MAPPING.items():
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
