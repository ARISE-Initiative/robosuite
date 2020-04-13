"""
    Tests the basic interface of all grippers
"""
from robosuite.models.grippers import (
    PandaGripper,
    PR2Gripper,
    RethinkGripper,
    RobotiqGripper,
    RobotiqThreeFingerGripper,
)


def test_all_gripper():
    grippers = [
        PandaGripper(),
        PR2Gripper(),
        RethinkGripper(),
        RobotiqGripper(),
        RobotiqThreeFingerGripper(),
    ]
    for gripper in grippers:
        _test_gripper(gripper)


def _test_gripper(gripper):
    action = gripper.format_action([1] * gripper.dof)
    assert action is not None
    action = list(action)

    assert gripper.init_qpos is not None
    init_qpos = list(gripper.init_qpos)

    assert gripper.dof > 0

    assert gripper.joints is not None
    joints = list(gripper.joints)

    assert gripper.contact_geoms is not None

    assert gripper.visualization_sites is not None

    assert gripper.visualization_geoms is not None
