"""Regression tests for the differential inverse-kinematics controller."""

import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


def test_ik_pose_zero_action_holds_initial_joint_configuration():
    """A neutral IK action must not be interpreted as a joint-position delta."""
    controller_config = suite.load_part_controller_config(default_controller="IK_POSE")
    controller_config = refactor_composite_controller_config(controller_config, "Panda", ["right"])
    env = suite.make(
        "Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
        seed=0,
    )

    try:
        env.reset()
        robot = env.robots[0]
        controller = robot.part_controllers["right"]
        initial_qpos = robot.get_robot_joint_positions().copy()

        assert controller.input_type == "absolute"
        for _ in range(100):
            env.step(np.zeros(env.action_dim))

        np.testing.assert_allclose(robot.get_robot_joint_positions(), initial_qpos, atol=1e-3)
        np.testing.assert_allclose(controller.goal_qpos, initial_qpos, atol=1e-3)
    finally:
        env.close()
