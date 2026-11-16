"""
Test that the flat environment action is split correctly between multiple robots.

RobotEnv caches the action dimension of each robot on reset, and uses it to hand each robot its own slice of the
action at every simulation step. This checks the cached dimensions match the robots, and that each robot receives
exactly its slice of the action, also after a reset.
"""
import numpy as np

import robosuite as suite


def test_action_split_between_robots():
    env = suite.make(
        "TwoArmLift",
        robots=["Panda", "Sawyer"],
        env_configuration="opposed",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    def record_control_actions():
        # Robots may be recreated on reset, so this needs to be done after each reset
        received = [[] for _ in env.robots]
        for robot, robot_received in zip(env.robots, received):

            def recording_control(action, policy_step=False, control=robot.control, out=robot_received):
                out.append(np.array(action))
                return control(action, policy_step=policy_step)

            robot.control = recording_control
        return received

    for _ in range(2):
        env.reset()
        received = record_control_actions()

        robot_action_dims = [robot.action_dim for robot in env.robots]
        assert env._robot_action_dims == robot_action_dims
        assert env.action_dim == sum(robot_action_dims)

        action = np.linspace(-1.0, 1.0, env.action_dim)
        env.step(action)

        cutoff = 0
        for robot_action_dim, robot_received in zip(robot_action_dims, received):
            # The robot is controlled at every simulation substep of the environment step
            assert len(robot_received) > 0
            for robot_action in robot_received:
                np.testing.assert_array_equal(robot_action, action[cutoff : cutoff + robot_action_dim])
            cutoff += robot_action_dim

    env.close()
