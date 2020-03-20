"""
Test all controllers on the Sawyer(Lift) task environment as a test case.

The following controllers are tested:
OSC - Position & Orientation
OSC - Position only
IK - Position & Orientation
Joint Impedance
Joint Velocity
Joint Torque

This (non-exhaustive) test script checks for qualitative irregularities in controller behavior.
However, this testing module also checks for action space correctness and dimensionality.
For every controller action space, runs through each dimension and executes a perturbation "test_value" from its
neutral (stationary) value for a certain amount of time "steps_per_action", and then returns to all neutral values
for time "steps_per_rest" before proceeding with the next action dim.

Please reference the controller README in the robosuite/controllers directory for an overview of each controller.
Controllers are expected to behave in a generally controlled manner, according to their control space.
    E.g.: the Pos / Ori controller should be expected to move linearly in the x direction first, then the y direction,
        then the z direction, and then begin rotating about its x-axis, then y-axis, then z-axis.

As this is strictly a qualitative set of tests, it is up to the developer / user to examine for specific irregularities.
However, the expected qualitative behavior is described below for each controller:

* EE_POS_ORI: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the global coordinate frame
* EE_POS: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
* EE_IK: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the local robot end effector frame
* JOINT_IMP: Robot Joints move sequentially in a controlled fashion
* JOINT_VEL: Robot Joints move sequentially in a controlled fashion
* JOINT_TORQUE: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the
            "controller" is really just a wrapper for direct torque control of the mujoco actuators. Therefore, a
            "neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!

"""
import numpy as np

import robosuite as suite

import os
import json
import robosuite.utils.transform_utils as T

# Define the controllers to use (action_dim, num_test_steps, test_value, neutral control values)
controllers = {
    "ee_pos_ori":   [7, 6, 0.1, np.array([0,0,0,0,0,0,-1], dtype=float)],
    "ee_pos":       [4, 3, 0.1, np.array([0,0,0,-1], dtype=float)],
    "ee_ik":        [8, 6, 0.01, np.array([0,0,0,0,0,0,1,-1], dtype=float)],
    "joint_imp":    [8, 7, 0.2, np.array([0,0,0,0,0,0,0,-1], dtype=float)],
    "joint_vel":    [8, 7, -0.05, np.array([0,0,0,0,0,0,0,-1], dtype=float)],
    "joint_torque": [8, 7, 0.001, np.array([0,0,0,0,0,0,0,-1], dtype=float)]
}

# Define the number of timesteps to use per controller action as well as timesteps in between actions
steps_per_action = 50
steps_per_rest = 25


def test_all_controllers():
    for controller_name in controllers.keys():
        # Define variables for each controller test
        action_dim = controllers[controller_name][0]
        num_test_steps = controllers[controller_name][1]
        test_value = controllers[controller_name][2]
        neutral = controllers[controller_name][3]

        # Define controller path to load
        controller_path = os.path.join(os.path.dirname(__file__),
                                       '../../robosuite', 'controllers/config/{}.json'.format(controller_name))
        with open(controller_path) as f:
            controller_config = json.load(f)

        # Now, create a test env for testing the controller on
        env = suite.make(
            "SawyerLift",
            has_renderer=True,  # use on-screen renderer for visual validation
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=(steps_per_action + steps_per_rest) * num_test_steps,
            controller_config=controller_config
        )
        print("Testing controller: {}...".format(controller_name))

        env.reset()
        # For localised controllers, get a closer camera angle for better viewing
        if controller_name in {"ee_pos_ori", "ee_pos", "ee_ik"}:
            env.viewer.set_camera(camera_id=2)

        # get action range
        action_min, action_max = env.action_spec
        assert action_min.shape == action_max.shape
        assert action_min.shape[0] == action_dim, "Expected {}, got {}".format(action_dim, action_min.shape[0])

        # Keep track of done variable to know when to break loop
        done = False
        count = 0
        # Loop through controller space
        while not done:
            action = neutral.copy()
            for i in range(steps_per_action):
                if controller_name == 'ee_ik' and count > 2:
                    # Convert from euler angle to quat here since we're working with quats
                    angle = np.zeros(3)
                    angle[count - 3] = test_value
                    action[3:7] = T.mat2quat(T.euler2mat(angle))
                else:
                    action[count] = test_value
                _, _, done, _ = env.step(action)
                env.render()
            count += 1
            for i in range(steps_per_rest):
                _, _, done, _ = env.step(neutral)
                env.render()

        # Tests passed!
        print("All controller tests completed.")


if __name__ == "__main__":

    test_all_controllers()
