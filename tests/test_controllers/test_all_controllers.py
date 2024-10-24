"""
Test all controllers on the Lift task with Sawyer robot environment as a test case.

The following controllers are tested:
Operational Space Control - Position & Orientation
Operational Space Control - Position only
Inverse Kinematics - Position & Orientation
Joint Impedance
Joint Velocity
Joint Torque

This (non-exhaustive) test script checks for qualitative irregularities in controller behavior.
However, this testing module also checks for action space correctness and dimensionality.
For every controller action space, runs through each dimension and executes a perturbation "test_value" from its
neutral (stationary) value for a certain amount of time "steps_per_action", and then returns to all neutral values
for time "steps_per_rest" before proceeding with the next action dim.

    E.g.: Given that the expected action space of the Pos / Ori (OSC_POSE) controller (without a gripper) is
    (dx, dy, dz, ax, ay, az), the testing sequence of actions over time will be:

        ***START OF TEST***
        ( dx,  0,  0,  0,  0,  0, grip)     <-- Translation in x-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest'   steps
        (  0, dy,  0,  0,  0,  0, grip)     <-- Translation in y-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest'   steps
        (  0,  0, dz,  0,  0,  0, grip)     <-- Translation in z-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest'   steps
        (  0,  0,  0,  a,  0,  0, grip)     <-- Rotation about x axis           for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest'   steps
        (  0,  0,  0,  0,  a,  0, grip)     <-- Rotation about y axis           for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest'   steps
        (  0,  0,  0,  0,  0,  a, grip)     <-- Rotation about z axis           for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest'   steps
        ***END OF TEST***

    Thus the OSC_POSE controller should be expected to sequentially move linearly in the x direction first,
        then the y direction, then the z direction, and then begin sequentially rotating about its x-axis,
        then y-axis, then z-axis.

Please reference the controller README in the robosuite/controllers directory for an overview of each controller.
Controllers are expected to behave in a generally controlled manner, according to their control space.
    E.g.: the Pos / Ori controller should be expected to move linearly in the x direction first, then the y direction,
        then the z direction, and then begin rotating about its x-axis, then y-axis, then z-axis.

As this is strictly a qualitative set of tests, it is up to the developer / user to examine for specific irregularities.
However, the expected qualitative behavior is described below for each controller:

* OSC_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis,
            y-axis, z-axis, relative to the global coordinate frame
* OSC_POSITION: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
* IK_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the local robot end effector frame
* JOINT_POSITION: Robot Joints move sequentially in a controlled fashion
* JOINT_VELOCITY: Robot Joints move sequentially in a controlled fashion
* JOINT_TORQUE: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the
            "controller" is really just a wrapper for direct torque control of the mujoco actuators. Therefore, a
            "neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!

Note that by default, there is no rendering. Rendering can be enabled by setting the --render flag when calling this
test script.

"""
import argparse
import logging

import numpy as np

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite import load_composite_controller_config
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.ERROR)

# Arguments for this test script
parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Whether to render this test or not for visual validation")
args = parser.parse_args()

# Define the controllers to use (action_dim, num_test_steps, test_value)
controllers = {
    "OSC_POSE": [7, 6, 0.1],
    "OSC_POSITION": [4, 3, 0.1],
    "IK_POSE": [7, 6, 0.01],
    "JOINT_POSITION": [8, 7, 0.2],
    "JOINT_VELOCITY": [8, 7, -0.1],
    "JOINT_TORQUE": [8, 7, 0.25],
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
        neutral = np.zeros(action_dim)

        # Define controller path to load
        controller_config = load_composite_controller_config(controller="BASIC")

        # Now, create a test env for testing the controller on
        env = suite.make(
            "Lift",
            robots="Sawyer",
            has_renderer=args.render,  # use on-screen renderer for visual validation only if requested
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=(steps_per_action + steps_per_rest) * num_test_steps,
            controller_configs=controller_config,
        )
        print("Testing controller: {}...".format(controller_name))

        env.reset()
        # If rendering, set controller to front view to get best angle for viewing robot movements
        if args.render:
            env.viewer.set_camera(camera_id=0)

        # get action range
        action_min, action_max = env.action_spec
        assert action_min.shape == action_max.shape
        assert action_min.shape[0] == action_dim, "Expected {}, got {}".format(action_dim, action_min.shape[0])

        # Keep track of done variable to know when to break loop
        count = 0
        # Loop through controller space
        while count < num_test_steps:
            action = neutral.copy()
            for i in range(steps_per_action):
                if controller_name in {"IK_POSE", "OSC_POSE"} and count > 2:
                    # Set this value to be the angle and set appropriate axis
                    vec = np.zeros(3)
                    vec[count - 3] = test_value
                    action[3:6] = vec
                else:
                    action[count] = test_value
                env.step(action)
                if args.render:
                    env.render()
            for i in range(steps_per_rest):
                env.step(neutral)
                if args.render:
                    env.render()
            count += 1

        # Shut down this env before starting the next test
        env.close()

    # Tests passed!
    print("All controller tests completed.")


if __name__ == "__main__":

    test_all_controllers()
