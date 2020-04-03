"""
Test the linear interpolator on the Lift task with Sawyer arm environment as a test case.

The linear interpolator is meant to increase the stability and overall safety of a robot arm's trajectory when reaching
a setpoint, "ramping up" the actual action command sent to a given controller from zero to the actual inputted action
over a fraction of the timesteps in betwteen each high-level input action (the "ramp ratio"). As a result, the
resulting trajectory is slower, but smoother, proportional to the interpolator's ramp ratio setting.

This test verifies that the linear interpolator works correctly on both the IK and OSC controller for both position and
orientation, and proceeds as follows:

    1. Given a constant delta position action, and with the interpolator disabled, we will measure the number of
        timesteps it takes for the Sawyer arm to reach a certain location threshold in the world frame

    2. We will repeat Step 1, but this time with the interpolator enabled and with a ramp ratio of 1.0 (max value)

    3. We expect and verify that the difference in timesteps measured between Steps 1 and 2 are as expected, according
        to the following equation:

            > Total distance travelled = d = ∫v(t) dt ≈ ∑ v_i * t_s, where t_s is the number of seconds per timestep

                > This is equivalent to the area plotted under the velocity-time curve. While this varies between
                    controllers (OSC is torque-based while IK is velocity-based), both controllers use a proportional
                    gain term for converging to a given setpoint, implying that an interpolated trajectory will have a
                    lower overall average velocity

                > Since average velocity v_bar ∝ d / timesteps, and our d is constant:

                        >> timesteps_2 > timesteps_1

    **************************************************************************************************************
    ** Therefore, we expect the interpolated trajectory (with a ramp ratio of 1) to take distinctly longer in   **
    **  order to reach the same end goal as the non-interpolated trajectory                                     **
    **************************************************************************************************************

Note: For the test, we set an arbitrary threshold ratio of 1.10 of interpolated time / non-interpolated time that we
        assume as the minimum
"""

import numpy as np

import robosuite as suite

import os
import json
import robosuite.utils.transform_utils as T
import argparse

# Define the threshold locations, delta values, and ratio #

# Translation trajectory
pos_y_threshold = 0.1
delta_pos_y = 0.01
pos_action_osc = [0, delta_pos_y * 40, 0]
pos_action_ik = [0, delta_pos_y, 0]

# Rotation trajectory
rot_r_threshold = np.pi / 2
delta_rot_r = 0.01
rot_action_osc = [-delta_rot_r * 40, 0, 0]
rot_action_ik = T.mat2quat(T.euler2mat([delta_rot_r * 5, 0, 0]))

# Concatenated thresholds and corresponding indexes (y = 1 in x,y,z; roll = 0 in r,p,y)
thresholds = [pos_y_threshold, rot_r_threshold]
indexes = [1, 0]

# Threshold ratio
min_ratio = 1.10

# Define arguments for this test
parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Whether to render tests or run headless")
args = parser.parse_args()


# Running the actual test #
def test_linear_interpolator():

    for controller_name in ["EE_POS_ORI", "EE_IK"]:

        for traj in ["pos", "ori"]:

            # Define counter to increment timesteps for each trajectory
            timesteps = [0, 0]

            for interpolator in [None, "linear"]:
                # Define numpy seed so we guarantee consistent starting pos / ori for each trajectory
                np.random.seed(3)

                # Define controller path to load
                controller_path = os.path.join(os.path.dirname(__file__),
                                               '../../robosuite',
                                               'controllers/config/{}.json'.format(controller_name.lower))
                with open(controller_path) as f:
                    controller_config = json.load(f)
                    controller_config["interpolation"] = interpolator
                    controller_config["ramp_ratio"] = 1.0

                # Now, create a test env for testing the controller on
                env = suite.make(
                    "Lift",
                    robots="Sawyer",
                    has_renderer=args.render,  # by default, don't use on-screen renderer for visual validation
                    has_offscreen_renderer=False,
                    use_camera_obs=False,
                    horizon=10000,
                    control_freq=20,
                    controller_configs=controller_config
                )

                # Reset the environment
                env.reset()

                # Notify user a new trajectory is beginning
                print("\nTesting controller {} with trajectory {} and interpolator={}...".format(
                    controller_name, traj, interpolator))

                # If rendering, set controller to front view to get best angle for viewing robot movements
                if args.render:
                    env.viewer.set_camera(camera_id=0)

                # Keep track of state of robot eef (pos, ori (euler))
                initial_state = [env.robots[0]._right_hand_pos, T.mat2euler(env.robots[0]._right_hand_orn)]
                dstate = [env.robots[0]._right_hand_pos - initial_state[0],
                          T.mat2euler(env.robots[0]._right_hand_orn) - initial_state[1]]

                # Define the uniform trajectory action
                if traj == "pos":
                    pos_act = pos_action_ik if controller_name == "ee_ik" else pos_action_osc
                    rot_act = T.mat2quat(T.euler2mat(np.zeros(3))) if controller_name == "ee_ik" else np.zeros(3)
                else:
                    pos_act = np.zeros(3)
                    rot_act = rot_action_ik if controller_name == "ee_ik" else rot_action_osc

                # Compose the action
                action = np.concatenate([pos_act, rot_act, [0]])

                # Determine which trajectory we're executing
                k = 0 if traj == "pos" else 1
                j = 0 if not interpolator else 1

                # Run trajectory until the threshold condition is met
                while abs(dstate[k][indexes[k]]) < abs(thresholds[k]):
                    env.step(action)
                    if args.render:
                        env.render()

                    # Update timestep count and state
                    timesteps[j] += 1
                    dstate = [env.robots[0]._right_hand_pos - initial_state[0],
                              T.mat2euler(env.robots[0]._right_hand_orn) - initial_state[1]]

                # When finished, print out the timestep results
                print("Completed trajectory. Took {} timesteps total.".format(timesteps[j]))

                # Shut down this env before starting the next test
                env.close()

            # Assert that the interpolated path is slower than the non-interpolated one
            assert timesteps[1] > min_ratio * timesteps[0], "Error: Interpolated trajectory time should be longer " \
                                                            "than non-interpolated!"

    # Tests completed!
    print()
    print("-" * 80)
    print("All linear interpolator testing completed.\n")


if __name__ == "__main__":
    test_linear_interpolator()
