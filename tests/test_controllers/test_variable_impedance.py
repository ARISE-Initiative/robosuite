"""
Test the variable impedance feature of impedance-based controllers (OSC, Joint Position) on the Lift task with
Sawyer arm environment as a test case.

The variable impedance feature allows per-action fine-grained control over the specific impedance gains when executing
impedance control (namely, "kp" and "damping" ratios). This allows a given controller to execute more complex and
potentially interactive trajectories by varying the net impedance of the controlled actuators over time.

This (qualitative) test verifies that the variable impedance works correctly on both the OSC Pose / Position and
Joint Position controllers, and proceeds as follows:

    1. Given a constant delta position action, and with the the kp values set to critically-damped, we will ramp up
        the kp values to its max and then ramp down the values. We qualitatively expect the arm to accelerate as the kp
        values are ramped, and then slow down as they are decreased.

    2. The environment will then be reset. Given a constant delta position action, and with kp values set to its
        default value, we will ramp up the damping values to its max and then ramp down the values. We qualitatively
        expect the arm to slow down as the damping values are ramped, and then increase in speed as they are decreased.

    3. We will repeat Step 1 and 2 for each of the tested controllers.

Periodic prijntouts should verify the above patterns; conversely, running the script with the "--render" argument will
render the trajectories to allow for visual analysis of gains
"""

import numpy as np

import robosuite as suite

import os
import json
import argparse


# Define the rate of change when sweeping through kp / damping values
num_timesteps_per_change = 10
percent_increase = 0.05

# Define delta values for trajectory
d = 0.05

# Define default values for fixing one of the two gains
kp_default = 150
damping_default = 1     # critically damped

# Define arguments for this test
parser = argparse.ArgumentParser()
parser.add_argument("--render", action="store_true", help="Whether to render tests or run headless")
args = parser.parse_args()


# Running the actual test #
def test_linear_interpolator():

    for controller_name in ["OSC_POSE", "OSC_POSITION", "JOINT_POSITION"]:

        # Define numpy seed so we guarantee consistent starting pos / ori for each trajectory
        np.random.seed(3)

        # Define controller path to load
        controller_path = os.path.join(os.path.dirname(__file__),
                                       '../../robosuite',
                                       'controllers/config/{}.json'.format(controller_name.lower()))

        # Load the controller
        with open(controller_path) as f:
            controller_config = json.load(f)

        # Manually edit impedance settings
        controller_config["impedance_mode"] = "variable"
        controller_config["kp_limits"] = [0, 300]
        controller_config["damping_limits"] = [0, 10]

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

        # Setup printing options for numbers
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        # Get limits on kp and damping values
        # Define control dim. Note that this is not the action space, but internal dimensionality of gains
        control_dim = 6 if "OSC" in controller_name else 7
        low, high = env.action_spec
        damping_low, kp_low = low[:control_dim], low[control_dim:2*control_dim]
        damping_high, kp_high = high[:control_dim], high[control_dim:2 * control_dim]
        damping_range = damping_high - damping_low
        kp_range = kp_high - kp_low

        # Get delta values for trajectory
        if controller_name == "OSC_POSE":
            delta = np.array([0, d, 0, 0, 0, 0, 0])
        elif controller_name == "OSC_POSITION":
            delta = np.array([0, d, 0])
        else:   # JOINT_POSITION
            delta = np.array([d, 0, 0, 0, 0, 0, 0])

        # Get total number of steps each test should take (num steps ramping up + num steps ramping down)
        total_steps = num_timesteps_per_change / percent_increase * 2

        # Run a test for both kp and damping
        gains = ["kp", "damping"]

        for gain in gains:

            # Reset the environment
            env.reset()

            # Hardcode the starting position for sawyer
            init_qpos = [-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628]
            env.robots[0].set_robot_joint_positions(init_qpos)
            env.robots[0].controller.update_initial_joints(init_qpos)

            # Notify user a new test is beginning
            print("\nTesting controller {} while sweeping {}...".format(controller_name, gain))

            # If rendering, set controller to front view to get best angle for viewing robot movements
            if args.render:
                env.viewer.set_camera(camera_id=0)

            # Keep track of relative changes in robot eef position
            last_pos = env.robots[0]._right_hand_pos

            # Initialize gains
            if gain == "kp":
                kp = kp_low
                damping = damping_default * np.ones(control_dim)
                gain_val = kp # alias for kp
                gain_range = kp_range
            else:   # "damping"
                kp = kp_default * np.ones(control_dim)
                damping = damping_low
                gain_val = damping # alias for damping
                gain_range = damping_range

            # Initialize counters
            i = 0
            sign = 1.0  # Whether to increase or decrease gain

            # Run trajectory until the threshold condition is met
            while i < total_steps:
                # Create action (damping, kp, traj, gripper)
                action = np.concatenate([damping, kp, sign*delta, [0]])

                # Take an environment step
                env.step(action)
                if args.render:
                    env.render()

                # Update the current change in state
                cur_pos = env.robots[0]._right_hand_pos

                # If we're at the end of the increase, switch direction of traj and gain changes
                if i == int(num_timesteps_per_change / percent_increase):
                    sign *= -1.0

                # Update gain if this is a changing step
                if i % num_timesteps_per_change == 0:
                    # Compare delta, print out to user, and update last_pos
                    delta_pos = np.linalg.norm(cur_pos - last_pos)
                    print("    Magnitude eef distance change with {} = {}: {:.5f}".format(
                        gain, gain_val[0], delta_pos
                    ))
                    last_pos = cur_pos
                    # Update gain
                    gain_val += percent_increase * gain_range * sign

                # Update timestep count
                i += 1

            # When finished, print out the timestep results
            print("Completed trajectory.")

            # Shut down this env before starting the next test
            env.close()

    # Tests completed!
    print()
    print("-" * 80)
    print("All variable impedance testing completed.\n")


if __name__ == "__main__":
    test_linear_interpolator()
