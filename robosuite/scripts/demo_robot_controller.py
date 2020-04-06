"""
This demo script demonstrates the various functionalities of each controller available within robosuite (list of
supported controllers are shown at the bottom of this docstring).

For a given controller, runs through each dimension and executes a perturbation "test_value" from its
neutral (stationary) value for a certain amount of time "steps_per_action", and then returns to all neutral values
for time "steps_per_rest" before proceeding with the next action dim.

    E.g.: Given that the expected action space of the Pos / Ori (EE_POS_ORI) controller (without a gripper) is
    (dx, dy, dz, droll, dpitch, dyaw), the testing sequence of actions over time will be:

        ***START OF DEMO***
        ( dx,  0,  0,  0,  0,  0, grip)     <-- Translation in x-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0, dy,  0,  0,  0,  0, grip)     <-- Translation in y-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0, dz,  0,  0,  0, grip)     <-- Translation in z-direction      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0, dr,  0,  0, grip)     <-- Rotation in roll (x) axis       for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0,  0, dp,  0, grip)     <-- Rotation in pitch (y) axis      for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        (  0,  0,  0,  0,  0, dy, grip)     <-- Rotation in yaw (z) axis        for 'steps_per_action' steps
        (  0,  0,  0,  0,  0,  0, grip)     <-- No movement (pause)             for 'steps_per_rest' steps
        ***END OF DEMO***

    Thus the EE_POS_ORI controller should be expected to sequentially move linearly in the x direction first,
        then the y direction, then the z direction, and then begin sequentially rotating about its x-axis,
        then y-axis, then z-axis.

Please reference the controller README in the robosuite/controllers directory for an overview of each controller.
Controllers are expected to behave in a generally controlled manner, according to their control space. The expected
sequential qualitative behavior during the test is described below for each controller:

* EE_POS_ORI: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the global coordinate frame
* EE_POS: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
* EE_IK: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the local robot end effector frame
* JOINT_IMP: Robot Joints move sequentially in a controlled fashion
* JOINT_VEL: Robot Joints move sequentially in a controlled fashion
* JOINT_TOR: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the
            "controller" is really just a wrapper for direct torque control of the mujoco actuators. Therefore, a
            "neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!

"""


import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import robosuite.utils.transform_utils as T


if __name__ == "__main__":
    # get the list of all environments
    envs = list(suite.environments.ALL_ENVIRONMENTS)

    # get the list of all controllers
    controllers_info = suite.controllers.ALL_CONTROLLERS_INFO
    controllers = list(suite.controllers.ALL_CONTROLLERS)

    # print info and select an environment
    print("Welcome to Surreal Robotics Suite v{}!".format(suite.__version__))
    print(suite.__logo__)
    print("Here is a list of environments in the suite:\n")

    for k, env in enumerate(envs):
        print("[{}] {}".format(k, env))
    print()
    try:
        s = input(
            "Choose an environment to run "
            + "(enter a number from 0 to {}): ".format(len(envs) - 1)
        )
        # parse input into a number within range
        k = min(max(int(s), 0), len(envs))
    except:
        k = 0
        print("Input is not valid. Use {} by default.\n".format(envs[k]))

    print("Here is a list of controllers in the suite:\n")

    for j, c in enumerate(controllers):
        print("[{}] {} - {}".format(j, c, controllers_info[c]))
    print()
    try:
        s = input(
            "Choose a controller for the robot "
            + "(enter a number from 0 to {}): ".format(len(controllers) - 1)
        )
        # parse input into a number within range
        j = min(max(int(s), 0), len(controllers) - 1)
    except:
        j = 0
        print("Input is not valid. Use {} by default.".format(controllers[j]))

    print()
    print("Press \"H\" to show the viewer control panel.")

    # Get chosen controller
    controller_name = controllers[j]

    # Load the desired controller
    config = load_controller_config(default_controller=controller_name)

    # Define the pre-defined controller actions to use (action_dim, num_test_steps, test_value, neutral control values)
    controller_settings = {
        "EE_POS_ORI": [6, 6, 0.1, np.array([0, 0, 0, 0, 0, 0], dtype=float)],
        "EE_POS": [3, 3, 0.1, np.array([0, 0, 0], dtype=float)],
        "EE_IK": [7, 6, 0.01, np.array([0, 0, 0, 0, 0, 0, 1], dtype=float)],
        "JOINT_IMP": [7, 7, 0.2, np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)],
        "JOINT_VEL": [7, 7, -0.05, np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)],
        "JOINT_TOR": [7, 7, 0.001, np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)]
    }

    # Define variables for each controller test
    action_dim = controller_settings[controller_name][0]
    num_test_steps = controller_settings[controller_name][1]
    test_value = controller_settings[controller_name][2]
    neutral = controller_settings[controller_name][3]

    # Define the number of timesteps to use per controller action as well as timesteps in between actions
    steps_per_action = 50
    steps_per_rest = 25

    # initialize the task
    env = suite.make(
        envs[k],
        controller_config=config,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        horizon=(steps_per_action + steps_per_rest) * num_test_steps,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # To accommodate for multi-armed robots (e.g.: Baxter), we need to make sure to fill any extra action space
    # Get total action dimension
    low, _ = env.action_spec
    total_action_dim = len(low)
    filler = np.zeros(total_action_dim - 2 * action_dim) if env.mujoco_robot.name == 'baxter' else \
        np.zeros(total_action_dim - action_dim)

    # Keep track of done variable to know when to break loop
    count = 0
    # Loop through controller space
    while count < num_test_steps:
        action = neutral.copy()
        for i in range(steps_per_action):
            if controller_name == 'ee_ik' and count > 2:
                # Convert from euler angle to quat here since we're working with quats
                angle = np.zeros(3)
                angle[count - 3] = test_value
                action[3:7] = T.mat2quat(T.euler2mat(angle))
            else:
                action[count] = test_value
            total_action = np.concatenate([action, action, filler]) if env.mujoco_robot.name == 'baxter' else \
                np.concatenate([action, filler])
            env.step(total_action)
            env.render()
        for i in range(steps_per_rest):
            total_action = np.concatenate([neutral, neutral, filler]) if env.mujoco_robot.name == 'baxter' else \
                np.concatenate([neutral, filler])
            env.step(total_action)
            env.render()
        count += 1

    # Shut down this env before starting the next test
    env.close()
