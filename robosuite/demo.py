import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config


if __name__ == "__main__":
    # get the list of all environments
    envs = sorted(suite.environments.ALL_ENVS)

    # get the list of all controllers
    controllers = {
        "Joint Velocity": "JOINT_VEL",
        "Joint Torque": "JOINT_TOR",
        "Joint Impedance": "JOINT_IMP",
        "End Effector Position": "EE_POS",
        "End Effector Position Orientation": "EE_POS_ORI",
        "End Effector Inverse Kinematics (note: must have pybullet installed!)": "EE_IK",
    }

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

    for j, controller in enumerate(list(controllers)):
        print("[{}] {}".format(j, controller))
    print()
    try:
        s = input(
            "Choose a controller for the robot "
            + "(enter a number from 0 to {}): ".format(len(controllers) - 1)
        )
        # parse input into a number within range
        j = min(max(int(s), 0), len(controllers))
    except:
        j = 0
        print("Input is not valid. Use {} by default.".format(list(controllers)[j]))

    # Load the desired controller
    config = load_controller_config(default_controller=list(controllers.values())[j])

    # initialize the task
    env = suite.make(
        envs[k],
        controller_config=config,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    for i in range(1000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
