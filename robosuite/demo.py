import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config


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
        print("[{}] {}".format(j, controllers_info[c]))
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

    # Load the desired controller
    config = load_controller_config(default_controller=controllers[j])

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
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
