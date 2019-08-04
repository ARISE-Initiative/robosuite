import numpy as np
import robosuite as suite
from robosuite.wrappers import DRWrapper

from PIL import Image

if __name__ == "__main__":

    # get the list of all environments
    envs = sorted(suite.environments.ALL_ENVS)

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
        print("Input is not valid. Use 0 by default.")
        k = 0

    # initialize the task
    env = suite.make(
        envs[k],
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=100,
        camera_name="agentview",
    )
    env = DRWrapper(env)
    env.reset()
    env.viewer.set_camera(camera_id=0)

    save_path_textured = '/Users/aqua/Documents/Workspace/Summer/svl_summer/domain_randomization/samples'

    # do visualization
    for i in range(10000):
        action = np.random.randn(env.dof)
        obs, reward, done, _ = env.step(action)
        env.render()

        # img = obs['image'][::-1]
        # img = Image.fromarray(img).convert('RGB')
        # print(save_path_textured + f'/from_states_{i}')
        # img.save(save_path_textured + f'/rand_sample_{i}.jpg')
