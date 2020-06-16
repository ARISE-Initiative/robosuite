"""
Useful script to make a small video of 100 resets of a given environment.
"""

import numpy as np
import imageio
from tqdm import tqdm

import robosuite as suite

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
        k = 0
        print("Input is not valid. Use {} by default.\n".format(envs[k]))

    # initialize the task
    env = suite.make(
        envs[k],
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
        eval_mode=True,
        perturb_evals=False,
    )

    # write a video
    video_writer = imageio.get_writer("reset_{}.mp4".format(envs[k]), fps=5)

    for _ in tqdm(range(32)):
        env.reset()
        video_img = np.array(env.sim.render(height=512, width=512, camera_name='agentview')[::-1])
        env.step(np.zeros(env.action_spec[0].shape[0]))
        video_writer.append_data(video_img)
    video_writer.close()