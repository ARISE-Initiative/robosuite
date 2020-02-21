import numpy as np
import robosuite as suite
from robosuite.wrappers import IKWrapper

if __name__ == "__main__":

    # get the list of all environments
    envs = sorted(suite.environments.ALL_ENVS)

    # print info and select an environment
    print("Welcome to Surreal Robotics Suite v{}!".format(suite.__version__))
    print(suite.__logo__)
    print("Here is a list of environments in the suite:\n")

    # for k, env in enumerate(envs):
    #     print("[{}] {}".format(k, env))
    k = 22 # manually select k to be Sawyer lift
    print("[{}] {}".format(k, envs[k]))
    # print()
    # try:
    #     s = input(
    #         "Choose an environment to run "
    #         + "(enter a number from 0 to {}): ".format(len(envs) - 1)
    #     )
    #     # parse input into a number within range
    #     k = min(max(int(s), 0), len(envs))
    # except:
    #     print("Input is not valid. Use 0 by default.")
    #     k = 0

    # initialize the task
    env = suite.make(
        envs[k],
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
    )
    env.reset()
    env.viewer.set_camera(camera_id=0)

    # do visualization
    controller = IKWrapper(env, action_repeat=1)
    # controller.reset() # will accidentally change camera???
    target_pixel = [236.67058215/1.84898904, 387.85022096/1.84898904,   1.84898904] # u, v, w
    u, v, w = target_pixel
    target_coord = env.from_pixel_to_world(u, v, w)
    target_body_part_name = 'right_gripper'
    pos0 = env.sim.data.get_body_xpos(target_body_part_name)
    # create trajectory from x0, y0, z0 to target coord in 100 steps
    traj = pos0 + np.linspace(0, 1, 500).reshape((-1, 1)) @ (target_coord - pos0).reshape((1, -1))
    # from IPython import embed
    # embed()
    for i in range(500):
        delta = traj[i] - env.sim.data.get_body_xpos(target_body_part_name)
        print(i, delta)
        controller.step(np.hstack((delta, [0, 0, 0, 1, 1]))) # pos, quat, gripper
        # action = np.random.randn(env.dof)
        # obs, reward, done, _ = env.step(action)
        env.render()