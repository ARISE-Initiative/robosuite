import numpy as np
import robosuite as suite
from robosuite.wrappers import IKWrapper
import robosuite.utils.transform_utils as T

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

    # change the controller to osc
    kwargs = {}
    kwargs["controller"] = "position_orientation"
    # kwargs["control_freq"] = 20
    kwargs["interpolation"] = None
    kwargs["use_default_controller_config"] = False
    import RobotTeleop
    from os.path import join as pjoin
    kwargs["controller_config_file"] = pjoin(RobotTeleop.__path__[0],
                                             "assets/osc/robosuite/sawyer.hjson")
    env = suite.make(
        envs[k],
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100,
        camera_name='agentview',
        **kwargs,
    )
    env.reset()
    env.viewer.set_camera(camera_id=2) # watch from agent view

    # do visualization
    controller = IKWrapper(env, action_repeat=10)
    # controller.reset() # will accidentally change camera???
    target_pixel = [ 39.93235, 41, 0.55339] # u, v, w
    u, v, w = target_pixel
    target_coord = env.from_pixel_to_world(u, v, w)
    target_body_part_name = 'right_hand'
    pos0 = env.sim.data.get_body_xpos(target_body_part_name)
    # from IPython import embed
    # embed()
    # create trajectory from x0, y0, z0 to target coord in 100 steps
    traj = pos0 + np.linspace(0, 1, 500).reshape((-1, 1)) @ (target_coord - pos0).reshape((1, -1))
    # from IPython import embed
    # embed()
    for i in range(1500):
        if i >= 500:
            env.render()
            continue
        delta_pos = target_coord - env.sim.data.get_body_xpos(target_body_part_name)
        # print(i, delta_pos)
        current_rotation = np.array(env.sim.data.body_xmat[env.sim.model.body_name2id(target_body_part_name)].reshape([3, 3]))
        rotation = np.eye(3)
        delta_rot_mat = rotation #current_rotation.dot(rotation.T)
        delta_rot_euler = -T.mat2euler(delta_rot_mat)
        action = np.hstack((delta_pos, delta_rot_euler, 1))
        env.step(action)
        # controller.step(np.hstack((delta, [0, 0, 0, 1, 1]))) # pos, quat, gripper
        env.render()

    w_interp = np.linspace(0.55339, 0.2, 1000)
    for w in w_interp:
        target_coord = env.from_pixel_to_world(u, v, w)
        delta_pos = target_coord - env.sim.data.get_body_xpos(target_body_part_name)
        # print(w, delta_pos, "target", target_coord)
        delta_rot_mat = np.eye(3)  # do not rotate
        delta_rot_euler = -T.mat2euler(delta_rot_mat)
        action = np.hstack((delta_pos, delta_rot_euler, 1))
        env.step(action)
        env.render()