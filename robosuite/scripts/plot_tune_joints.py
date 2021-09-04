import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import robosuite
from robosuite.robots import SingleArm 
import numpy as np
from IPython import embed
from imageio import mimwrite
from copy import deepcopy
import json

from matplotlib import cm
c = plt.cm.jet(np.linspace(0,1,14))
import json
import os
"""
# JRH summary of Ziegler-Nichols method
1) set damping_ratio and kp to zero
2) increase kp slowly until you get "overshoot", (positive ERR on first half of "diffs", neg    ative ERR on second half of "diffs"
3) increase damping ratio to squash the overshoot
"""

def run_test():
    horizon = 100
    min_max = 1
    # with position controller, the arm is always pointed down
    #controller_config = robosuite.load_controller_config(default_controller='OSC_POSE')
    #controller_config = robosuite.load_controller_config(custom_fpath = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/jaco_osc_position.json'))
    controller_config = robosuite.load_controller_config(custom_fpath = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/jaco_osc_pose.json'))

    robot_name = args.robot_name
    env = robosuite.make("Lift", robots=robot_name,
                         has_renderer=False,        
                         has_offscreen_renderer=True, 
                         ignore_done=True, 
                         use_camera_obs=True,
                         use_object_obs=False,
                         camera_names='nearfrontview',
                         controller_configs=controller_config, 
                         control_freq=20, 
                         horizon=horizon)
    o = env.reset()
    positions = []
    orientations = []
    target_positions = []
    target_orientations = []
    joint_torques = []
    frames = []
    active_robot = env.robots[0]
    action_size = active_robot.controller.control_dim + 1
    null_action = np.zeros(action_size)
    action_array = np.zeros((horizon, action_size))
    max_step_size = env.action_spec[0][0]
    # how far to reach
    step_distance = max_step_size * horizon * .2
    step_cnt = 10
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 2] = max_step_size
        step_cnt += 1

    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 1] = max_step_size
        step_cnt += 1

    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 2] = -max_step_size
        step_cnt += 1


    eef_pos = o['robot0_eef_pos']
    eef_quat = o['robot0_eef_quat']
    target_position = deepcopy(eef_pos)
    print("START", target_position)


    for x in range(horizon):
        # warmup actions
        o,r,done,_ = env.step(action_array[x])

        eef_pos = o['robot0_eef_pos']
        eef_quat = o['robot0_eef_quat']
        target_position +=  action_array[x,:3]
        target_positions.append(deepcopy(target_position))
        positions.append(eef_pos)
        frames.append(o['nearfrontview_image'][::-1])
        joint_torques.append(active_robot.torques)

    mimwrite(args.movie_file, frames)

    plt.figure()
    joint_torques = np.array(joint_torques)
    for d in range(joint_torques.shape[1]):
        plt.plot(joint_torques[:,d], label=d, c=c[d] )
    print(joint_torques.max(0))
    plt.legend()
    plt.savefig('torques.png')
    plt.close()

    target_positions = np.array(target_positions)
    plt.figure()
    for d in range(target_positions.shape[1]):
        plt.plot(target_positions[:,d], linestyle='--', c=c[d], label=d)
        plt.plot(np.array(positions)[:,d], c=c[d])
    plt.legend()
    plt.savefig('error.png')
    plt.close()

    plt.figure()
    for d in range(action_array.shape[1]):
        plt.plot(action_array[:,d], linestyle='--', label=d)
    plt.legend()
    plt.savefig('action.png')
    plt.close()

 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_name', default='Jaco')
    parser.add_argument('--active_joint', default=3, type=int, help='joint to test')
    parser.add_argument('--controller_name', default='OSC_POSITION', type=str, help='controller name')
    parser.add_argument('--config_file', default='', help='path to config file. if not configured, will default to ../confgis/robot_name_joint_position.json')
    parser.add_argument('--num_rest_steps', default=3, type=int)
    parser.add_argument('--movie_file', default='tune.mp4')
    args = parser.parse_args() 
    if args.config_file == '':
        args.config_file = '../controllers/config/%s_%s.json'%(args.robot_name.lower(), args.controller_name.lower())
    run_test()

