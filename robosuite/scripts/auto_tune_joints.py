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
c = plt.cm.jet(np.linspace(0,1,8))
"""
# JRH summary of Ziegler-Nichols method
1) set damping_ratio and kp to zero
2) increase kp slowly until you get "overshoot", (positive ERR on first half of "diffs", negative ERR on second half of "diffs"
3) increase damping ratio to squash the overshoot
4) watch and make sure you aren't "railing" the torque values on the big diffs (30.5 for the first 4 joints on Jaco). If this is happening, you may need to decrease the step size (min_max)
I want it to slightly undershoot the biggest joint_diff in 1 step in a tuned controller
All others should have ~.000x error
"""
import json
import os

def run_test():
    horizon = 300
    #controller_config = robosuite.load_controller_config(default_controller='OSC_POSE')

    #controller_config = robosuite.load_controller_config(custom_fpath = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/jaco_osc_pose_10hz.json'))
    controller_config = robosuite.load_controller_config(custom_fpath = os.path.join(os.path.dirname(__file__), '..', 'controllers/config/jaco_osc_pose_1hz.json'))
    robot_name = args.robot_name
    env = robosuite.make("Lift", robots=robot_name,
                         has_renderer=False,        
                         has_offscreen_renderer=True, 
                         ignore_done=True, 
                         use_camera_obs=True,
                         use_object_obs=False,
                         camera_names='frontview',
                         controller_configs=controller_config, 
                         control_freq=1, 
                         horizon=horizon)
    active_robot = env.robots[0]
    init_qpos = deepcopy(active_robot.init_qpos)
    print("before, initial", active_robot.controller.initial_joint)
    env.robots[0].controller.update_initial_joints(init_qpos)
    print("after, initial", active_robot.controller.initial_joint)
    o = env.reset()
    print("after, initial", active_robot.controller.initial_joint)
    positions = []
    orientations = []
    target_positions = []
    target_orientations = []
    joint_torques = []
    frames = []
    action_size = active_robot.controller.control_dim + 1
    action_array = np.zeros((horizon, action_size) )
    null_action = np.zeros(action_size)
    step_distance = .2
    max_step_size = np.abs(env.action_spec[0][0])
 
    eef_pos = o['robot0_eef_pos']
    eef_quat = o['robot0_eef_quat']
    target_position = deepcopy(eef_pos)
    targets = []
    new_target = deepcopy(target_position)
    step_cnt = 0
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 1] = max_step_size
        new_target += action_array[step_cnt, :3] 
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 1] = -max_step_size
        new_target += action_array[step_cnt, :3] 
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1

    new_target = deepcopy(target_position)
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 0] = max_step_size
        new_target += action_array[step_cnt, :3] 
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        step_cnt += 1
        targets.append(deepcopy(new_target))
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 0] = -max_step_size
        new_target += action_array[step_cnt, :3] 
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1

 
    new_target = deepcopy(target_position)
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 2] = max_step_size
        new_target += action_array[step_cnt, :3] 
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in np.arange(0, step_distance, max_step_size):
        action_array[step_cnt, 2] = -max_step_size
        new_target += action_array[step_cnt, :3] 
        targets.append(deepcopy(new_target))
        step_cnt += 1
    for x in range(10):
        targets.append(deepcopy(new_target))
        step_cnt += 1
 
    action_array = action_array[:step_cnt]
    num_action_steps = action_array.shape[0]

    prev_eef = deepcopy(eef_pos)
    plt.figure()
    plt.plot(action_array)
    plt.savefig('action.png')
    plt.close()
    print('init pos', init_qpos)
    active_robot = env.robots[0]
    for i in range(action_array.shape[0]):
        action = list(targets[i]-prev_eef) + [0, 0, 0, 0]
        target_position = prev_eef + action[:3]
        target_positions.append(target_position)
        o,r,done,_ = env.step(action)
        joint_torques.append(active_robot.torques)
        eef_pos = o['robot0_eef_pos']
        positions.append(eef_pos)
        eef_quat = o['robot0_eef_quat']
        orientations.append(eef_quat)
        frames.append(o['frontview_image'][::-1])
        error = target_position-eef_pos
        prev_eef = deepcopy(eef_pos)
    mimwrite(args.movie_file, frames)

    plt.figure()
    target_positions = np.array(target_positions)
    for d in range(target_positions.shape[1]):
        plt.plot(target_positions[:,d], linestyle='--', c=c[d], label=str(d)+' step target')
        plt.plot(np.array(positions)[:,d], c=c[d])
        plt.plot(np.array(targets)[:,d], linestyle=':', c=c[d], label=str(d)+' ideal target')
    plt.legend()
    plt.savefig('pos.png')
    plt.close()

    plt.figure()
    for d in range(target_positions.shape[1]):
        err = target_positions[:,d]-np.array(positions)[:,d]
        plt.plot(err, c=c[d], label=d)
    plt.legend()
    plt.savefig('error.png')
    plt.close()


    plt.figure()
    joint_torques = np.array(joint_torques)
    for d in range(joint_torques.shape[1]):
        plt.plot(joint_torques[:,d], label=d, c=c[d] )
    plt.legend()
    plt.savefig('torques.png')
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

