"""Use ACT policy to eval can pick and place.

"""
import pickle
from config.config import POLICY_CONFIG, TRAIN_CONFIG, CHECKPOINT_DIR, device # must import first
import argparse
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from policy import ACTPolicy
import torch
import os

from utils.utils import get_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="agentview",
        camera_names=["robot0_eye_in_hand", "frontview", "birdview"],
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    ckpt_path = os.path.join(CHECKPOINT_DIR, args.environment, TRAIN_CONFIG['eval_ckpt_name'])
    policy = ACTPolicy(POLICY_CONFIG)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(CHECKPOINT_DIR, args.environment, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    camera_names = POLICY_CONFIG['camera_names']
    query_frequency = POLICY_CONFIG['num_queries']

        # Reset the environment
    obs = env.reset()
    all_actions = None

    for t in range(TRAIN_CONFIG['num_epochs']):
        qpos = np.arctan2(obs['robot0_joint_pos_sin'], obs['robot0_joint_pos_cos'])
        grasp = [0]
        if 'grasp' in obs:
            grasp = [obs['grasp']]
        qpos = np.concatenate((qpos, grasp))
        qpos = pre_process(qpos)
        qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)

        with torch.inference_mode():
            if t % query_frequency == 0:
                all_actions = policy(qpos, get_image(obs, camera_names, device))

            cur_action = all_actions[:, t % query_frequency]
            cur_action = cur_action.squeeze(0).cpu().numpy()
            cur_action = post_process(cur_action)

        # If action is none, then this a reset so we should break
        if cur_action is None:
            print('No action')
            break

        obs, reward, done, info = env.step(cur_action)
        
        
        env.render()
    print("End of episode")
