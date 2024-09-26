"""Use ACT policy to eval can pick and place.

"""
import pickle
import argparse
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
import jax
from octo.model.octo_model import OctoModel
from functools import partial
from octo.utils.train_callbacks import supply_rng
import cv2


def get_image(obs, cam_name, resize_height, resize_width, old_image):
    img = cv2.resize(np.array(obs[cam_name + "_image"]), (resize_height, resize_width))
    img = cv2.rotate(img, cv2.ROTATE_180)
    if old_image.size == 0:
        return np.repeat(img[np.newaxis, :, :, :], 2, axis=0)
    else:
        return np.stack([old_image, img], axis=0)


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
    
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")

    stats = model.dataset_statistics['taco_play']['action']


    pre_process = lambda s_qpos: (s_qpos - stats['mean']) / stats['std']

    # Reset the environment
    obs = env.reset()
    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["taco_play"]["action"],
        ),
    )
    task = model.create_tasks(texts=["pick up the can"])
    old_wrist_img = np.array([])
    old_primary_img = np.array([])

    for t in range(400):
        model_observations = dict()
        model_observations['timestep_pad_mask'] = np.array([True, True])
        model_observations['image_primary'] = get_image(obs, 'frontview', 256, 256, old_primary_img)
        model_observations['image_wrist'] = get_image(obs, 'robot0_eye_in_hand', 128, 128, old_wrist_img)


        old_primary_img = model_observations['image_primary'][1]
        old_wrist_img = model_observations['image_wrist'][1]

        model_observations = jax.tree_map(lambda x: x[None], model_observations)

        # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
        actions = model.sample_actions(
            model_observations, 
            task, 
            unnormalization_statistics=model.dataset_statistics["taco_play"]["action"], 
            rng=jax.random.PRNGKey(0)
        )
        actions = actions[0][0] # remove batch dim and get first action
        obs, reward, done, info = env.step(actions)
        
        env.render()
    print("End of episode")
