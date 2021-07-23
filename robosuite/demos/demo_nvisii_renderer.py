import argparse
import numpy as np
import robosuite as suite
from robosuite.renderers.nvisii.nvisii_wrapper import NViSIIWrapper

from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    '''
    Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
                             PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal, 
                             PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    '''
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--video_mode", type=str, default=False)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--spp", type=int, default=256)

    args = parser.parse_args()

    args.video_mode = str2bool(args.video_mode)

    temp = suite.make(
                args.env,
                robots = args.robots,
                reward_shaping=True,
                has_renderer=False,           # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,          # use object-centric feature
                use_camera_obs=False,         # no camera observations
                control_freq=10,
            )

    env = NViSIIWrapper(
        env=temp,
        img_path='images',
        width=args.width,
        height=args.height,
        spp=args.spp,
        use_noise=False,
        debug_mode=False,
        video_mode=args.video_mode,
        verbose=1,
    )

    env.reset()

    action_space = env.action_dim

    for i in range(args.timesteps):
        action = np.random.randn(action_space)
        obs, reward, done, info = env.step(action)

        if args.video_mode:
            env.render(render_type="png")
        else:
            if i % 100 == 0:
                env.render(render_type="png")

    env.close()
    
    print('Done.')
