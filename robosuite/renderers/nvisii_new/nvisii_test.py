import numpy as np
import robosuite as suite
from nvisii_wrapper import NViSIIWrapper

if __name__ == '__main__':

    # Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
    #                          PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal, 
    #                          PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    # Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e

    env = NViSIIWrapper(
        env = suite.make(
                "TwoArmLift",
                robots = ["Baxter"],
                reward_shaping=True,
                has_renderer=False,           # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,          # use object-centric feature
                use_camera_obs=False,         # no camera observations
                control_freq=10,
            ),
        img_path='images',
        spp=512,
        use_noise=False,
        debug_mode=False,
        video_mode=True,
        verbose=1
    )

    env.reset()

    for i in range(600):
        action = np.random.randn(16)
        obs, reward, done, info = env.step(action)

        env.render(render_type="png")

    env.close()
    
    print('Done.')