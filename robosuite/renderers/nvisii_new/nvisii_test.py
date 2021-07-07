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
                "Door",
                robots = ["Panda"],
                reward_shaping=True,
                has_renderer=False,           # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,          # use object-centric feature
                use_camera_obs=False,         # no camera observations
                control_freq=10, 
            ),
        img_path='images',
        spp=100,
        use_noise=False,
        debug_mode=False,
    )

    env.reset()

    for i in range(500):
        action = np.random.randn(8)
        obs, reward, done, info = env.step(action)

        if i%100 == 0:
            env.render(render_type="png")
            print('Rendering image... ' + str(i/100 + 1))

    env.close()
    
    print('Done.')