"""NViSII rendering demo.
This script shows how to render a robosuite environment using NViSII, a ray tracing based
renderer. Note that the example script below uses random actions to move the robot. 
Example:
    $ python nvisii_demo_script.py
"""
import sys
import numpy as np
import robosuite as suite

sys.path.append('../renderers/nvisii/')

from nvisii_render_wrapper import NViSIIWrapper

if __name__ == '__main__':

    # Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
    #                          PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal, 
    #                          PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    # Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
    # NViSII rendering currently only has support for the Panda and Sawyer robots

    # Create the NViSII wrpper object
    env = NViSIIWrapper(
        env = suite.make(
                "TwoArmPegInHole",                # environent name
                robots = ["Panda", "Panda"],
                reward_shaping=True,
                has_renderer=False,           # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,          # use object-centric feature
                use_camera_obs=False,         # no camera observations
                control_freq=10, 
            ),
        img_path='images', # path where images are stored
        spp=256,           # samples per pixel for images
        use_noise=False,   # add noise to the images
        debug_mode=False,  # interactive setting
    )

    env.reset() # resets the environment

    image_count = 1

    for i in range(500):

        action = np.random.randn(14) # choose a random action
        obs, reward, done, info = env.step(action) # take a step for the environment

        # renders an image every 100 steps (can be changed accordingly)
        if i%100 == 0:
            env.render(render_type = "png")
            print('Rendering image ... ' + str(image_count))
            image_count += 1

    env.close() # close the environment

    print('Done.')