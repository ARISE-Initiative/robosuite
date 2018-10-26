"""pygame rendering demo.

This script provides an example of using the pygame library for rendering
camera observations as an alternative to the default mujoco_py renderer.

Example:
    $ python run_pygame_renderer.py --environment BaxterPegInHole --width 1000 --height 1000
"""

import sys
import argparse
import pygame
import numpy as np

import robosuite


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="BaxterLift")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()

    width = args.width
    height = args.height
    screen = pygame.display.set_mode((width, height))

    env = robosuite.make(
        args.environment,
        has_renderer=False,
        ignore_done=True,
        camera_height=height,
        camera_width=width,
        show_gripper_visualization=True,
        use_camera_obs=True,
        use_object_obs=False,
    )

    for i in range(args.timesteps):

        # issue random actions
        action = 0.5 * np.random.randn(env.dof)
        obs, reward, done, info = env.step(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # read camera observation
        im = np.flip(obs["image"].transpose((1, 0, 2)), 1)
        pygame.pixelcopy.array_to_surface(screen, im)
        pygame.display.update()

        if i % 100 == 0:
            print("step #{}".format(i))

        if done:
            break
