"""pygame rendering demo.

This script provides an example of using the pygame library for rendering
camera observations as an alternative to the default mujoco_py renderer.
This is useful for running robosuite on operating systems where mujoco_py is incompatible.

Example:
    $ python demo_pygame_renderer.py --environment Stack --width 1000 --height 1000
"""

import argparse
import sys

import numpy as np
import pygame

import robosuite

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--camera", type=str, default="frontview", help="Name of camera to render")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    args = parser.parse_args()

    width = args.width
    height = args.height
    screen = pygame.display.set_mode((width, height))

    env = robosuite.make(
        args.environment,
        robots=args.robots,
        has_renderer=False,
        ignore_done=True,
        camera_names=args.camera,
        camera_heights=height,
        camera_widths=width,
        use_camera_obs=True,
        use_object_obs=False,
    )

    for i in range(args.timesteps):

        # issue random actions
        action = np.random.randn(env.robots[0].dof)
        obs, reward, done, info = env.step(action)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # read camera observation
        im = np.flip(obs[args.camera + "_image"].transpose((1, 0, 2)), 1)
        pygame.pixelcopy.array_to_surface(screen, im)
        pygame.display.update()

        if i % 100 == 0:
            print("step #{}".format(i))

        if done:
            break
