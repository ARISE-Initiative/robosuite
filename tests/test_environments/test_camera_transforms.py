"""
Test script for camera transforms. This test will read the ground-truth 
object state in the Lift environment, transform it into a pixel location
in the camera frame, then transform it back to the world frame, and assert
that the values are close.
"""
import random
import numpy as np

import robosuite
import robosuite.utils.camera_utils as CU
from robosuite.controllers import load_controller_config

def test_camera_transforms():
    # set seeds
    random.seed(0)
    np.random.seed(0)

    camera_name = "agentview"
    camera_height = 120
    camera_width = 120
    env = robosuite.make(
        "Lift",
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names=[camera_name],
        camera_depths=[True],
        camera_heights=[camera_height],
        camera_widths=[camera_width],
        reward_shaping=True,
        control_freq=20,
    )
    sim = env.sim
    obs_dict = env.reset()

    # ground-truth object position
    obj_pos = obs_dict["object-state"][:3]

    # camera frame
    image = obs_dict["{}_image".format(camera_name)][::-1]

    # unnormalized depth map
    depth_map = obs_dict["{}_depth".format(camera_name)][::-1]
    depth_map = CU.get_real_depth_map(sim=sim, depth_map=depth_map)

    # get camera matrices
    world_to_camera = CU.get_camera_transform_matrix(
        sim=sim, 
        camera_name=camera_name, 
        camera_height=camera_height, 
        camera_width=camera_width,
    )
    camera_to_world = np.linalg.inv(world_to_camera)

    # transform object position into camera pixel
    obj_pixel = CU.project_points_from_world_to_camera(
        points=obj_pos, 
        world_to_camera_transform=world_to_camera, 
        camera_height=camera_height, 
        camera_width=camera_width,
    )

    # transform from camera pixel back to world position
    estimated_obj_pos = CU.transform_from_pixels_to_world(
        pixels=obj_pixel, 
        depth_map=depth_map, 
        camera_to_world_transform=camera_to_world,
    )

    # the most we should be off by in the z-direction is 3^0.5 times the maximum half-size of the cube
    max_z_err = np.sqrt(3) * 0.022
    z_err = np.abs(obj_pos[2] - estimated_obj_pos[2]) 
    assert z_err < max_z_err

    print("pixel: {}".format(obj_pixel))
    print("obj pos: {}".format(obj_pos))
    print("estimated obj pos: {}".format(estimated_obj_pos))
    print("z err: {}".format(z_err))


if __name__ == "__main__":

    test_camera_transforms()

