import imageio
import numpy as np

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

import robosuite
try:
    import robosuite
    import robosuite.utils.transform_utils as T
except:
    print("WARNING: could not import robosuite")


if __name__ == "__main__":

    video_rollout_path = "./demo_{}.mp4".format(0)
    env_name = "SawyerLiftPosition"
    test_camera_name = 'agentview'

    controller_config = {'control_delta': True,
                           'damping': 1,
                           'force_control': False,
                           'input_max': 1,
                           'input_min': -1,
                           'interpolation': None,
                           'kp': 150,
                           'orientation_limits': None,
                           'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                           'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                           'position_limits': None,
                           'ramp_ratio': 0.2,
                           'type': 'EE_POS_ORI',
                           'uncouple_pos_ori': True}
    env = robosuite.make(
        env_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=False,
        use_camera_obs=True,
        camera_depth=False,
        camera_height=120,  # 240, #84,
        camera_width=160,  # 320, #84,
        camera_name="agentview",
        gripper_visualization=False,
        reward_shaping=True,
        control_freq=20,
        use_indicator_object=False,  # True: debug, set indicator object
        indicator_num=1,
        eval_mode=False,  # if generating dataset, set to False
        controller_config=controller_config
    )
    kp_env = robosuite.make(
        env_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=False,
        use_camera_obs=True,
        camera_depth=False,
        camera_height=120,  # 240, #84,
        camera_width=160,  # 320, #84,
        camera_name="agentview",
        gripper_visualization=False,
        reward_shaping=True,
        control_freq=20,
        use_indicator_object=True,  # True: debug, set indicator object
        indicator_num=1,
        eval_mode=False,  # if generating dataset, set to False
        controller_config=controller_config
    )

    write_video = True
    ob_dict = env.reset()
    env.set_robot_joint_positions([0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
    # make teleop visualization site colors transparent
    env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
    env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    ob_dict = env._get_observation()
    if write_video:  # and video_eval_ind % 9 == 0:
        video_writer = imageio.get_writer(video_rollout_path, fps=20)
        video_skip = 5
        video_count = 0

    gripper = -1.0  # for hard coded policy

    for step in range(200):
        ac = np.array([0., 0., 0., 1.]) # hard_coded_policy(ob)
        ob_dict, r, done, _ = env.step(ac)
        kp_env.step(ac)
        if write_video:
            if video_count % video_skip == 0:
                kp_env.move_indicator(np.array([0.5, 0.01, 0.81]), 0) # this indicator is used to distinguish env and kp_env
                video_img = env.sim.render(height=512, width=512,
                                                    camera_name=test_camera_name)[::-1]  # agentview
                # env.render()
                video_writer.append_data(video_img)
            video_count += 1
# """
# Test script.
# """
# import imageio
# import numpy as np
#
# import robosuite
# import robosuite.utils.transform_utils as T
# from robosuite.controllers import load_controller_config
#
# NUM_RENDERS = 1 # 2
#
# if __name__ == "__main__":
#
#     # envs
#     env_name_1 = "SawyerLift"
#     env_name_2 = "SawyerStack"
#
#     # controller
#     controller = "EE_POS"
#
#     # camera to render from first env
#     test_camera_name = 'agentview'
#
#     # path to output video
#     video_rollout_path = "./test_demo_{}.mp4".format(0)
#
#     controller_config = load_controller_config(default_controller=controller)
#
#     env = robosuite.make(
#         env_name_1,
#         controller_config=controller_config,
#         has_renderer=False,
#         has_offscreen_renderer=True,
#         use_camera_obs=False,
#         control_freq=20,
#         camera_height=128,
#         camera_width=128,
#     )
#     env2 = robosuite.make(
#         env_name_2,
#         controller_config=controller_config,
#         has_renderer=False,
#         has_offscreen_renderer=True,
#         use_camera_obs=True,
#         control_freq=20,
#         camera_height=512,
#         camera_width=512,
#         camera_name='agentview',
#     )
#
#     env.reset()
#     env2.reset()
#
#     video_writer = imageio.get_writer(video_rollout_path, fps=20)
#     video_skip = 5
#     video_count = 0
#
#     low, high = env.action_spec
#     for step in range(200):
#
#         # step each env with a random action
#         action = np.random.uniform(low, high)
#         env.step(action)
#         env2.step(action)
#
#         # render video for first env
#         if video_count % video_skip == 0:
#             for _ in range(NUM_RENDERS):
#                 video_img = env.sim.render(
#                     height=512,
#                     width=512,
#                     camera_name=test_camera_name,
#                 )[::-1]
#             video_writer.append_data(video_img)
#         video_count += 1
