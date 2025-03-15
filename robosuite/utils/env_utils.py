import robosuite as suite
import robosuite.utils.transform_utils as T
import numpy as np

def get_eef_pos(obs):
    return obs['robot0_eef_pos']

def get_eef_quat(obs):
    if obs['robot0_eef_quat'][-1] < 0:
        return - obs['robot0_eef_quat']
    return obs['robot0_eef_quat']

def get_gripper_dist(obs):
    return obs['robot0_gripper_qpos'][0] - obs['robot0_gripper_qpos'][1]

def get_obs(env):
    _is_v1 = (suite.__version__.split(".")[0] == "1")
    if _is_v1:
        assert (int(suite.__version__.split(".")[1]) >= 2), "only support robosuite v0.3 and v1.2+"
    obs = (
        env._get_observations(force_update=True)
        if _is_v1
        else env._get_observation()
    )
    return obs

def stablize_env(env, render=False, write_video=False, video_kwargs=None):
    for _ in range(4):
        obs, _, _, _ = env.step(np.zeros(env.action_spec[0].shape[0]))
        if render:
            env.render()
        if write_video:
            video_kwargs['video_writer'].append_data(obs['agentview' + "_image"])
            video_kwargs['video_count'] += 1

def get_axisangle_error(cur_quat, target_quat):
    cur_orn = T.quat2mat(cur_quat)
    goal_orn = T.quat2mat(target_quat)
    rot_error = goal_orn @ cur_orn.T
    quat_error = T.mat2quat(rot_error)
    axisangle_error = T.quat2axisangle(quat_error)
    return axisangle_error

def get_target_quat(cur_quat, axisangle_error):
    quat_error = T.axisangle2quat(axisangle_error)
    rotation_mat_error = T.quat2mat(quat_error)
    current_orientation = T.quat2mat(cur_quat)
    goal_orientation = np.dot(rotation_mat_error, current_orientation)
    goal_quat = T.mat2quat(goal_orientation)
    return goal_quat