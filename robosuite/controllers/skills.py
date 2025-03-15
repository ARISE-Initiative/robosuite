import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.env_utils import get_eef_pos, get_eef_quat, get_axisangle_error
from robosuite.utils.primitive_utils import inverse_scale_action

class BaseSkill:
    def __init__(self,
                 env,
                 image_obs_in_info,
                 aff_type,
                 render,
                 global_xyz_bounds,
                 delta_xyz_scale,
                 yaw_bounds,
                 lift_height,
                 lift_thres,
                 reach_thres,
                 push_thres,
                 aff_thres,
                 yaw_thres,
                 binary_gripper,
                 controller_type,
                 **config
                 ):
        self._env = env

        self._num_ac_calls = None
        self._params = None
        self._state = None
        self._normalize_pos_params = None
        self.skill_obs_list = []
        self.skill_image_obs_list = []

        self._config = dict(
            global_xyz_bounds=global_xyz_bounds,
            delta_xyz_scale=delta_xyz_scale,
            yaw_bounds=yaw_bounds,
            lift_height=lift_height,
            lift_thres=lift_thres,
            reach_thres=reach_thres,
            push_thres=push_thres,
            yaw_thres=yaw_thres,
            aff_thres=aff_thres,
            aff_type=aff_type,
            binary_gripper=binary_gripper,
            image_obs_in_info=image_obs_in_info,
            render=render,
            controller_type=controller_type,
            **config,
        )

        if env is not None:
            for k in ['global_xyz_bounds', 'delta_xyz_scale']:
                assert self._config[k] is not None
                self._config[k] = np.array(self._config[k])

        assert self._config['aff_type'] in [None, 'sparse', 'dense']

    def get_param_dim(self):
        raise NotImplementedError

    def get_param_spec(self):
        param_dim = self.get_param_dim()
        low = np.ones(param_dim) * -1.
        high = np.ones(param_dim) * 1.
        return low, high

    def _check_params_dim(self):
        if self._params is None:
            assert self.get_param_dim() == 0
        else:
            assert len(self._params) == self.get_param_dim()

    def _update_state(self):
        raise NotImplementedError

    def get_aff_centers(self):
        raise NotImplementedError

    def get_aff_noise_scale(self):
        return self._config['aff_noise_scale'] if 'aff_noise_scale' in self._config else None

    def _get_reach_pos(self):
        raise NotImplementedError

    def _reset(self, params, norm):
        self._params = np.array(params).copy()
        self._normalize_pos_params = norm
        self._num_ac_calls = 0
        self._state = None

    def _get_pos_ac(self):
        raise NotImplementedError

    def _get_ori_ac(self):
        raise NotImplementedError

    def _get_gripper_ac(self):
        raise NotImplementedError

    def _get_binary_gripper_ac(self, gripper_action):
        if np.abs(gripper_action) < 0.10:
            gripper_action[:] = 0
        elif gripper_action < 0:
            gripper_action[:] = -1
        else:
            gripper_action[:] = 1
        return gripper_action

    def get_max_ac_calls(self):
        return self._config['max_ac_calls']

    def _get_unnormalized_params(self, params, bounds):
        params = params.copy()
        params = np.clip(params, -1, 1)
        params = (params + 1) / 2
        low, high = bounds[0], bounds[1]
        return low + (high - low) * params

    def _get_normalized_params(self, params, bounds):
        params = params.copy()
        low, high = bounds[0], bounds[1]
        params = (params - low) / (high - low)
        params = params * 2 - 1
        params = np.clip(params, -1, 1)
        return params

    def is_success(self):
        raise NotImplementedError

    def skill_done(self):
        return self.is_success() or (self._num_ac_calls > self._config['max_ac_calls'])
    
    def _update_info(self, info):
        info['num_ac_calls'] = self._num_ac_calls
        info['skill_success'] = self.is_success()
        info['env_success'] = self._env.env._check_success()

    def _reached_goal_ori_y(self):
        if not self._config['use_ori_params']:
            return True
        obs = self._env.get_observation()
        cur_quat = get_eef_quat(obs)
        cur_y = T.mat2euler(T.quat2mat(cur_quat), axes='rxyz')[-1:]
        target_y = self._get_ori_ac()
        if target_y is None:
            return True
        target_y = target_y.copy()
        ee_yaw_diff = np.minimum(
            (cur_y - target_y) % (2 * np.pi),
            (target_y - cur_y) % (2 * np.pi)
        )
        return ee_yaw_diff[-1] <= self._config['yaw_thres']

    def _get_info(self):
        info = self._env.env._get_skill_info()
        return info

    def _get_action(self):
        self._update_state()
        self._num_ac_calls += 1

    def act(self, params, norm):
        self._reset(params, norm)
        image_obs = []
        reward_sum = 0
        obs_list = []
        state_list = []
        action_list = []
        obs = self._env.get_observation()
        state = self._env.get_state()
        obs_list.append(obs)
        state_list.append(state["states"])
        while True:
            action = self._get_action()
            action_list.append(action)
            obs, reward, done, info = self._env.step(action)
            obs_list.append(obs)
            state_list.append(self._env.get_state()["states"])
            info['last_gripper_ac'] = action[-1:]
            if self._config['render']:
                self._env.render()
            reward_sum += reward
            if self._config['image_obs_in_info']:
                image_obs.append(self._env.render(mode="rgb_array", height=256, width=256, camera_name='agentview'))

            if self.skill_done() or done:
                self._env_done = done
                break

        if self._config['image_obs_in_info']:
            info['image_obs'] = image_obs
        info['obs_list'] = obs_list
        info['state_list'] = state_list
        info['action_list'] = action_list
        info['env_done'] = done
        self._update_info(info)
        return dict(obs=obs, info=info)

    def check_interesting_interaction(self):
        assert self.skill_done() or self._env_done
        return True

    def _check_grasp(self, obj=None):
        info = self._get_info()
        if info is not None:
            return np.any(info['grasped_obj'])
        return self._env.env._check_grasp(gripper=self._env.env.robots[0].gripper, object_geoms=obj)

class AtomicSkill(BaseSkill):
    def __init__(self,
                 use_ori_params,
                 max_ac_calls,
                 **config
                 ):
        super().__init__(
            max_ac_calls=max_ac_calls,
            use_ori_params=use_ori_params,
            **config
        )

    def get_param_dim(self):
        # return 7
        if self._config['controller_type'] == 'OSC_POSITION':
            return 4
        elif self._config['controller_type'] in ['OSC_POSE', 'OSC_POSITION_YAW']:
            return 7
        raise NotImplementedError
        # else:
        #     raise NotImplementedError
        # if self._config['use_ori_params']:
        #     return 5
        # else:
        #     return 4

    def _update_state(self):
        self._state = None

    def _get_gripper_ac(self):
        self._check_params_dim()
        gripper_action = self._params[-1:].copy()
        if self._config['binary_gripper']:
            gripper_action = self._get_binary_gripper_ac(gripper_action)
        return gripper_action

    def is_success(self):
        return True

    def _get_action(self):
        self._check_params_dim()
        super()._get_action()
        gripper_action = self._get_gripper_ac()
        return np.concatenate([self._params[:-1], gripper_action])

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        return True

    def _test_start_state(self):
        return True

class GripperSkill(BaseSkill):
    def __init__(self,
                 max_ac_calls,
                 skill_type,
                 **config
                 ):
        super().__init__(
            max_ac_calls=max_ac_calls,
            _use_ori_params=False,
            **config
        )
        self._skill_type = skill_type

    def get_param_dim(self):
        self.pos_dim = None
        self.orn_dim = None
        self.gripper_dim = None
        return 0

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self._num_gripper_steps = 0

    def _update_state(self):
        self._state = None
        self._num_gripper_steps += 1

    def _get_pos_ac(self):
        return None

    def _get_ori_ac(self):
        return None

    def _get_reach_pos(self):
        obs = self._env.get_observation()
        eef_pos = get_eef_pos(obs)
        return eef_pos

    def _get_gripper_ac(self):
        if self._skill_type in ['close']:
            gripper_action = np.array([1, ])
        elif self._skill_type in ['open']:
            gripper_action = np.array([-1, ])
        else:
            raise ValueError

        return gripper_action

    def is_success(self):
        return self._num_ac_calls >= self._config['max_ac_calls']

    def get_aff_centers(self):
        return None

    def _get_action(self):
        super()._get_action()
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            if self._config['controller_type'] == 'OSC_POSE':
                return np.concatenate([[0, 0, 0, 0, 0, 0], gripper_action])
            return np.concatenate([[0, 0, 0, 0], gripper_action])
        return np.concatenate([[0, 0, 0], gripper_action])

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        return True

    def _test_start_state(self):
        for obj in self._env.env.pnp_objs:
            if self._check_grasp(obj):
                return True
        return False

class ReachSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED']

    def __init__(self,
                 max_ac_calls,
                 use_gripper_params,
                 use_ori_params,
                 **config):
        super().__init__(
            use_gripper_params=use_gripper_params,
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            **config
        )

    def get_param_dim(self):
        param_dim = 3
        self.pos_dim = (0, 3)
        self.orn_dim = None
        self.gripper_dim = None
        if self._config['use_ori_params']:
            param_dim += 1
            self.orn_dim = (3, 4)
        if self._config['use_gripper_params']:
            if self.orn_dim is not None:
                self.gripper_dim = (4, 5)
            else:
                self.gripper_dim = (3, 4)
            param_dim += 1
        return param_dim

    def _update_state(self):
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        th = self._config['reach_thres']
        lift_th = self._config['lift_thres']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - lift_th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < lift_th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori_y = self._reached_goal_ori_y()

        if reached_xyz and reached_ori_y:
            self._state = 'REACHED'
        else:
            if reached_xy and reached_ori_y:
                self._state = 'HOVERING'
            else:
                if reached_lift:
                    self._state = 'LIFTED'
                else:
                    self._state = 'INIT'
        assert self._state in ReachSkill.STATES

    def _get_pos_ac(self):
        self._check_params_dim()
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state in ['HOVERING', 'REACHED']:
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        # if self._state == 'INIT':
        #     return None
        param_y = self._params[3:4].copy()
        ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        return ori_y

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self.initial_grasped = False
        for obj in self._env.env.pnp_objs:
            if self._check_grasp(obj):
                self.initial_grasped = True

    def _get_reach_pos(self):
        if self._normalize_pos_params:
            pos = self._get_unnormalized_params(
                self._params[:3], self._config['global_xyz_bounds']
            )
        else:
            pos = self._params[:3]
        # if pos[0] > 0.08:
        #     pos[2] = min(0.85, pos[2])
        return pos

    def _get_gripper_ac(self):
        self._check_params_dim()
        if self._config['use_gripper_params']:
            gripper_action = self._params[-1:].copy()
            if self._config['binary_gripper']:
                gripper_action = self._get_binary_gripper_ac(gripper_action)
            return gripper_action
        return np.array([0, ])

    def is_success(self):
        return self._state == 'REACHED'

    def get_aff_centers(self):
        info = self._get_info()
        if info is None:
            return None
        aff_centers = info.get('reach_pos', None)
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        pos = self._get_pos_ac()
        pos_action = pos - cur_pos
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            target_y = self._get_ori_ac()
            if target_y is None:
                ori_action = np.array([0, 0, 0])
            else:
                target_euler = np.concatenate([[np.pi, 0], target_y])
                target_quat = T.mat2quat(T.euler2mat(target_euler))
                cur_quat = get_eef_quat(obs)
                ori_action = get_axisangle_error(cur_quat, target_quat)

            if self._config['controller_type'] == 'OSC_POSE':
                action = np.concatenate([pos_action, ori_action, gripper_action])
            else:
                action = np.concatenate([pos_action, ori_action[2:], gripper_action])
        else:
            action = np.concatenate([pos_action, gripper_action])

        action = inverse_scale_action(self._env, action)
        return action

    def _test_start_state(self):
        return True

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        end_grasped = False
        for obj in self._env.env.pnp_objs:
            if self._check_grasp(obj):
                end_grasped = True
        if (self.initial_grasped and (not end_grasped)) or ((not self.initial_grasped) and end_grasped):
            return False
        return True

class GraspSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'GRASPED']

    def __init__(self,
                 max_ac_calls,
                 max_reach_steps,
                 max_grasp_steps,
                 use_ori_params,
                 **config
                 ):
        super().__init__(
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            max_reach_steps=max_reach_steps,
            max_grasp_steps=max_grasp_steps,
            **config
        )
        self._num_reach_steps = None
        self._num_grasp_steps = None

    def get_param_dim(self):
        self.pos_dim = (0, 3)
        self.orn_dim = None
        self.gripper_dim = None
        if self._config['use_ori_params']:
            self.orn_dim = (3, 4)
            return 4
        else:
            return 3

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self._num_reach_steps = 0
        self._num_grasp_steps = 0
        self._skill_is_success = True

    def _get_reach_pos(self):
        if self._normalize_pos_params:
            pos = self._get_unnormalized_params(
                self._params[:3], self._config['global_xyz_bounds']
            )
        else:
            pos = self._params[:3]
        # if pos[0] > 0.08:
        #     pos[2] = min(0.85, pos[2])
        return pos

    def _update_state(self):
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        th = self._config['reach_thres']
        lift_th = self._config['lift_thres']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - lift_th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < lift_th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori_y = self._reached_goal_ori_y()

        if self._state == 'GRASPED' \
                or self._num_grasp_steps >= self._config['max_grasp_steps']:
            self._state = 'GRASPED'
            self._num_grasp_steps += 1
        elif self._state == 'REACHED' or (reached_xyz and reached_ori_y) or \
                self._num_reach_steps >= self._config['max_reach_steps']:
            if (self._state != 'REACHED' or not (reached_xyz and reached_ori_y)) \
                    and self._num_reach_steps >= self._config['max_reach_steps']:
                self._skill_is_success = False
            self._state = 'REACHED'
            self._num_grasp_steps += 1
        elif reached_xy and reached_ori_y:
            self._state = 'HOVERING'
            self._num_reach_steps += 1
        elif reached_lift:
            self._state = 'LIFTED'
            self._num_reach_steps += 1
        else:
            self._state = 'INIT'
            self._num_reach_steps += 1

        assert self._state in GraspSkill.STATES

    def _get_pos_ac(self):
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'HOVERING':
            pos = goal_pos.copy()
        elif self._state == 'REACHED':
            pos = goal_pos.copy()
        elif self._state == 'GRASPED':
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        # if self._state == 'INIT':
        #     return None
        param_y = self._params[3:4].copy()
        ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        return ori_y

    def _get_gripper_ac(self):
        if self._state in ['GRASPED', 'REACHED']:
            gripper_action = np.array([1, ])
        else:
            gripper_action = np.array([-1, ])
        return gripper_action

    def skill_done(self):
        return self._num_grasp_steps > self._config['max_grasp_steps']

    def is_success(self):
        return self._skill_is_success and self._state == 'GRASPED'

    def get_aff_centers(self):
        info = self._get_info()
        if info is not None:
            aff_centers = info.get('grasp_pos', None)
        else:
            aff_centers = []
            for obj_id in range(len(self._env.env.pnp_objs)):
                try:
                    obj_size = self._env.env.pnp_objs[obj_id].size
                    if obj_size[0] > 0.04 and obj_size[1] > 0.04:
                        continue
                except:
                    pass
                obj_pos = self._env.env.sim.data.body_xpos[self._env.env.pnp_obj_body_ids[obj_id]].copy()
                aff_centers.append(obj_pos)
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        pos = self._get_pos_ac()
        pos_action = pos - cur_pos
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            target_y = self._get_ori_ac()
            if target_y is None:
                ori_action = np.array([0, 0, 0])
            else:
                target_euler = np.concatenate([[np.pi, 0], target_y])
                target_quat = T.mat2quat(T.euler2mat(target_euler))
                cur_quat = get_eef_quat(obs)
                ori_action = get_axisangle_error(cur_quat, target_quat)
            if self._config['controller_type'] == 'OSC_POSE':
                action = np.concatenate([pos_action, ori_action, gripper_action])
            else:
                action = np.concatenate([pos_action, ori_action[2:], gripper_action])
        else:
            action = np.concatenate([pos_action, gripper_action])

        action = inverse_scale_action(self._env, action)
        return action

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        for obj_id, obj in enumerate(self._env.env.pnp_objs):
            try:
                obj_size = obj.size
                if obj_size[0] > 0.04 and obj_size[1] > 0.04:
                    continue
            except:
                pass
            if self._check_grasp(obj):
                # end_obs = self._env.get_observation()
                # eef_pos = get_eef_pos(end_obs)
                # if np.all(np.abs(eef_pos - self._env.env.sim.data.body_xpos[self._env.env.obj_body_ids[obj_id]]) < obj_size):
                return True
        return False

    def _test_start_state(self):
        if len(self._env.env.pnp_objs) == 0:
            return False
        for obj in self._env.env.pnp_objs:
            if self._check_grasp(obj):
                return False
        return True

class PlaceSkill(BaseSkill):
    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'PLACED']

    def __init__(self,
                 max_ac_calls,
                 max_reach_steps,
                 max_place_steps,
                 use_ori_params,
                 **config
                 ):
        super().__init__(
            use_ori_params=use_ori_params,
            max_ac_calls=max_ac_calls,
            max_reach_steps=max_reach_steps,
            max_place_steps=max_place_steps,
            **config
        )
        self._num_reach_steps = None
        self._num_place_steps = None

    def get_param_dim(self):
        self.pos_dim = (0, 3)
        self.orn_dim = None
        self.gripper_dim = None
        if self._config['use_ori_params']:
            self.orn_dim = (3, 4)
            return 4
        else:
            return 3

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self._num_reach_steps = 0
        self._num_place_steps = 0
        self._initial_grasped_obj_body_id = None
        self._skill_is_success = True
        self._skill_is_interesting = True

    def _get_reach_pos(self):
        if self._normalize_pos_params:
            pos = self._get_unnormalized_params(
                self._params[:3], self._config['global_xyz_bounds']
            )
        else:
            pos = self._params[:3]
        return pos

    def _update_state(self):
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        th = self._config['reach_thres']
        lift_th = self._config['lift_thres']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - lift_th)
        reached_xy = (np.linalg.norm(cur_pos[0:2] - goal_pos[0:2]) < lift_th)
        reached_xyz = (np.linalg.norm(cur_pos - goal_pos) < th)
        reached_ori_y = self._reached_goal_ori_y()

        if self._state == 'PLACED' or self._num_place_steps >= self._config['max_place_steps']:
            self._state = 'PLACED'
            self._num_place_steps += 1
        elif self._state == 'REACHED' or (reached_xyz and reached_ori_y) or self._num_reach_steps >= self._config['max_reach_steps']:
            if (self._state != 'REACHED' or not (reached_xyz and reached_ori_y)) \
                    and self._num_reach_steps >= self._config['max_reach_steps']:
                self._skill_is_success = False
            self._state = 'REACHED'
            self._num_place_steps += 1
        elif reached_xy and reached_ori_y:
            grasped_flag = False
            for obj_id in range(len(self._env.env.pnp_objs)):
                obj = self._env.env.pnp_objs[obj_id]
                if self._check_grasp(obj):
                    grasped_flag = True
                    break
            if not grasped_flag:
                self._skill_is_interesting = False
            self._state = 'HOVERING'
            self._num_reach_steps += 1
        elif reached_lift:
            grasped_flag = False
            for obj_id in range(len(self._env.env.pnp_objs)):
                obj = self._env.env.pnp_objs[obj_id]
                if self._check_grasp(obj):
                    grasped_flag = True
                    break
            if not grasped_flag:
                self._skill_is_interesting = False
            self._state = 'LIFTED'
            self._num_reach_steps += 1
        else:
            grasped_flag = False
            for obj_id in range(len(self._env.env.pnp_objs)):
                obj = self._env.env.pnp_objs[obj_id]
                if self._check_grasp(obj):
                    grasped_flag = True
                    break
            # if not grasped_flag:
            #     self._skill_is_interesting = False
            self._state = 'INIT'
            self._num_reach_steps += 1

        assert self._state in PlaceSkill.STATES

    def _get_pos_ac(self):
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        goal_pos = self._get_reach_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'LIFTED':
            pos = goal_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'HOVERING':
            pos = goal_pos.copy()
        elif self._state == 'REACHED':
            pos = goal_pos.copy()
        elif self._state == 'PLACED':
            pos = goal_pos.copy()
        else:
            raise NotImplementedError

        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        if self._state == 'INIT':
            return None
        param_y = self._params[3:4].copy()
        ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        return ori_y

    def _get_gripper_ac(self):
        if self._state in ['PLACED', 'REACHED']:
            gripper_action = np.array([-1, ])
        else:
            gripper_action = np.array([1, ])
        return gripper_action

    def skill_done(self):
        return self._num_place_steps > self._config['max_place_steps']

    def is_success(self):
        return self._skill_is_success and self._state == 'PLACED'

    def get_aff_centers(self):
        info = self._get_info()
        if info is not None:
            aff_centers = info.get('place_pos', None)
            if aff_centers is None:
                return None
            return np.array(aff_centers, copy=True)
        else:
            return None

    def _get_action(self):
        super()._get_action()
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        pos = self._get_pos_ac()
        pos_action = pos - cur_pos
        if self._num_ac_calls < 4:
            pos_action *= 0.
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            target_y = self._get_ori_ac()
            if target_y is None:
                ori_action = np.array([0, 0, 0])
            else:
                target_euler = np.concatenate([[np.pi, 0], target_y])
                target_quat = T.mat2quat(T.euler2mat(target_euler))
                cur_quat = get_eef_quat(obs)
                ori_action = get_axisangle_error(cur_quat, target_quat)
            if self._config['controller_type'] == 'OSC_POSE':
                action = np.concatenate([pos_action, ori_action, gripper_action])
            else:
                action = np.concatenate([pos_action, ori_action[2:], gripper_action])
        else:
            action = np.concatenate([pos_action, gripper_action])

        action = inverse_scale_action(self._env, action)
        return action

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        end_obs = self._env.get_observation()
        eef_pos = get_eef_pos(end_obs)
        for obj in self._env.env.pnp_objs:
            if self._check_grasp(obj):
                return False
        if not self._skill_is_interesting:
            return False
        return True

    def _test_start_state(self):
        for obj in self._env.env.pnp_objs:
            if self._check_grasp(obj):
                return True
        return False

class PushSkill(BaseSkill):

    STATES = ['INIT', 'LIFTED', 'HOVERING', 'REACHED', 'PUSHED']

    def __init__(self,
                 max_ac_calls,
                 max_reach_steps,
                 max_push_steps,
                 use_ori_params,
                 **config
                 ):
        super().__init__(
            max_ac_calls=max_ac_calls,
            use_ori_params=use_ori_params,
            max_reach_steps=max_reach_steps,
            max_push_steps=max_push_steps,
            **config
        )
        self._num_reach_steps = None
        self._num_push_steps = None

    def get_param_dim(self):
        # no gripper dim in params
        self.pos_dim = (0, 3)
        self.orn_dim = None
        self.gripper_dim = None
        if self._config['use_ori_params']:
            self.orn_dim = (3, 4)
            self.delta_dim = (4, 7)
            return 7
        else:
            self.delta_dim = (3, 6)
            return 6

    def _reset(self, params, norm):
        super()._reset(params, norm)
        self._num_reach_steps = 0
        self._num_push_steps = 0
        self._skill_is_success = True
        self._initial_obj_pos = []
        for obj_id in range(len(self._env.env.push_objs)):
            self._initial_obj_pos.append(self._env.env.sim.data.body_xpos[self._env.env.push_obj_body_ids[obj_id]].copy())

    def _get_reach_pos(self):
        if self._normalize_pos_params:
            push_reach_bounds = np.array(self._config['global_xyz_bounds']).copy()
            if self._config['push_height_thres'] is not None:
                push_reach_bounds[1][2] = push_reach_bounds[0][2] + self._config['push_height_thres']
            pos = self._get_unnormalized_params(
                self._params[:3], push_reach_bounds)
        else:
            pos = self._params[:3]

        # if pos[0] > 0.08:
        #     pos[2] = min(0.85, pos[2])
        return pos

    def _get_push_pos(self):
        src_pos = self._get_reach_pos()
        pos = src_pos.copy()

        delta_pos = self._params[-3:].copy()
        if self._normalize_pos_params:
            delta_pos = np.clip(delta_pos, -1, 1)
            delta_pos *= self._config['delta_xyz_scale']
        pos += delta_pos

        pos = np.clip(pos, self._config['global_xyz_bounds'][0], self._config['global_xyz_bounds'][1])
        # if pos[0] > 0.08:
        #     pos[2] = min(0.85, pos[2])
        return pos

    def _update_state(self):
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        src_pos = self._get_reach_pos()
        target_pos = self._get_push_pos()

        reach_th = self._config['reach_thres']
        lift_th = self._config['lift_thres']
        push_th = self._config['push_thres']
        reached_lift = (cur_pos[2] >= self._config['lift_height'] - lift_th)
        reached_src_xy = (np.linalg.norm(cur_pos[0:2] - src_pos[0:2]) < lift_th)
        reached_src_xyz = (np.linalg.norm(cur_pos - src_pos) < reach_th)
        reached_target_xyz = (np.linalg.norm(cur_pos - target_pos) < push_th)
        reached_ori_y = self._reached_goal_ori_y()

        if self._state == 'REACHED' and reached_target_xyz:
            self._state = 'PUSHED'
            self._num_push_steps += 1
        else:
            if self._state == 'REACHED' or (reached_src_xyz and reached_ori_y) or \
                    self._num_reach_steps >= self._config['max_reach_steps']:
                if (self._state != 'REACHED' or not (reached_src_xyz and reached_ori_y)) \
                        and self._num_reach_steps >= self._config['max_reach_steps']:
                    self._skill_is_success = False
                self._state = 'REACHED'
                self._num_push_steps += 1
            else:
                if reached_src_xy and reached_ori_y:
                    self._state = 'HOVERING'
                    self._num_reach_steps += 1
                    if self._env.env._has_gripper_contact:
                        self._skill_is_success = False
                else:
                    if reached_lift:
                        self._state = 'LIFTED'
                        self._num_reach_steps += 1
                    else:
                        self._state = 'INIT'
                        self._num_reach_steps += 1
            assert self._state in PushSkill.STATES


    def _get_pos_ac(self):
        self._check_params_dim()
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        src_pos = self._get_reach_pos()
        target_pos = self._get_push_pos()

        if self._state == 'INIT':
            pos = cur_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'LIFTED':
            pos = src_pos.copy()
            pos[2] = max(self._config['lift_height'], pos[2])
        elif self._state == 'HOVERING':
            pos = src_pos.copy()
        elif self._state == 'REACHED':
            pos = target_pos.copy()
        elif self._state == 'PUSHED':
            pos = target_pos.copy()
        else:
            raise NotImplementedError
        return pos

    def _get_ori_ac(self):
        self._check_params_dim()
        assert self._config['use_ori_params']
        # if self._state == 'INIT':
        #     return None
        param_y = self._params[3:4].copy()
        ori_y = self._get_unnormalized_params(param_y, self._config['yaw_bounds'])
        return ori_y

    def _get_gripper_ac(self):
        self._check_params_dim()
        gripper_action = np.array([-1, ])
        return gripper_action

    def is_success(self):
        return self._state == 'PUSHED' and self._skill_is_success

    def get_aff_centers(self):
        info = self._get_info()
        if info is not None:
            aff_centers = info.get('push_pos', None)
        else:
            aff_centers = []
            for obj_id in range(len(self._env.env.push_objs)):
                obj_pos = self._env.env.sim.data.body_xpos[self._env.env.push_obj_body_ids[obj_id]].copy()
                aff_centers.append(obj_pos)
        if aff_centers is None:
            return None
        return np.array(aff_centers, copy=True)

    def _get_action(self):
        super()._get_action()
        obs = self._env.get_observation()
        cur_pos = get_eef_pos(obs)
        pos = self._get_pos_ac()
        if self._state == 'REACHED':
            pos_action = (pos - cur_pos) * 1.5
        else:
            pos_action = pos - cur_pos
        # print("action target pos", pos)
        # print(self._state)
        # print("cur pos", cur_pos)
        gripper_action = self._get_gripper_ac()
        if self._config['use_ori_params']:
            target_y = self._get_ori_ac()
            if target_y is None:
                ori_action = np.array([0, 0, 0])
            else:
                target_euler = np.concatenate([[np.pi, 0], target_y])
                target_quat = T.mat2quat(T.euler2mat(target_euler))
                cur_quat = get_eef_quat(obs)
                ori_action = get_axisangle_error(cur_quat, target_quat)
            if self._config['controller_type'] == 'OSC_POSE':
                action = np.concatenate([pos_action, ori_action, gripper_action])
            else:
                action = np.concatenate([pos_action, ori_action[2:], gripper_action])
        else:
            action = np.concatenate([pos_action, gripper_action])
        action = inverse_scale_action(self._env, action)
        return action

    def check_interesting_interaction(self):
        super().check_interesting_interaction()
        for obj_id in range(len(self._env.env.push_objs)):
            try:
                obj_size = self._env.env.push_objs[obj_id].size
                if obj_size[0] < 0.04 and obj_size[1] < 0.04:
                    continue
            except:
                pass
            obj_pos = self._env.env.sim.data.body_xpos[self._env.env.push_obj_body_ids[obj_id]].copy()
            initial_obj_pos = self._initial_obj_pos[obj_id]
            obs = self._env.get_observation()
            eef_pos = get_eef_pos(obs)
            if np.linalg.norm(obj_pos[:2] - initial_obj_pos[:2]) > 0.07:
                return True
        return False

    def _test_start_state(self):
        if len(self._env.env.push_objs) == 0:
            return False
        for obj in self._env.env.pnp_objs:
            if self._check_grasp(obj):
                return False
        return True

