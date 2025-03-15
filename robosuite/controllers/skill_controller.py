import collections
import copy
import numpy as np
from collections import OrderedDict

from robosuite.controllers.skills import (
    AtomicSkill,
    ReachSkill,
    GraspSkill,
    PushSkill,
    GripperSkill,
    PlaceSkill,
)

DELTA_XYZ_SCALE = np.array([0.4, 0.4, 0.01])


class SkillController:

    def __init__(self,
                 env,
                 primitive_set=None,
                 controller_type=None,
                 image_obs_in_info=False,
                 aff_type='sparse',
                 render=False,
                 reach_use_gripper=False,
                 env_idx=None,
                 output_mode=None):

        self._env = env
        self._env_idx = env_idx
        self.primitive_set = primitive_set
        self.output_mode = output_mode
        if controller_type in ['OSC_POSE', 'OSC_POSITION_YAW']:
            _use_ori_params = True
        elif controller_type == 'OSC_POSITION':
            _use_ori_params = False
        else:
            raise NotImplementedError
        base_config = dict(
            env=self._env,
            aff_type=aff_type,
            image_obs_in_info=image_obs_in_info,
            render=render,
            use_ori_params=_use_ori_params,
            global_xyz_bounds=self._env.env.eef_bounds if self._env is not None else None,
            delta_xyz_scale=DELTA_XYZ_SCALE,
            yaw_bounds=np.array([
                [-np.pi / 2],
                [np.pi / 2]
            ]),
            lift_height=env.env.table_offset[2] + 0.15 if self._env is not None else None,
            lift_thres=0.02,
            reach_thres=0.01,
            push_thres=0.015,
            aff_thres=0.08,
            yaw_thres=0.20,
            grasp_thres=0.03,
            binary_gripper=False,
            env_idx=env_idx,
            push_height_thres=None,
            controller_type=controller_type
        )

        try:
            base_config["lift_height"] = self._env.env.lift_height
        except:
            pass

        self.atomic = AtomicSkill(
            max_ac_calls=1,
            **base_config
        )

        self.gripper_release = GripperSkill(
            max_ac_calls=10,
            skill_type='open',
            **base_config
        )

        self.place = PlaceSkill(
            max_ac_calls=80,
            max_reach_steps=70,
            max_place_steps=10,
            aff_noise_scale=np.array([0.01, 0.01, 0.1]),
            **base_config
        )

        self.reach = ReachSkill(
            max_ac_calls=70,
            use_gripper_params=reach_use_gripper,
            aff_noise_scale=np.array([0.005, 0.005, 0.01]),
            **base_config
        )

        self.grasp = GraspSkill(
            max_ac_calls=80,
            max_reach_steps=70,
            max_grasp_steps=10,
            aff_noise_scale=np.array([0.001, 0.001, 0.001]),
            **base_config
        )

        self.push = PushSkill(
            max_ac_calls=120,
            max_reach_steps=70,
            max_push_steps=50,
            aff_noise_scale=np.array([0.1, 0.1, 0.005]),
            **base_config
        )

        self.name_to_skill = OrderedDict()

        if 'atomic' in primitive_set:
            self.name_to_skill['atomic'] = self.atomic
        if 'place' in primitive_set:
            self.name_to_skill['place'] = self.place
        if 'reach' in primitive_set:
            self.name_to_skill['reach'] = self.reach
        if 'grasp' in primitive_set:
            self.name_to_skill['grasp'] = self.grasp
        if 'push' in primitive_set:
            self.name_to_skill['push'] = self.push
        if 'gripper_release' in primitive_set:
            self.name_to_skill['gripper_release'] = self.gripper_release


        self.output_dim = 0
        self.primitive_dim_info = {}
        if self.output_mode == 'concat':
            start_idx = 0
            for _p_name in primitive_set:
                _param_dim = self.name_to_skill[_p_name].get_param_dim()
                self.primitive_dim_info[_p_name] = dict(
                    start_idx=start_idx,
                    param_dim=_param_dim,
                )
                self.output_dim += _param_dim
                start_idx += _param_dim
        elif self.output_mode == 'max':
            for _p_name in primitive_set:
                _param_dim = self.name_to_skill[_p_name].get_param_dim()
                self.primitive_dim_info[_p_name] = dict(
                    start_idx=0,
                    param_dim=_param_dim
                )
                self.output_dim = max(self.output_dim, _param_dim)
        else:
            raise NotImplementedError

    def get_skill(self, p_name):
        return self.name_to_skill[p_name]

    def test_start_state(self, p_name):
        return self.name_to_skill[p_name]._test_start_state()

    def reset_skill(self, p_name, skill_args, norm):
        skill = self.name_to_skill[p_name]
        param_dim = skill.get_param_dim()
        assert len(skill_args) == param_dim
        try:
            if norm or p_name == 'atomic':
                try:
                    assert (skill_args <= 1.).all() and (skill_args >= -1.).all()
                except:
                    print("out of bounds:", skill_args)
            else:
                norm_args = self.get_normalized_params(p_name=p_name, unnorm_params=skill_args)
                assert (norm_args <= 1.).all and (skill_args >= -1.).all()
        except:
            print("p", p_name)
            print("args", skill_args)
            raise ValueError
        skill._reset(skill_args, norm)

    def step_action(self, p_name):
        skill = self.name_to_skill[p_name]
        action = skill._get_action()
        return action

    def skill_done(self, p_name):
        skill = self.name_to_skill[p_name]
        return skill.skill_done()

    def skill_success(self, p_name):
        skill = self.name_to_skill[p_name]
        return skill.is_success()

    def skill_interest(self, p_name):
        skill = self.name_to_skill[p_name]
        return skill.check_interesting_interaction()

    def output_to_args(self, p_name, output):
        param_dim_info = self.get_skill_param_dim(p_name)
        assert len(output) == self.output_dim
        return output[param_dim_info['start_idx']: param_dim_info['start_idx'] + param_dim_info['param_dim']]

    def args_to_output(self, p_name, skill_args):
        param_dim_info = self.get_skill_param_dim(p_name)
        assert len(skill_args) == param_dim_info['param_dim']
        output = np.zeros(self.output_dim)
        output[param_dim_info['start_idx']: param_dim_info['start_idx'] + param_dim_info['param_dim']] = skill_args
        return output

    def get_skill_param_dim(self, p_name):
        mask = np.zeros(self.output_dim)
        mask[self.primitive_dim_info[p_name]['start_idx']: \
             self.primitive_dim_info[p_name]['start_idx'] + self.primitive_dim_info[p_name]['param_dim']] = \
            np.ones(self.primitive_dim_info[p_name]['param_dim'])
        return dict(
            start_idx=self.primitive_dim_info[p_name]['start_idx'],
            param_dim=self.primitive_dim_info[p_name]['param_dim'],
            output_dim=self.output_dim,
            mask=mask
        )

    def execute(self, p_name, skill_args, norm, **kwargs):
        # len(args) = maximal argument length
        skill = self.name_to_skill[p_name]
        param_dim = skill.get_param_dim()
        assert len(skill_args) == param_dim
        try:
            if norm or p_name == 'atomic':
                assert (skill_args <= 1.).all() and (skill_args >= -1.).all()
            else:
                norm_args = self.get_normalized_params(p_name=p_name, unnorm_params=skill_args)
                assert (norm_args <= 1.).all and (skill_args >= -1.).all()
        except:
            print("p", p_name)
            print("args", skill_args)
            pass
        ret = skill.act(skill_args, norm=norm)
        if ret is not None:
            ret['info']['interest_interaction'] = skill.check_interesting_interaction()
            ret['info']['done_interaction'] = skill.is_success()
        return ret

    def get_normalized_params(self, p_name, unnorm_params):
        skill = self.name_to_skill[p_name]
        param_dim = skill.get_param_dim()
        skill_unnorm_params = unnorm_params[:param_dim]
        pad_len = len(unnorm_params) - param_dim
        norm_params = []
        if skill.pos_dim is not None:
            norm_params.append(skill._get_normalized_params(skill_unnorm_params[skill.pos_dim[0], skill.pos_dim[1]], skill._config['global_xyz_bounds']))
        if skill.orn_dim is not None:
            norm_params.append(skill._get_normalized_params(skill_unnorm_params[skill.orn_dim[0], skill.orn_dim[1]], skill._config['yaw_bounds']))
        if skill.gripper_dim is not None:
            norm_params.append(skill_unnorm_params[skill.gripper_dim[0], skill.gripper_dim[1]])
        if p_name == 'push':
            assert skill.delta_dim is not None
            unnorm_delta = skill_unnorm_params[skill.delta_dim[0], skill.delta_dim[1]]
            norm_delta = unnorm_delta / skill._config['delta_xyz_scale']
            norm_delta = np.clip(norm_delta, -1, 1)
            norm_params.append(norm_delta)
        return np.concatenate([np.concatenate(norm_params), np.zeros(pad_len)])

    def get_unnormalized_params(self, p_name, norm_params):
        skill = self.name_to_skill[p_name]
        param_dim = skill.get_param_dim()
        skill_norm_params = norm_params[:param_dim]
        pad_len = len(norm_params) - param_dim
        unnorm_params = []
        if skill.pos_dim is not None:
            unnorm_params.append(skill._get_unnormalized_params(skill_norm_params[skill.pos_dim[0], skill.pos_dim[1]], skill._config['global_xyz_bounds']))
        if skill.orn_dim is not None:
            unnorm_params.append(skill._get_unnormalized_params(skill_norm_params[skill.orn_dim[0], skill.orn_dim[1]], skill._config['yaw_bound']))
        if skill.gripper is not None:
            unnorm_params.append(skill_norm_params[skill.gripper_dim[0], skill.gripper_dim[1]])
        if p_name == 'push':
            assert skill.delta_dim is not None
            norm_delta = skill_norm_params[skill.delta_dim[0], skill.delta_dim[1]]
            norm_delta = np.clip(norm_delta, -1, 1)
            unnorm_delta = norm_delta * skill._config['delta_xyz_scale']
            unnorm_params.append(unnorm_delta)
        return np.concatenate([np.concatenate(unnorm_params), np.zeros(pad_len)])