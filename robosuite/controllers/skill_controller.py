import collections
import copy
import numpy as np
from robosuite.controllers.skills import (
	AtomicSkill,
	ReachSkill,
	ReachOSCSkill,
	GraspSkill,
	PushSkill,
	GripperSkill,
)

class SkillController:

	SKILL_NAMES = [
		'atomic',
		'reach_osc', 'reach',
		'grasp',
		'push',
		'open', 'close',
	]

	def __init__(
			self,
			env,
			config,
	):
		self._env = env
		self._setup_config(config)
		self._num_objs = None

		self._cur_skill = None
		self._num_ac_calls = None
		self._max_ac_calls = None
		self._pos_is_delta = None
		self._ori_is_delta = None
		self._num_skill_repeats = 0

	def _setup_config(self, config):
		default_config = dict(
			success_penalty_fac=1.0,
			aff_penalty_fac=1.0,
			skills=['atomic'],
		)
		self._config = copy.deepcopy(default_config)
		self._config.update(config)

		skill_names = self._config['skills']
		for skill_name in skill_names:
			assert skill_name in SkillController.SKILL_NAMES
		skill_names = [
			skill_name for skill_name in SkillController.SKILL_NAMES
			if skill_name in skill_names
		]

		assert len(skill_names) >= 1
		assert self._config['success_penalty_fac'] >= 1.0

		if skill_names == ['atomic']:
			self._config['ignore_aff_rew'] = True
		elif 'ignore_aff_rew' not in self._config:
			self._config['ignore_aff_rew'] = False

		assert self._config['aff_penalty_fac'] >= 0.0

		self._skills = collections.OrderedDict()
		for skill_name in skill_names:
			base_skill_config = self._config.get('base_config', {})
			skill_config = copy.deepcopy(base_skill_config)
			if skill_name == 'atomic':
				skill_class = AtomicSkill
				skill_config.update(self._config.get('atomic_config', {}))
			elif skill_name == 'reach':
				skill_class = ReachSkill
				skill_config.update(self._config.get('reach_config', {}))
			elif skill_name == 'reach_osc':
				skill_class = ReachOSCSkill
				skill_config.update(self._config.get('reach_config', {}))
			elif skill_name == 'grasp':
				skill_class = GraspSkill
				skill_config.update(self._config.get('grasp_config', {}))
			elif skill_name == 'push':
				skill_class = PushSkill
				skill_config.update(self._config.get('push_config', {}))
			elif skill_name in ['open', 'close']:
				skill_class = GripperSkill
				skill_config.update(self._config.get('gripper_config', {}))
			else:
				raise ValueError
			skill_config.update(
				self._config.get(skill_name + '_config', {})
			)
			self._skills[skill_name] = skill_class(skill_name, **skill_config)

		self._param_dims = None

	def get_skill_dim(self):
		num_skills = len(self._skills)
		if num_skills <= 1:
			return 0
		else:
			return num_skills

	def get_param_dim(self, base_param_dim):
		if self._param_dims is None:
			self._param_dims = collections.OrderedDict()
			for skill_name, skill in self._skills.items():
				dim = skill.get_param_dim(base_param_dim)
				self._param_dims[skill_name] = dim
		return np.max(list(self._param_dims.values()))

	def get_skill_names(self):
		return list(self._skills.keys())

	def reset(self, action):
		skill_name = self.get_skill_name_from_action(action)
		params = self.get_params_from_action(action)
		self._cur_skill = self._skills[skill_name]

		self._env._reset_skill()
		robot = self._env.robots[0]
		skill_config_update = dict(
			robot_controller_dim=robot.controller.control_dim,
			robot_gripper_dim=robot.gripper.dof,
			robot_controller=robot.controller,
		)
		info = self._get_info()
		self._cur_skill.reset(params, skill_config_update, info)
		self._num_ac_calls = 0
		self._max_ac_calls = self._cur_skill.get_max_ac_calls()
		self._pos_is_delta = None
		self._ori_is_delta = None

	def step(self):
		info = self._get_info()
		skill = self._cur_skill
		skill.update_state(info)

		pos, pos_is_delta = skill.get_pos_ac(info)
		ori, ori_is_delta = skill.get_ori_ac(info)
		g = skill.get_gripper_ac(info)

		self._pos_is_delta = pos_is_delta
		self._ori_is_delta = ori_is_delta
		self._num_ac_calls += 1

		return np.concatenate([pos, ori, g])

	def _get_info(self):
		info = self._env._get_skill_info()
		robot = self._env.robots[0]
		info['cur_ee_pos'] = np.array(robot.sim.data.site_xpos[robot.eef_site_id])
		return info

	def ac_is_delta(self):
		assert self._pos_is_delta in [True, False]
		assert self._ori_is_delta in [True, False]
		return self._pos_is_delta, self._ori_is_delta

	def is_success(self):
		info = self._get_info()
		return self._cur_skill.is_success(info)

	def done(self):
		return self.is_success() or (self._num_ac_calls >= self._max_ac_calls)

	def get_num_ac_calls(self):
		return self._num_ac_calls

	def get_aff_reward(self):
		return self._cur_skill.get_aff_reward()

	def get_aff_success(self):
		return self._cur_skill.get_aff_success()

	def post_process_reward(self, reward):
		if not self.is_success():
			reward = reward / self._config['success_penalty_fac']

		if self._config['ignore_aff_rew']:
			return reward

		aff_reward = self.get_aff_reward()
		assert 0.0 <= aff_reward <= 1.0
		aff_penalty = 1.0 - aff_reward
		aff_penalty_fac = self._config['aff_penalty_fac']
		reward -= (aff_penalty_fac * aff_penalty)

		return reward

	def get_skill_name_from_action(self, action):
		skill_dim = self.get_skill_dim()
		skill_names = self.get_skill_names()
		if skill_dim == 0:
			# there is only one skill to choose from
			return skill_names[0]

		skill_action = action[:skill_dim]
		skill_idx = np.argmax(skill_action)
		return skill_names[skill_idx]

	def get_params_from_action(self, action):
		skill_dim = self.get_skill_dim()
		param_dims = len(action) - skill_dim
		return action[-param_dims:]

	def get_skill_code(self, skill_name, default=None):
		skill_names = self.get_skill_names()
		if skill_name not in skill_names:
			skill_name = default
		if skill_name is None:
			return None
		skill_idx = skill_names.index(skill_name)
		skill_dim = self.get_skill_dim()
		if skill_dim > 0:
			skill_code = np.zeros(skill_dim)
			skill_code[skill_idx] = 1.0
			return skill_code
		else:
			return None

	def get_skill_id_from_action(self, action):
		skill_name = self.get_skill_name_from_action(action)
		skill_names = self.get_skill_names()
		return skill_names.index(skill_name)

	def get_skill_names_and_colors(self):
		skill_color_map = dict(
			atomic='gold',
			reach='dodgerblue',
			reach_osc='dodgerblue',
			grasp='green',
			push='orange',
			open='darkgoldenrod',
			close='gray',
		)

		skill_names = self.get_skill_names()
		return skill_names, [skill_color_map[skill_name] for skill_name in skill_names]

	def get_full_skill_name_map(self):
		map = dict(
			atomic='Atomic',
			reach='Reach',
			reach_osc='Reach OSC',
			grasp='Grasp',
			push='Push',
			open='Release',
			close='Close',
		)

		return map