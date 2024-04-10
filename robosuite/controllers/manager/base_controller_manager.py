import numpy as np
from collections import OrderedDict
from robosuite.controllers import controller_factory, load_controller_config

class BaseControllerManager():
    def __init__(self, 
                sim,
                robot_model,
                grippers) -> None: 
        # TODO: grippers repeat with members inside robot_model. Currently having this additioanl field to make naming query easy.
        self.sim = sim
        self.robot_model = robot_model

        self.grippers = grippers

        self.controllers = OrderedDict()

        self._action_split_indexes = OrderedDict()

        self.controller_config = None

        self.arms = self.robot_model.arms

        self._applied_action_dict = {}

    def load_controller_config(self, controller_config):
        self.controller_config = controller_config
        self.controllers.clear()
        self._action_split_indexes.clear()
        self._init_controllers()
        self.setup_action_split_idx()

    def _init_controllers(self):
        for part_name in self.controller_config.keys():
            self.controllers[part_name] = controller_factory(self.controller_config[part_name]["type"], self.controller_config[part_name])

    def setup_action_split_idx(self):
        previous_idx = 0
        last_idx = 0
        for part_name, controller in self.controllers.items():
            if part_name in self.grippers.keys():
                last_idx += self.grippers[part_name].dof
            else:
                last_idx += controller.control_dim
            self._action_split_indexes[part_name] = (previous_idx, last_idx)
            previous_idx = last_idx

    def set_goal(self, all_action):
        self.sim.forward()

        for part_name, controller in self.controllers.items():
            start_idx, end_idx = self._action_split_indexes[part_name]
            action = all_action[start_idx:end_idx]
            if part_name in self.grippers.keys():
                action = self.grippers[part_name].format_action(action)
            controller.set_goal(action)

    def reset(self):
        for part_name, controller in self.controllers.items():
            controller.reset_goal()

    def compute_applied_action(self, enabled_parts):
        self.sim.forward()
        self.update_state()
        self._applied_action_dict.clear()
        for part_name, controller in self.controllers.items():
            if enabled_parts.get(part_name, False):
                self._applied_action_dict[part_name] = controller.run_controller()

        return self._applied_action_dict

    def get_control_dim(self, part_name):
        if part_name not in self.controllers:
            return 0
        else:
            return self.controllers[part_name].control_dim

    def update_state(self):
        for arm in self.arms:
            self.controllers[arm].update_base_pose()

    def get_controller(self, part_name):
        return self.controllers[part_name]


    @property
    def action_limits(self):
        low, high = [], []
        for part_name, controller in self.controllers.items():
            if part_name not in self.arms:
                if part_name in self.grippers.keys():
                    low_g, high_g = (
                        ([-1] * self.grippers[part_name].dof, [1] * self.grippers[part_name].dof)
                    )
                    low, high = np.concatenate([low, low_g]), np.concatenate([high, high_g])
                else:
                    control_dim = controller.control_dim
                    low_c, high_c = ([-1] * control_dim, [1] * control_dim)
                    low, high = np.concatenate([low, low_c]), np.concatenate([high, high_c])
            else:
                low_c, high_c = controller.control_limits
                low, high = np.concatenate([low, low_c]), np.concatenate([high, high_c])
        return low, high
