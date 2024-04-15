from copy import deepcopy

import numpy as np

from .composite_controller import CompositeController


class WheeledRobotController(CompositeController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_pos = None
        self.base_ori = None

    def reset(self):
        super().reset()
        self.mode = None

        self.base_pos = None
        self.base_ori = None

    def set_goal(self, all_action):
        # self.sim.forward()

        nonzero_arm_action = np.any(all_action[:6] != 0)
        if nonzero_arm_action:
            origin_updated = False
        else:
            origin_updated = True

        for part_name, controller in self.controllers.items():
            start_idx, end_idx = self._action_split_indexes[part_name]
            action = all_action[start_idx:end_idx]
            if part_name in self.grippers.keys():
                action = self.grippers[part_name].format_action(action)

            if part_name in ["left", "right"]:
                controller.set_goal(action, origin_updated=origin_updated)
            else:
                controller.set_goal(action)

    def update_state(self):
        self._prev_base_pos = deepcopy(self.base_pos)
        self._prev_base_ori = deepcopy(self.base_ori)

        base_pos, base_ori = self.controllers["base"].get_base_pose()

        self.base_pos = base_pos
        self.base_ori = base_ori

        if self._prev_base_pos is None:
            base_vel = np.zeros(3)
        else:
            base_vel = self.base_pos - self._prev_base_pos

        ref_pos = base_pos + 75.0 * base_vel
        ref_ori = base_ori

        for arm in self.arms:
            self.controllers[arm].update_origin(ref_pos, ref_ori)
