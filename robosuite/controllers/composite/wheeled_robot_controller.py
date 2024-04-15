import numpy as np

from .composite_controller import CompositeController


class WheeledRobotController(CompositeController):
    def reset(self):
        super().reset()
        self.mode = None

    def set_goal(self, all_action):
        # self.sim.forward()

        nonzero_arm_action = np.any(all_action[:6] != 0)
        if nonzero_arm_action:
            self.mode = "arm"
        else:
            self.mode = "base"

        # compute goal base pose
        # goal_base_pose =

        for part_name, controller in self.controllers.items():
            start_idx, end_idx = self._action_split_indexes[part_name]
            action = all_action[start_idx:end_idx]
            if part_name in self.grippers.keys():
                action = self.grippers[part_name].format_action(action)

            if part_name in ["left", "right"]:
                controller.set_goal(action, base_updated=(self.mode == "base"))
            else:
                controller.set_goal(action)

    # def update_state(self):
    #     for arm in self.arms:
    #         self.controllers[arm].update_base_pose()
