import numpy as np


class MobileBaseController:
    def __init__(
        self,
        sim,
    ):
        self.sim = sim

    def reset(self):
        self.init_pos, self.init_ori = self.get_base_pose()

    def get_base_pose(self):
        # YifengTODO: change them to not-hardcoded values
        base_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id("mobile_base0_center")])
        base_rot = np.array(self.sim.data.site_xmat[self.sim.model.site_name2id("mobile_base0_center")].reshape([3, 3]))
        return base_pos, base_rot

    def run_controller(self):
        # TODO: don't hardcode control range
        ctrl_range = np.array([[-1.00, 1.00], [-1.00, 1.00], [-4.00, 4.00]])
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_base_action = bias + weight * self.base_action_actual

        # applied_base_action[0] *= -1
        return applied_base_action

    def set_goal(self, action, set_pos=None, set_ori=None):
        curr_pos, curr_ori = self.get_base_pose()

        # transform the action relative to initial base orientation
        init_theta = np.arctan2(self.init_pos[1], self.init_pos[0])
        curr_theta = np.arctan2(curr_pos[1], curr_pos[0])

        base_action = np.copy([action[i] for i in [1, 0, 3]])
        # input raw base action is delta relative to current pose of base
        # controller expects deltas relative to initial pose of base at start of episode
        # transform deltas from current base pose coordinates to initial base pose coordinates
        x, y = base_action[0:2]
        theta = curr_theta - init_theta
        base_action[0] = x * np.cos(theta) - y * np.sin(theta)
        base_action[1] = x * np.sin(theta) + y * np.cos(theta)

        self.base_action_actual = base_action
