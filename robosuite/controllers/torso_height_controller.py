import numpy as np

import robosuite.utils.transform_utils as T


class TorsoHeightController:
    def __init__(
        self,
        sim,
        joint_indexes,
        **kwargs
    ):
        self.sim = sim

        self.joint_index = joint_indexes["joints"]
        self.qpos_index = joint_indexes["qpos"]
        self.qvel_index = joint_indexes["qvel"]
        self.control_dim = len(self.joint_index)

    def reset(self):
        self._target_height = None
        self._controlling_height = False

    def set_goal(self, action):
        height_action = action[0]
        if abs(height_action) < 0.1:
            self._controlling_height = False
            self.height_action_actual = 0.0
        else:
            self._controlling_height = True
            self.height_action_actual = height_action

    def run_controller(self):
        # joint_name = f"{self.base_name}joint_torso_height"
        joint_name = self.sim.model.joint_id2name(self.joint_index[0])
        # if self._target_height is None or self._controlling_height:
        #     self._target_height = self.sim.data.get_joint_qpos(joint_name)
        #     self._target_height = 
        current_height = self.sim.data.get_joint_qpos(joint_name)
        if not self._controlling_height:
            z_error = self._target_height - current_height
            return 100 * z_error
        else:
            return self.height_action_actual
