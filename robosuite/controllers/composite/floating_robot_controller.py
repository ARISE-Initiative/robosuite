from collections import deque
from copy import deepcopy

import numpy as np

import robosuite.utils.transform_utils as T

from .composite_controller import CompositeController


class FloatingRobotController(CompositeController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_pose_history = deque(maxlen=20)

    def reset(self):
        super().reset()
        self.mode = None
        self.base_pose_history = deque(maxlen=20)

    def update_state(self):
        base_pos, base_ori = self.get_base_pose()
        self.base_pose_history.extend([(base_pos, base_ori)])

        # compute origin_position
        prev_base_pos = self.base_pose_history[max(0, len(self.base_pose_history) - 2)][0]
        base_vel_pos = base_pos - prev_base_pos
        origin_pos = base_pos + 75.0 * base_vel_pos

        # compute origin orientation
        prev_base_ori = self.base_pose_history[max(0, len(self.base_pose_history) - 20)][1]
        base_ori_euler = T.mat2euler(base_ori)
        curr_base_yaw = T.mat2euler(base_ori)[2]
        prev_base_yaw = T.mat2euler(prev_base_ori)[2]
        rot_vel = curr_base_yaw - prev_base_yaw
        base_ori_euler[2] += 5.0 * rot_vel
        origin_ori = T.euler2mat(base_ori_euler)

        for arm in self.arms:
            self.controllers[arm].update_origin(origin_pos, origin_ori)
