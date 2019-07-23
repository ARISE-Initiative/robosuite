"""
This file implements a wrapper for controlling robosuite environments
using same configuration, obs/action space, and IK as in RobotTeleop.

TODO: For now, requires `robosuite` branch of RobotTeleop
"""

import os
import time
import numpy as np
import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import Wrapper

try:
    from RobotTeleop import make_robot, make_controller, make_config
    import RobotTeleop.utils as U
except:
    print(f'RobotTeleop is needed to to teleop')


class TeleopWrapper(Wrapper):
    env = None

    def __init__(self, env, config='BaseServerConfig'):
        super().__init__(env)
        self.config = make_config(config)
        self.config.infer_settings()
        self.robot = make_robot(self.config.robot.type, config=self.config, env=self.env)
        self.controller = make_controller(self.config.controller.type, robot=self.robot, config=self.config)

    def reset(self):
        self.robot.reset()
        self.controller.reset()

    def step(self, action):
        """
        action assumed to be
        [ delta_pos, delta_rot (euler angles), gripper_status]
        gripper will be closed when gripper_status is non 0
        """
        self.robot.robot_arm.blocking = False

        if self.config.controller.ik.control_with_orn:
            cur_rot = self.robot.eef_orientation()
            cur_rot = U.mat2euler(cur_rot)

            new_rot = cur_rot + action[3:6]
            new_rot = U.euler2quat(new_rot)
            new_rot = U.quat2mat(new_rot)
        else:
            new_rot = self.robot.eef_orientation()

        action = {
            'dpos': np.array(action[:3]),
            'rotation': new_rot,
            'timestamp': time.time(),
            'engaged': True,
            'zoom': 0,
            'sensitivity': 0,
            'valid': True,
            'grasp': action[-1]
        }

        self.toggle_gripper(action['grasp'])

        self.controller.apply_control(action)

        self.sleep()
        self.last_t = time.time()

        obs = self._get_observation()
        reward = self.reward()
        done = self.env._check_success()
        self.robot.robot_arm.blocking = True
        return obs, reward, done, {}