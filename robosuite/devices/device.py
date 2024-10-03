import abc  # for abstract base class definitions

import numpy as np

import robosuite.utils.transform_utils as T


class Device(metaclass=abc.ABCMeta):
    """
    Base class for all robot controllers.
    Defines basic interface for all controllers to adhere to.
    Also contains shared logic for managing multiple and/or multiarmed robots.
    """

    def __init__(self, env):
        """
        Args:
            env (RobotEnv): The environment which contains the robot(s) to control
                            using this device.
        """
        self.env = env
        self.all_robot_arms = [robot.arms for robot in self.env.robots]
        self.num_robots = len(self.all_robot_arms)

    def _reset_internal_state(self):
        """
        Resets internal state related to robot control
        """
        self.grasp_states = [[False] * len(self.all_robot_arms[i]) for i in range(self.num_robots)]
        self.active_arm_indices = [0] * len(self.all_robot_arms)
        self.active_robot = 0
        self.base_modes = [False] * len(self.all_robot_arms)

        self._prev_target = {arm: None for arm in self.all_robot_arms[self.active_robot]}

    @property
    def active_arm(self):
        return self.all_robot_arms[self.active_robot][self.active_arm_index]

    @property
    def grasp(self):
        return self.grasp_states[self.active_robot][self.active_arm_index]

    @property
    def active_arm_index(self):
        return self.active_arm_indices[self.active_robot]

    @property
    def base_mode(self):
        return self.base_modes[self.active_robot]

    @active_arm_index.setter
    def active_arm_index(self, value):
        self.active_arm_indices[self.active_robot] = value

    @abc.abstractmethod
    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_controller_state(self):
        """Returns the current state of the device, a dictionary of pos, orn, grasp, and reset."""
        raise NotImplementedError

    def _prescale_raw_actions(self, dpos, drotation):
        raise NotImplementedError

    def input2action(self, mirror_actions=False):
        """
        Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

        If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action

        Args:
            mirror_actions (bool): actions corresponding to viewing robot from behind.
                first axis: left/right. second axis: back/forward. third axis: down/up.

        Returns:
            2-tuple:

                - (None or np.array): Action interpreted from @device including any gripper action(s). None if we get a
                    reset signal from the device
                - (None or int): 1 if desired close, -1 if desired open gripper state. None if get a reset signal from the
                    device

        """
        robot = self.env.robots[self.active_robot]
        active_arm = self.active_arm

        state = self.get_controller_state()
        # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
        #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
        #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
        dpos, rotation, raw_drotation, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["raw_drotation"],
            state["grasp"],
            state["reset"],
        )

        if mirror_actions:
            dpos[0], dpos[1] = dpos[1], dpos[0]
            raw_drotation[0], raw_drotation[1] = raw_drotation[1], raw_drotation[0]

            dpos[1] *= -1
            raw_drotation[0] *= -1

        # If we're resetting, immediately return None
        if reset:
            return None

        # Get controller reference
        controller = robot.part_controllers[active_arm]
        gripper_dof = robot.gripper[active_arm].dof

        assert controller.name in ["OSC_POSE", "JOINT_POSITION"], "only supporting OSC_POSE and JOINT_POSITION for now"

        # process raw device inputs
        drotation = raw_drotation[[1, 0, 2]]
        # Flip z
        drotation[2] = -drotation[2]
        # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
        dpos, drotation = self._prescale_raw_actions(dpos, drotation)
        # map 0 to -1 (open) and map 1 to 1 (closed)
        grasp = 1 if grasp else -1

        ac_dict = {}
        # populate delta actions for the arms
        for arm in robot.arms:
            # OSC keys
            ac_dict[f"{arm}_delta"] = np.zeros(6)
            ac_dict[f"{arm}_abs"] = robot.part_controllers[arm].delta_to_abs_action(np.zeros(6))
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

            # # mink keys
            # new_pos, new_ori = self.get_updated_pose_target(
            #     arm,
            #     f"gripper0_{arm}_grip_site",
            #     np.zeros(6),
            # )
            # ac_dict[f"gripper0_{arm}_grip_site_pos"] = new_pos
            # ac_dict[f"gripper0_{arm}_grip_site_axis_angle"] = new_ori

        if robot.is_mobile:
            base_mode = bool(state["base_mode"])
            if base_mode is True:
                arm_delta = np.zeros(6)
                base_ac = np.array([dpos[0], dpos[1], drotation[2]])
                torso_ac = np.array([dpos[2]])
            else:
                arm_delta = np.concatenate([dpos, drotation])
                base_ac = np.zeros(3)
                torso_ac = np.zeros(1)

            controller = robot.part_controllers[active_arm]

            # # mink keys
            # new_pos, new_ori = self.get_updated_pose_target(
            #     active_arm,
            #     f"gripper0_{active_arm}_grip_site",
            #     arm_delta,
            # )
            # ac_dict[f"gripper0_{active_arm}_grip_site_pos"] = new_pos
            # ac_dict[f"gripper0_{active_arm}_grip_site_axis_angle"] = new_ori

            # populate action dict items
            ac_dict[f"{active_arm}_delta"] = arm_delta
            ac_dict[f"{active_arm}_abs"] = robot.part_controllers[active_arm].delta_to_abs_action(arm_delta)
            ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper_dof)
            ac_dict["base"] = base_ac
            # ac_dict["torso"] = torso_ac
            ac_dict["base_mode"] = np.array([1 if base_mode is True else -1])
        else:
            # Create action based on action space of individual robot
            ac_dict[f"{active_arm}_delta"] = np.concatenate([dpos, drotation])
            ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper_dof)

        # clip actions between -1 and 1
        for (k, v) in ac_dict.items():
            if "abs" not in k:
                ac_dict[k] = np.clip(v, -1, 1)

        return ac_dict

    def get_updated_pose_target(self, arm, site_name, delta, target_based_update=True):
        if target_based_update is True and self._prev_target[arm] is not None:
            curr_pos = self._prev_target[arm][0:3].copy()
            curr_ori = T.quat2mat(T.axisangle2quat(self._prev_target[arm][3:6].copy()))
        else:
            curr_pos = self.env.sim.data.get_site_xpos(site_name).copy()
            curr_ori = self.env.sim.data.get_site_xmat(site_name).copy()

        # self.env.sim.model.site_rgba[self.env.sim.model.site_name2id(site_name)] = [1.0, 0.0, 0.0, 1.0]

        new_pos = curr_pos + delta[0:3] * 0.05
        delta_ori = T.quat2mat(T.axisangle2quat(delta[3:6] * 0.01))
        new_ori = np.dot(delta_ori, curr_ori)
        new_axisangle = T.quat2axisangle(T.mat2quat(new_ori))
        # new_axisangle = T.quat2axisangle(T.mat2quat(curr_ori))

        self._prev_target[arm] = np.concatenate([new_pos, new_axisangle])

        if arm == "right":
            print(new_axisangle)

        return new_pos, new_axisangle
