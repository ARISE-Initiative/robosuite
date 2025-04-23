import abc
from typing import Dict, List, Optional  # for abstract base class definitions

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.controllers.parts.arm.osc import OperationalSpaceController


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
        self._all_robot_arms = None

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
    def all_robot_arms(self):
        if self._all_robot_arms is None:
            self._all_robot_arms = [robot.arms for robot in self.env.robots]
        return self._all_robot_arms

    @property
    def num_robots(self):
        return len(self.all_robot_arms)

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
    def get_controller_state(self) -> Dict:
        """Returns the current state of the device, a dictionary of pos, orn, grasp, and reset."""
        raise NotImplementedError

    def _postprocess_device_outputs(self, dpos, drotation):
        raise NotImplementedError

    def input2action(self, mirror_actions=False) -> Optional[Dict]:
        """
        Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

        If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action

        Args:
            mirror_actions (bool): actions corresponding to viewing robot from behind.
                first axis: left/right. second axis: back/forward. third axis: down/up.

        Returns:
            Optional[Dict]: Dictionary of actions to be fed into env.step()
                            if reset is triggered, returns None
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
            dpos[0] *= -1
            dpos[1] *= -1
            raw_drotation[0] *= -1
            raw_drotation[1] *= -1

        # If we're resetting, immediately return None
        if reset:
            return None

        # Get controller reference
        controller = robot.part_controllers[active_arm]
        gripper = robot.gripper[active_arm]
        gripper_dof = robot.gripper[active_arm].dof

        assert controller.name in ["OSC_POSE", "JOINT_POSITION"], "only supporting OSC_POSE and JOINT_POSITION for now"

        # process raw device inputs
        drotation = raw_drotation[[1, 0, 2]]
        # Flip z
        drotation[2] = -drotation[2]
        # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
        dpos, drotation = self._postprocess_device_outputs(dpos, drotation)
        # map 0 to -1 (open) and map 1 to 1 (closed)
        grasp = 1 if grasp else -1

        ac_dict = {}
        # populate delta actions for the arms
        for arm in robot.arms:
            # OSC keys
            arm_action = self.get_arm_action(
                robot,
                arm,
                norm_delta=np.zeros(6),
            )
            ac_dict[f"{arm}_abs"] = arm_action["abs"]
            ac_dict[f"{arm}_delta"] = arm_action["delta"]
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)

        if robot.is_mobile:
            base_mode = bool(state["base_mode"])
            if base_mode is True:
                arm_norm_delta = np.zeros(6)
                base_ac = np.array([dpos[0], dpos[1], drotation[2]])
                torso_ac = np.array([dpos[2]])
            else:
                arm_norm_delta = np.concatenate([dpos, drotation])
                base_ac = np.zeros(3)
                torso_ac = np.zeros(1)

            ac_dict["base"] = base_ac
            # ac_dict["torso"] = torso_ac
            ac_dict["base_mode"] = np.array([1 if base_mode is True else -1])
        else:
            arm_norm_delta = np.concatenate([dpos, drotation])

        # populate action dict items for arm and grippers
        arm_action = self.get_arm_action(
            robot,
            active_arm,
            norm_delta=arm_norm_delta,
        )
        ac_dict[f"{active_arm}_abs"] = arm_action["abs"]
        ac_dict[f"{active_arm}_delta"] = arm_action["delta"]

        if hasattr(gripper, "grasp_qpos"):
            ac_dict[f"{active_arm}_gripper"] = getattr(gripper, "grasp_qpos")[grasp]
        else:
            ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper_dof)

        # clip actions between -1 and 1
        for (k, v) in ac_dict.items():
            if "abs" not in k and "gripper" not in k:
                ac_dict[k] = np.clip(v, -1, 1)

        return ac_dict

    def get_arm_action(self, robot, arm, norm_delta, goal_update_mode="target"):
        assert np.all(norm_delta <= 1.0) and np.all(norm_delta >= -1.0)

        assert goal_update_mode in [
            "achieved",
            "target",
        ]  # update next target either based on achieved pose or current target pose

        # TODO: the logic between OSC and while body based ik is fragmented right now. Unify
        if isinstance(robot.part_controllers[arm], OperationalSpaceController):
            arm_controller = robot.part_controllers[arm]
            delta_action = arm_controller.scale_action(norm_delta.copy())
            abs_action = arm_controller.delta_to_abs_action(delta_action, goal_update_mode=None)
            return {
                "delta": norm_delta,
                "abs": abs_action,
            }
        elif robot.composite_controller_config["type"] in ["WHOLE_BODY_MINK_IK", "HYBRID_WHOLE_BODY_MINK_IK"]:
            ref_frame = self.env.robots[0].composite_controller.composite_controller_specific_config.get(
                "ik_input_ref_frame", "world"
            )

            delta_action = norm_delta.copy()
            delta_action[0:3] *= 0.05
            delta_action[3:6] *= 0.15

            # general case
            if goal_update_mode == "achieved" or self._prev_target[arm] is None:
                site_name = f"gripper0_{arm}_grip_site"
                # update next target based on current achieved pose
                pos = self.env.sim.data.get_site_xpos(site_name).copy()
                ori = self.env.sim.data.get_site_xmat(site_name).copy()
                if ref_frame == "base":
                    # convert target in world coordinate to
                    pose_in_world = np.eye(4)
                    pose_in_world[:3, 3] = pos
                    pose_in_world[:3, :3] = ori
                    pose_in_base = self.env.robots[0].composite_controller.joint_action_policy.transform_pose(
                        src_frame_pose=pose_in_world,
                        src_frame="world",  # mocap pose is world coordinates
                        dst_frame=ref_frame,
                    )
                    pos, ori = pose_in_base[:3, 3], pose_in_base[:3, :3]
            else:
                # update next target based on previous target pose
                pos = self._prev_target[arm][0:3].copy()
                ori = T.quat2mat(T.axisangle2quat(self._prev_target[arm][3:6].copy()))

            # new positions computed in world frame coordinates
            new_pos = pos + delta_action[0:3]
            delta_ori = T.quat2mat(T.axisangle2quat(delta_action[3:6]))
            new_ori = np.dot(delta_ori, ori)
            new_axisangle = T.quat2axisangle(T.mat2quat(new_ori))

            abs_action = np.concatenate([new_pos, new_axisangle])
            self._prev_target[arm] = abs_action.copy()

            return {
                "delta": delta_action,
                "abs": abs_action,
            }
        elif robot.composite_controller_config["type"] in ["WHOLE_BODY_IK"]:
            if goal_update_mode == "achieved" or self._prev_target[arm] is None:
                site_name = f"gripper0_{arm}_grip_site"
                # update next target based on current achieved pose
                pos = self.env.sim.data.get_site_xpos(site_name).copy()
                ori = self.env.sim.data.get_site_xmat(site_name).copy()
            else:
                # update next target based on previous target pose
                pos = self._prev_target[arm][0:3].copy()
                ori = T.quat2mat(T.axisangle2quat(self._prev_target[arm][3:6].copy()))

            delta_action = norm_delta.copy()
            delta_action[0:3] *= 0.05
            delta_action[3:6] *= 0.15

            # new positions computed in world frame coordinates
            new_pos = pos + delta_action[0:3]
            delta_ori = T.quat2mat(T.axisangle2quat(delta_action[3:6]))
            new_ori = np.dot(delta_ori, ori)
            new_axisangle = T.quat2axisangle(T.mat2quat(new_ori))

            abs_action = np.concatenate([new_pos, new_axisangle])
            self._prev_target[arm] = abs_action.copy()

            return {
                "delta": delta_action,
                "abs": abs_action,
            }
        else:
            raise NotImplementedError
