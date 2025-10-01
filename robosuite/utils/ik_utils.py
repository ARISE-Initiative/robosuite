from typing import Dict, List, Literal, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER


def get_nullspace_gains(joint_names: List[str], weight_dict: Dict[str, float]) -> np.ndarray:
    return np.array([weight_dict.get(joint, 1.0) for joint in joint_names])


class IKSolver:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_config: Dict,
        damping: float,
        integration_dt: float,
        max_dq: float,
        max_dq_torso: float = 0.3,
        input_type: Literal["keyboard", "mocap", "pkl"] = "keyboard",
        debug: bool = False,
        input_action_repr: Literal["absolute", "relative", "relative_pose"] = "absolute",
        input_file: Optional[str] = None,
        input_rotation_repr: Literal["quat_wxyz", "axis_angle"] = "axis_angle",
        input_ref_frame: Literal["world", "base"] = "world",
    ):
        """
        Args:
            input_action_repr:
                absolute: input actions are absolute positions and rotations.
                relative: input actions are relative to the current position and rotation, separately.
                relative_pose: input actions are relative_pose to the pose of the respective reference site.
        """
        self.full_model = model
        self.full_model_data = data

        self.damping = damping
        self.integration_dt = integration_dt
        self.max_dq = max_dq
        self.max_dq_torso = max_dq_torso

        self.joint_names = robot_config["joint_names"]
        self.site_names = robot_config["end_effector_sites"]
        self.site_ids = [
            self.full_model.site(robot_config["end_effector_sites"][i]).id
            for i in range(len(robot_config["end_effector_sites"]))
        ]
        self.dof_ids = np.array([self.full_model.joint(name).id for name in robot_config["joint_names"]])
        # self.actuator_ids = np.array([self.full_model.actuator(name).id for name in robot_config['joint_names']])  # this works if actuators match joints
        self.actuator_ids = np.array([i for i in range(20)])  # TODO no hardcode; cur for GR1; model.nu is 32
        self.key_id = (
            self.full_model.key(robot_config["initial_keyframe"]).id if "initial_keyframe" in robot_config else None
        )

        self.jac_temps: List[np.ndarray] = [np.zeros((6, self.full_model.nv)) for _ in range(len(self.site_ids))]
        self.twists: List[np.ndarray] = [np.zeros(6) for _ in range(len(self.site_ids))]
        self.site_quats: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]
        self.site_quat_conjs: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]
        self.error_quats: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]

        self.input_action_repr = input_action_repr
        self.input_rotation_repr = input_rotation_repr
        self.input_ref_frame = input_ref_frame
        ROTATION_REPRESENTATION_DIMS: Dict[str, int] = {"quat_wxyz": 4, "axis_angle": 3}
        self.rot_dim = ROTATION_REPRESENTATION_DIMS[input_rotation_repr]
        self.pos_dim = 3
        self.control_dim = len(self.site_names) * (self.pos_dim + self.rot_dim)
        # hardcoded control limits for now
        self.control_limits = np.array([-np.inf] * self.control_dim), np.array([np.inf] * self.control_dim)
        self.debug_iter = 0
        self.debug = debug
        if debug:
            self.pre_clip_errors: List[np.ndarray] = []

        self.input_type = input_type
        if input_type == "mocap":
            self.mocap_ids = [self.full_model.body(name).mocapid[0] for name in robot_config["mocap_bodies"]]
        elif input_type == "pkl":
            self.mocap_ids = [self.full_model.body(name).mocapid[0] for name in robot_config["mocap_bodies"]]
            self.pkl_t = 0
            import pickle

            with open(input_file, "rb") as f:
                input_file = pickle.load(f)
            self.history = input_file

        # Nullspace control
        self.q0 = np.zeros(len(self.dof_ids))
        self.Kn = np.array(robot_config["nullspace_gains"])

        # Initialize error and error_dot
        self.error_prev = np.zeros_like(self.q0)
        self.error_dot = np.zeros_like(self.q0)

    def action_split_indexes(self) -> Dict[str, Tuple[int, int]]:
        action_split_indexes: Dict[str, Tuple[int, int]] = {}
        previous_idx = 0

        for site_name in self.site_names:
            total_dim = self.pos_dim + self.rot_dim
            last_idx = previous_idx + total_dim
            simplified_site_name = "left" if "left" in site_name else "right"  # hack to simplify site names
            # goal is to specify the end effector actions as "left" or "right" instead of the actual site name
            # we assume that the site names for the ik solver are unique and contain "left" or "right" in them
            action_split_indexes[simplified_site_name] = (previous_idx, last_idx)
            previous_idx = last_idx

        return action_split_indexes

    def reset_to_initial_state(self):
        if self.key_id is not None:
            mujoco.mj_resetDataKeyframe(self.full_model, self.full_model_data, self.key_id)
        else:
            print("No initial keyframe set. Skipping reset.")

    def set_target_positions(self, target_positions: List):
        for i, pos in enumerate(target_positions):
            self.full_model_data.mocap_pos[self.mocap_ids[i]] = pos

    def set_target_rotations(self, target_rotations: List):
        for i, quat_wxyz in enumerate(target_rotations):
            self.full_model_data.mocap_quat[self.mocap_ids[i]] = quat_wxyz

    def get_targets(self):
        # by default set target to current site position
        target_pos = np.array([self.full_model_data.site(site_id).xpos for site_id in self.site_ids])
        target_ori_mat = np.array([self.full_model_data.site(site_id).xmat for site_id in self.site_ids])
        target_ori = np.array([np.ones(4) for _ in range(len(self.site_ids))])
        [mujoco.mju_mat2Quat(target_ori[i], target_ori_mat[i]) for i in range(len(self.site_ids))]

        if self.input_type == "mocap":
            for i in range(len(self.site_ids)):
                target_pos[i] = self.full_model_data.mocap_pos[self.mocap_ids[i]]
                target_ori[i] = self.full_model_data.mocap_quat[self.mocap_ids[i]]
        elif self.input_type == "pkl":
            left_index, right_index = 0, 1
            self.pkl_t += 1 / 10  # hardcoded pkl playback freq for now
            if int(self.pkl_t) >= len(self.history["left_eef_pos"]):
                print("Reached end of pkl file. Exiting.")
                exit()
            target_pos[left_index] = self.history["left_eef_pos"][int(self.pkl_t)]
            target_pos[right_index] = self.history["right_eef_pos"][int(self.pkl_t)]
            target_ori[left_index] = self.history["left_eef_quat_wxyz"][int(self.pkl_t)]
            target_ori[right_index] = self.history["right_eef_quat_wxyz"][int(self.pkl_t)]
            self.full_model_data.mocap_pos[self.mocap_ids[left_index]] = self.history["left_eef_pos"][int(self.pkl_t)]
            self.full_model_data.mocap_pos[self.mocap_ids[right_index]] = self.history["right_eef_pos"][int(self.pkl_t)]
            self.full_model_data.mocap_quat[self.mocap_ids[left_index]] = self.history["left_eef_quat_wxyz"][
                int(self.pkl_t)
            ]
            self.full_model_data.mocap_quat[self.mocap_ids[right_index]] = self.history["right_eef_quat_wxyz"][
                int(self.pkl_t)
            ]
        else:
            raise ValueError(f"Invalid input type {self.input_type}")

        return target_pos, target_ori

    def _compute_jacobian(self, model: mujoco.MjModel, data: mujoco.MjData):
        for i, site_id in enumerate(self.site_ids):
            mujoco.mj_jacSite(model, data, self.jac_temps[i][:3], self.jac_temps[i][3:], site_id)

        jac = np.vstack(self.jac_temps)
        # compute jacobian for our dof_ids
        jac = jac[:, self.dof_ids]
        return jac

    def forward_kinematics(self, qpos: np.ndarray) -> Dict[str, np.ndarray]:
        data = mujoco.MjData(self.full_model)
        data.qpos[self.dof_ids] = qpos
        mujoco.mj_kinematics(self.full_model, data)
        return {name: data.site(site_id).xpos for name, site_id in zip(self.site_names, self.site_ids)}

    def transform_pose(
        self, src_frame_pose: np.ndarray, src_frame: Literal["world", "base"], dst_frame: Literal["world", "base"]
    ) -> np.ndarray:
        """
        Transforms src_frame_pose from src_frame to dst_frame.
        """
        if src_frame == dst_frame:
            return src_frame_pose

        X_src_frame_pose = src_frame_pose
        # convert src frame pose to world frame pose
        if src_frame != "world":
            X_W_src_frame = T.make_pose(
                translation=self.full_model.body(src_frame).pos,
                rotation=T.quat2mat(np.roll(self.full_model.body(src_frame).quat, shift=-1)),
            )
            X_W_pose = X_W_src_frame @ X_src_frame_pose
        else:
            X_W_pose = src_frame_pose

        # now convert to destination frame
        if dst_frame == "world":
            X_dst_frame_pose = X_W_pose
        elif dst_frame == "base":
            X_dst_frame_W = np.linalg.inv(
                T.make_pose(
                    translation=self.full_model.body("robot0_base").pos,
                    rotation=T.quat2mat(np.roll(self.full_model.body("robot0_base").quat, shift=-1)),
                )
            )  # hardcode name of base
            X_dst_frame_pose = X_dst_frame_W.dot(X_W_pose)

        return X_dst_frame_pose

    def solve(
        self,
        target_action: np.ndarray,
        Kpos: float = 0.95,
        Kori: float = 0.95,
    ):
        target_action = target_action.reshape(len(self.site_names), -1)
        target_pos = target_action[:, : self.pos_dim]
        target_ori = target_action[:, self.pos_dim :]
        target_quat_wxyz = None

        if self.input_rotation_repr == "axis_angle":
            target_quat_wxyz = np.array([np.roll(T.axisangle2quat(target_ori[i]), 1) for i in range(len(target_ori))])
        elif self.input_rotation_repr == "mat":
            target_quat_wxyz = np.array([np.roll(T.mat2quat(target_ori[i])) for i in range(len(target_ori))])
        elif self.input_rotation_repr == "quat_wxyz":
            target_quat_wxyz = target_ori

        if "relative" in self.input_action_repr:
            cur_pos = np.array([self.full_model_data.site(site_id).xpos for site_id in self.site_ids])
            cur_ori = np.array([self.full_model_data.site(site_id).xmat for site_id in self.site_ids])
        if self.input_action_repr == "relative":
            # decoupled pos and rotation deltas
            target_pos += cur_pos
            target_quat_xyzw = np.array(
                [
                    T.quat_multiply(T.mat2quat(cur_ori[i].reshape(3, 3)), np.roll(target_quat_wxyz[i], -1))
                    for i in range(len(self.site_ids))
                ]
            )
            target_quat_wxyz = np.array([np.roll(target_quat_xyzw[i], shift=1) for i in range(len(self.site_ids))])
        elif self.input_action_repr == "relative_pose":
            cur_poses = np.zeros((len(self.site_ids), 4, 4))
            for i in range(len(self.site_ids)):
                cur_poses[i, :3, :3] = cur_ori[i].reshape(3, 3)
                cur_poses[i, :3, 3] = cur_pos[i]
                cur_poses[i, 3, :] = [0, 0, 0, 1]

            # Convert target action to target pose
            target_poses = np.zeros_like(cur_poses)
            for i in range(len(self.site_ids)):
                target_poses[i, :3, :3] = T.quat2mat(target_quat_wxyz[i])
                target_poses[i, :3, 3] = target_pos[i]
                target_poses[i, 3, :] = [0, 0, 0, 1]

            # Apply target pose to current pose
            new_target_poses = np.array([np.dot(cur_poses[i], target_poses[i]) for i in range(len(self.site_ids))])

            # Split new target pose back into position and quaternion
            target_pos = new_target_poses[:, :3, 3]
            target_quat_wxyz = np.array(
                [np.roll(T.mat2quat(new_target_poses[i, :3, :3]), shift=1) for i in range(len(self.site_ids))]
            )

        if self.input_ref_frame == "base":
            for i in range(len(target_pos)):
                X_B_goal = T.make_pose(
                    translation=target_pos[i],
                    rotation=T.quat2mat(np.roll(target_quat_wxyz[i], -1)),
                )
                X_W_goal = self.transform_pose(X_B_goal, src_frame="robot0_base", dst_frame="world")
                target_pos[i] = X_W_goal[:3, 3]
                target_quat_wxyz[i] = np.roll(T.mat2quat(X_W_goal[:3, :3]), 1)

        jac = self._compute_jacobian(self.full_model, self.full_model_data)

        for i in range(len(self.site_ids)):
            dx = target_pos[i] - self.full_model_data.site(self.site_ids[i]).xpos
            self.twists[i][:3] = Kpos * dx / self.integration_dt
            mujoco.mju_mat2Quat(self.site_quats[i], self.full_model_data.site(self.site_ids[i]).xmat)
            mujoco.mju_negQuat(self.site_quat_conjs[i], self.site_quats[i])
            mujoco.mju_mulQuat(self.error_quats[i], target_quat_wxyz[i], self.site_quat_conjs[i])
            mujoco.mju_quat2Vel(self.twists[i][3:], self.error_quats[i], 1.0)
            self.twists[i][3:] *= Kori / self.integration_dt

        self.twist = np.hstack(self.twists)
        diag = self.damping**2 * np.eye(len(self.twist))
        eye = np.eye(len(self.dof_ids))
        # basically dq = J^{-1} dx. This formulation is nicer since m is small in dimensionality.
        self.dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, self.twist)
        # Nullspace control: intuitively, (eye - np.linalg.pinv(jac) @ jac)
        # projects dqs into the nullspace of the Jacobian, which intuitively
        # only allows movements that don't affect the task space.
        dq_null = (eye - np.linalg.pinv(jac) @ jac) @ (self.Kn * (self.q0 - self.full_model_data.qpos[self.dof_ids]))
        self.dq += dq_null

        if self.max_dq > 0:
            dq_abs_max = np.abs(self.dq).max()
            if dq_abs_max > self.max_dq:
                self.dq *= self.max_dq / dq_abs_max

        torso_joint_ids = [self.full_model.joint(name).id for name in self.joint_names if "torso" in name]
        if len(torso_joint_ids) > 0 and self.max_dq_torso > 0:
            dq_torso = self.dq[torso_joint_ids]
            dq_torso_abs_max = np.abs(dq_torso).max()
            if dq_torso_abs_max > self.max_dq_torso:
                dq_torso *= self.max_dq_torso / dq_torso_abs_max
            self.dq[torso_joint_ids] = dq_torso

        # get the desired joint angles by integrating the desired joint velocities
        self.q_des = self.full_model_data.qpos[self.dof_ids].copy()
        # mujoco.mj_integratePos(self.full_model, self.q_des, self.dq, self.integration_dt)
        self.q_des += self.dq * self.integration_dt  # manually integrate q_des

        pre_clip_error = np.inf
        post_clip_error = np.inf

        if self.debug and self.debug_iter % 10 == 0:
            # compare q_des's forward kinematics with target_pos
            integrated_pos: Dict[str, np.ndarray] = self.forward_kinematics(self.q_des)
            integrated_pos_np = np.array([integrated_pos[site] for site in integrated_pos])
            pre_clip_error = np.linalg.norm(target_pos - integrated_pos_np)
            ROBOSUITE_DEFAULT_LOGGER.info(f"IK error before clipping based on joint ranges: {pre_clip_error}")
            # self.pre_clip_errors.append(pre_clip_error)

        # Set the control signal.
        np.clip(self.q_des, *self.full_model.jnt_range[self.dof_ids].T, out=self.q_des)

        if self.debug and self.debug_iter % 10 == 0:
            # compare self.q_des's forward kinematics with target_pos
            integrated_pos: Dict[str, np.ndarray] = self.forward_kinematics(self.q_des)
            integrated_pos_np = np.array([integrated_pos[site] for site in integrated_pos])
            post_clip_error = np.linalg.norm(target_pos - integrated_pos_np)
            ROBOSUITE_DEFAULT_LOGGER.info(f"IK error after clipping based on joint ranges: {post_clip_error}")

        self.debug_iter += 1

        return self.q_des
