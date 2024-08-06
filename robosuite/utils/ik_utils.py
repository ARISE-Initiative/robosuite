from typing import Dict, List, Optional, Literal, Tuple

import mujoco
import mujoco.viewer
import numpy as np

import robosuite.utils.transform_utils as T


def get_Kn(joint_names: List[str], weight_dict: Dict[str, float]) -> np.ndarray:
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
        input_file: Optional[str] = None,
        input_rotation_repr: Literal["quat_wxyz", "axis_angle"] = "axis_angle",
    ):
        self.model = model
        self.data = data

        self.damping = damping
        self.integration_dt = integration_dt
        self.max_dq = max_dq
        self.max_dq_torso = max_dq_torso

        self.joint_names = robot_config['joint_names']
        self.site_names = robot_config['end_effector_sites']
        self.site_ids = [self.model.site(robot_config['end_effector_sites'][i]).id for i in range(len(robot_config['end_effector_sites']))]
        self.dof_ids = np.array([self.model.joint(name).id for name in robot_config['joint_names']])
        # self.actuator_ids = np.array([self.model.actuator(name).id for name in robot_config['joint_names']])  # this works if actuators match joints
        self.actuator_ids = np.array([i for i in range(20)])  # TODO no hardcode; cur for GR1; model.nu is 32
        self.key_id = self.model.key(robot_config['initial_keyframe']).id if 'initial_keyframe' in robot_config else None

        self.jac_temps: List[np.ndarray] = [np.zeros((6, self.model.nv)) for _ in range(len(self.site_ids))]
        self.twists: List[np.ndarray] = [np.zeros(6) for _ in range(len(self.site_ids))]
        self.site_quats: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]
        self.site_quat_conjs: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]
        self.error_quats: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]

        self.input_rotation_repr = input_rotation_repr
        ROTATION_REPRESENTATION_DIMS: Dict[str, int] = {"quat_wxyz": 4, "axis_angle": 3}
        self.rot_dim = ROTATION_REPRESENTATION_DIMS[input_rotation_repr]
        self.pos_dim = 3
        self.control_dim = len(self.site_names) * (self.pos_dim + self.rot_dim)
        # hardcoded control limits for now
        self.control_limits = np.array([-np.inf] * self.control_dim), np.array([np.inf] * self.control_dim)
        self.i = 0
        self.debug = debug

        self.input_type = input_type
        if input_type == "mocap":
            self.mocap_ids = [self.model.body(name).mocapid[0] for name in robot_config['mocap_bodies']]
        elif input_type == "pkl":
            self.mocap_ids = [self.model.body(name).mocapid[0] for name in robot_config['mocap_bodies']]
            self.pkl_t = 0
            import pickle
            with open(input_file, 'rb') as f:
                input_file = pickle.load(f)
            self.history = input_file

        # Nullspace control
        self.q0 = np.zeros(len(self.dof_ids))
        self.Kn = np.array(robot_config['nullspace_gains'])

        # Initialize error and error_dot
        self.error_prev = np.zeros_like(self.q0)
        self.error_dot = np.zeros_like(self.q0)

    def action_split_indexes(self) -> Dict[str, Tuple[int, int]]:
        action_split_indexes: Dict[str, Tuple[int, int]] = {}
        previous_idx = 0

        for site_name in self.site_names:
            total_dim = self.pos_dim + self.rot_dim
            last_idx = previous_idx + total_dim
            action_split_indexes[site_name + "_pos"] = (previous_idx, previous_idx + self.pos_dim)
            action_split_indexes[site_name + f"_{self.input_rotation_repr}"] = (previous_idx + self.pos_dim, last_idx)
            previous_idx = last_idx

        return action_split_indexes

    def reset_to_initial_state(self):
        if self.key_id is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        else:
            print("No initial keyframe set. Skipping reset.")

    def set_target_positions(self, target_positions: List):
        for i, pos in enumerate(target_positions):
            self.data.mocap_pos[self.mocap_ids[i]] = pos

    def set_target_rotations(self, target_rotations: List):
        for i, quat_wxyz in enumerate(target_rotations):
            self.data.mocap_quat[self.mocap_ids[i]] = quat_wxyz

    def get_targets(self):
        # by default set target to current site position
        target_pos = np.array([self.data.site(site_id).xpos for site_id in self.site_ids])
        target_ori_mat = np.array([self.data.site(site_id).xmat for site_id in self.site_ids])
        target_ori = np.array([np.ones(4) for _ in range(len(self.site_ids))])
        [mujoco.mju_mat2Quat(target_ori[i], target_ori_mat[i]) for i in range(len(self.site_ids))]
        
        if self.input_type == "mocap":
            for i in range(len(self.site_ids)):
                target_pos[i] = self.data.mocap_pos[self.mocap_ids[i]]
                target_ori[i] = self.data.mocap_quat[self.mocap_ids[i]]
        elif self.input_type == "pkl":
            left_index, right_index = 0, 1
            self.pkl_t += 1 / 10  # hardcoded pkl playback freq for now
            if int(self.pkl_t) >= len(self.history['left_eef_pos']):
                print("Reached end of pkl file. Exiting.")
                exit()
            target_pos[left_index] = self.history['left_eef_pos'][int(self.pkl_t)]
            target_pos[right_index] = self.history['right_eef_pos'][int(self.pkl_t)]
            target_ori[left_index] = self.history['left_eef_quat_wxyz'][int(self.pkl_t)]
            target_ori[right_index] = self.history['right_eef_quat_wxyz'][int(self.pkl_t)]
            self.data.mocap_pos[self.mocap_ids[left_index]] = self.history['left_eef_pos'][int(self.pkl_t)]
            self.data.mocap_pos[self.mocap_ids[right_index]] = self.history['right_eef_pos'][int(self.pkl_t)]
            self.data.mocap_quat[self.mocap_ids[left_index]] = self.history['left_eef_quat_wxyz'][int(self.pkl_t)]
            self.data.mocap_quat[self.mocap_ids[right_index]] = self.history['right_eef_quat_wxyz'][int(self.pkl_t)]
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
        data = mujoco.MjData(self.model)
        data.qpos[self.dof_ids] = qpos
        mujoco.mj_kinematics(self.model, data)
        return {name: data.site(site_id).xpos for name, site_id in zip(self.site_names, self.site_ids)}

    def solve_ik(
        self,
        target_action: np.ndarray,
        max_dq_torso: float = 0.2,  # hardcoded for GR1; else torso shakes
        use_torque_actuation: bool = False, 
        Kpos: float = 0.95,
        Kori: float = 0.95,
        update_sim: bool = False,
    ):
        target_action = target_action.reshape(len(self.site_names), -1)
        target_pos = target_action[:, :self.pos_dim]
        target_ori = target_action[:, self.pos_dim:]

        jac = self._compute_jacobian(self.model, self.data)

        if self.input_rotation_repr == "axis_angle":
            target_quat_wxyz = np.array([np.roll(T.axisangle2quat(target_ori[i]), 1) for i in range(len(target_ori))])
        elif self.input_rotation_repr == "mat":
            target_quat_wxyz = np.array([np.roll(T.mat2quat(target_ori[i])) for i in range(len(target_ori))])
        elif self.input_rotation_repr == "quat_wxyz":
            target_quat_wxyz = target_ori

        for i in range(len(self.site_ids)):
            dx = target_pos[i] - self.data.site(self.site_ids[i]).xpos
            self.twists[i][:3] = Kpos * dx / self.integration_dt
            mujoco.mju_mat2Quat(self.site_quats[i], self.data.site(self.site_ids[i]).xmat)
            mujoco.mju_negQuat(self.site_quat_conjs[i], self.site_quats[i])
            mujoco.mju_mulQuat(self.error_quats[i], target_quat_wxyz[i], self.site_quat_conjs[i])
            mujoco.mju_quat2Vel(self.twists[i][3:], self.error_quats[i], 1.0)
            self.twists[i][3:] *= Kori / self.integration_dt

        self.twist = np.hstack(self.twists)
        diag = self.damping ** 2 * np.eye(len(self.twist))
        eye = np.eye(len(self.dof_ids))
        # basically dq = J^{-1} dx. This formulation is nicer since m is small in dimensionality.
        self.dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, self.twist)
        # Nullspace control: intuitively, (eye - np.linalg.pinv(jac) @ jac) 
        # projects dqs into the nullspace of the Jacobian, which intuitively 
        # only allows movements that don't affect the task space.
        dq_null = (eye - np.linalg.pinv(jac) @ jac) @ \
            (self.Kn * (self.q0 - self.data.qpos[self.dof_ids]))
        self.dq += dq_null

        if self.max_dq > 0:
            dq_abs_max = np.abs(self.dq).max()
            if dq_abs_max > self.max_dq:
                self.dq *= self.max_dq / dq_abs_max

        torso_joint_ids = [self.model.joint(name).id for name in self.joint_names if "torso" in name]
        if len(torso_joint_ids) > 0 and self.max_dq_torso > 0:
            dq_torso = self.dq[torso_joint_ids]
            dq_torso_abs_max = np.abs(dq_torso).max()
            if dq_torso_abs_max > self.max_dq_torso:
                dq_torso *= self.max_dq_torso / dq_torso_abs_max
            self.dq[torso_joint_ids] = dq_torso

        # get the desired joint angles by integrating the desired joint velocities
        self.q_des = self.data.qpos[self.dof_ids].copy()
        # mujoco.mj_integratePos(self.model, q_des, dq, self.integration_dt)
        self.q_des += self.dq * self.integration_dt  # manually integrate q_des

        pre_clip_error = np.inf
        post_clip_error = np.inf

        if self.debug and self.i % 10 == 0:
            # compare q_des's forward kinematics with target_pos
            integrated_pos: Dict[str, np.ndarray] = self.forward_kinematics(self.q_des)
            integrated_pos_np = np.array([integrated_pos[site] for site in integrated_pos])
            pre_clip_error = np.linalg.norm(target_pos - integrated_pos_np)
            print(f"internal error pre clip: {pre_clip_error}")

        self.i += 1

        if use_torque_actuation:
            # Using torque based control (motor actuators) + pos + vel joint control doesn't work for my code
            Kp = 1000
            Kd = 100
            self.Kp = Kp
            self.Kd = Kd

            position_error = self.q_des - self.data.qpos[self.dof_ids]
            velocity_error = -self.data.qvel[self.dof_ids]

            self.pos_curr = self.data.qpos[self.dof_ids].copy()
            self.position_error = position_error
            self.velocity_error = velocity_error

            if self.debug and self.i % 100 == 0:
                print(f"position_error: {np.max(np.abs(position_error))}")
                print(f"velocity_error: {np.max(np.abs(velocity_error))}")

            desired_torque = Kp * position_error + Kd * velocity_error
            self.gravity_compensation = self.data.qfrc_bias[self.dof_ids]

            # scale desired torque by mass matrix if doing torque control!
            mass_matrix = np.zeros(shape=(self.model.nv, self.model.nv), dtype=np.float64, order="C")
            mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
            mass_matrix = np.reshape(mass_matrix, (len(self.data.qvel), len(self.data.qvel)))
            self.mass_matrix = mass_matrix[self.dof_ids][:, self.dof_ids]
            torque = np.dot(self.mass_matrix, desired_torque) + self.gravity_compensation
            if update_sim:
                self.data.ctrl[self.actuator_ids] = torque
            else:
                return torque
        else:
            # Set the control signal.
            np.clip(self.q_des, *self.model.jnt_range[self.dof_ids].T, out=self.q_des)

            if self.debug and self.i % 10 == 0:
                # compare self.q_des's forward kinematics with target_pos
                integrated_pos: Dict[str, np.ndarray] = self.forward_kinematics(self.q_des)
                integrated_pos_np = np.array([integrated_pos[site] for site in integrated_pos])
                post_clip_error = np.linalg.norm(target_pos - integrated_pos_np)
                print(f"internal error post clip: {post_clip_error}")

                # check if we're outside the joint limits
                if np.any(self.q_des < self.model.jnt_range[self.dof_ids][:, 0]) or np.any(self.q_des > self.model.jnt_range[self.dof_ids][:, 1]):
                    # get dof ids exceeding joint limits
                    exceeding_ids = np.where((self.q_des < self.model.jnt_range[self.dof_ids][:, 0]) | (self.q_des > self.model.jnt_range[self.dof_ids][:, 1]))[0]
                    exceeding_joint_names = [self.model.joint(i).name for i in self.dof_ids[exceeding_ids]]
                    for joint_name in exceeding_joint_names:
                        if "gripper" not in joint_name:
                            print(f"Joint {joint_name} has exceeded its limits.")
            if update_sim:
                self.data.ctrl[self.actuator_ids] = self.q_des[self.dof_ids]
            else:
                return self.q_des
