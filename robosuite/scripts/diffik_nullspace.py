from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple, Literal

import mujoco
import mujoco.viewer
import numpy as np
import tyro

# from devices import KeyboardHandler
# from utils import get_joint_qpos_addr, quaternion

# TODO(klin): move these to an IK file

@dataclass
class Config:
    input_type: Literal["keyboard", "mocap", "pkl"] = "mocap"
    input_file: str = "recordings/gr1_eef_targets_gr1_w_hands_from_avp_preprocessor.pkl"
    debug: bool = False
    use_torque_actuation: bool = True
    use_fixed_mocap_targets: bool = False

    def __post_init__(self):
        if self.input_type == "pkl":
            assert self.input_file is not None, "Please provide a valid input file for pkl input type."


def load_model_and_data(model_path: str) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data

def get_Kn(joint_names: List[str], weight_dict: Dict[str, float]) -> np.ndarray:
    return np.array([weight_dict.get(joint, 1.0) for joint in joint_names])



import math
from typing import Tuple, Union

import mujoco
import numpy as np


def get_joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> Union[int, Tuple[int, int]]:
    """
    Get the qpos address for the given joint name in the provided MuJoCo model.
    
    Args:
        model (mujoco.MjModel): The MuJoCo model.
        joint_name (str): The name of the joint.

    Returns:
        Union[int, Tuple[int, int]]: The address of the joint's qpos. Returns an integer
        if the joint is 1-dimensional, otherwise returns a tuple with the start and end addresses.
    """
    joint_id = model.joint(joint_name).id
    joint_type = model.jnt_type[joint_id]
    joint_addr = model.jnt_qposadr[joint_id]

    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        ndim = 7
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        ndim = 4
    else:
        assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ndim = 1

    if ndim == 1:
        return joint_addr
    else:
        return joint_addr, joint_addr + ndim


def unit_vector(data, axis=None, out=None):
    """
    Taken from robosuite.utils.transform_utils.py

    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    E.g.:
        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True

        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True

        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True

        >>> list(unit_vector([]))
        []

        >>> list(unit_vector([1.0]))
        [1.0]

    Args:
        data (np.array): data to normalize
        axis (None or int): If specified, determines specific axis along data to normalize
        out (None or np.array): If specified, will store computation in this variable

    Returns:
        None or np.array: If @out is not specified, will return normalized vector. Otherwise, stores the output in @out
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def quaternion(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)  # Normalize the axis vector
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    return np.array([*(axis * sin_half_angle), np.cos(half_angle)])

def rotation_matrix(angle, direction, point=None):
    """
    Taken from robosuite.utils.transform_utils.py

    Returns matrix to rotate about axis defined by point and direction.

    E.g.:
        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True

        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True

        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    Args:
        angle (float): Magnitude of rotation
        direction (np.array): (ax,ay,az) axis about which to rotate
        point (None or np.array): If specified, is the (x,y,z) point about which the rotation will occur

    Returns:
        np.array: 4x4 homogeneous matrix that includes the desired rotation
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float32, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

class RobotController:
    def __init__(
        self, 
        model: mujoco.MjModel, 
        data: mujoco.MjData, 
        robot_config: Dict, 
        input_type: Literal["keyboard", "mocap", "pkl"] = "keyboard", 
        debug: bool = False, 
        input_file: Optional[str] = None
    ):
        self.model = model
        self.data = data
        self.joint_names = robot_config['joint_names']
        self.site_names = robot_config['end_effector_sites']
        self.site_ids = [self.model.site(robot_config['end_effector_sites'][i]).id for i in range(len(robot_config['end_effector_sites']))]
        self.body_ids = [self.model.body(name).id for name in robot_config['body_names']]
        self.joint_qpos_ids = np.array([get_joint_qpos_addr(model, joint_name) for joint_name in robot_config['joint_names']])
        self.dof_ids = np.array([self.model.joint(name).id for name in robot_config['joint_names']])
        # self.actuator_ids = np.array([self.model.actuator(name).id for name in robot_config['joint_names']])  # this works if actuators match joints
        self.actuator_ids = np.array([i for i in range(20)])  # TODO figure out how to not hardcode this --- currently hardcoded for GR1
        # self.key_id = self.model.key(robot_config['initial_keyframe']).id

        self.jac_temps: List[np.ndarray] = [np.zeros((6, self.model.nv)) for _ in range(len(self.site_ids))]
        self.twists: List[np.ndarray] = [np.zeros(6) for _ in range(len(self.site_ids))]
        self.site_quats: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]
        self.site_quat_conjs: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]
        self.error_quats: List[np.ndarray] = [np.zeros(4) for _ in range(len(self.site_ids))]

        self.i = 0
        self.debug = debug

        self.input_type = input_type
        if input_type == "keyboard":
            self.keyboard_handler = KeyboardHandler()
            self.pos_sensitivity = 0.1
            self.rot_sensitivity = 0.1
        elif input_type == "mocap":
            self.mocap_ids = [self.model.body(name).mocapid[0] for name in robot_config['mocap_bodies']]
        elif input_type == "pkl":
            self.mocap_ids = [self.model.body(name).mocapid[0] for name in robot_config['mocap_bodies']]
            self.pkl_t = 0
            import pickle
            with open(input_file, 'rb') as f:
                input_file = pickle.load(f)
            self.history = input_file

        # Nullspace control
        # self.q0 = self.model.key(robot_config['initial_keyframe']).qpos
        # all zeros for now
        self.q0 = np.zeros(len(self.dof_ids))
        self.Kn = np.array(robot_config['nullspace_gains'])

        # Initialize error and error_dot
        self.error_prev = np.zeros_like(self.q0)
        self.error_dot = np.zeros_like(self.q0)

    def apply_gravity_compensation(self):
        self.model.body_gravcomp[self.body_ids] = 1.0

    def reset_to_initial_state(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)

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

        if self.input_type == "keyboard":
            keys = self.keyboard_handler.get_keyboard_input()
        
            if 'right' in keys:
                target_pos[0, 1] += self.pos_sensitivity
            if 'left' in keys:
                target_pos[0, 1] -= self.pos_sensitivity
            if 'up' in keys:
                target_pos[0, 0] -= self.pos_sensitivity
            if 'down' in keys:
                target_pos[0, 0] += self.pos_sensitivity
            if 'up_z' in keys:
                target_pos[0, 2] += self.pos_sensitivity
            if 'down_z' in keys:
                target_pos[0, 2] -= self.pos_sensitivity

            dquat = np.array([1.0, 0.0, 0.0, 0.0])
            if 'e' in keys:
                dquat = quaternion(angle=0.1 * self.rot_sensitivity, axis=np.array([1.0, 0.0, 0.0]))
            elif 'r' in keys:
                dquat = quaternion(angle=-0.1 * self.rot_sensitivity, axis=np.array([1.0, 0.0, 0.0]))
            elif 'y' in keys:
                dquat = quaternion(angle=0.1 * self.rot_sensitivity, axis=np.array([0.0, 1.0, 0.0]))
            elif 'h' in keys:
                dquat = quaternion(angle=-0.1 * self.rot_sensitivity, axis=np.array([0.0, 1.0, 0.0]))
            elif 'p' in keys:
                dquat = quaternion(angle=0.1 * self.rot_sensitivity, axis=np.array([0.0, 0.0, 1.0]))
            elif 'o' in keys:
                dquat = quaternion(angle=-0.1 * self.rot_sensitivity, axis=np.array([0.0, 0.0, 1.0]))

            mujoco.mju_mulQuat(target_ori[0], target_ori[0], dquat)
        elif self.input_type == "mocap":
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
        target_pos: np.ndarray,
        target_ori: np.ndarray,
        damping: float, 
        integration_dt: float, 
        max_actuation_val: float, 
        use_torque_actuation: bool = True, 
        Kpos: float = 0.95, 
        Kori: float = 0.95,
        update_sim: bool = True,
    ):
        jac = self._compute_jacobian(self.model, self.data)

        for i in range(len(self.site_ids)):
            dx = target_pos[i] - self.data.site(self.site_ids[i]).xpos
            self.twists[i][:3] = Kpos * dx / integration_dt
            mujoco.mju_mat2Quat(self.site_quats[i], self.data.site(self.site_ids[i]).xmat)
            mujoco.mju_negQuat(self.site_quat_conjs[i], self.site_quats[i])
            mujoco.mju_mulQuat(self.error_quats[i], target_ori[i], self.site_quat_conjs[i])
            mujoco.mju_quat2Vel(self.twists[i][3:], self.error_quats[i], 1.0)
            self.twists[i][3:] *= Kori / integration_dt

        self.twist = np.hstack(self.twists)
        diag = damping ** 2 * np.eye(len(self.twist))
        eye = np.eye(len(self.dof_ids))
        # basically dq = J^{-1} dx. This formulation is nicer since m is small in dimensionality.
        self.dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, self.twist)
        # Nullspace control: intuitively, (eye - np.linalg.pinv(jac) @ jac) 
        # projects dqs into the nullspace of the Jacobian, which intuitively 
        # only allows movements that don't affect the task space.
        dq_null = (eye - np.linalg.pinv(jac) @ jac) @ \
            (self.Kn * (self.q0 - self.data.qpos[self.dof_ids]))
        self.dq += dq_null

        if max_actuation_val > 0:
            dq_abs_max = np.abs(self.dq).max()
            if dq_abs_max > max_actuation_val:
                self.dq *= max_actuation_val / dq_abs_max

        # get the desired joint angles by integrating the desired joint velocities
        self.q_des = self.data.qpos[self.dof_ids].copy()
        # mujoco.mj_integratePos(self.model, q_des, dq, integration_dt)
        self.q_des += self.dq * integration_dt  # manually integrate q_des

        pre_clip_error = np.inf
        post_clip_error = np.inf

        if self.debug and self.i % 10 == 0:
            # compare q_des's forward kinematics with target_pos
            integrated_pos: Dict[str, np.ndarray] = self.forward_kinematics(self.q_des)
            integrated_pos_np = np.array([integrated_pos[site] for site in integrated_pos])
            print(f"internal error pre clip: {np.linalg.norm(target_pos - integrated_pos_np)}")
            pre_clip_error = np.linalg.norm(target_pos - integrated_pos_np)

        self.i += 1

        if use_torque_actuation:
            # Using torque based control (motor actuators) + pos + vel joint control doesn't work for my code
            Kp = 1000
            Kd = 100

            position_error = self.q_des - self.data.qpos[self.dof_ids]
            velocity_error = -self.data.qvel[self.dof_ids]

            if self.debug and self.i % 100 == 0:
                if np.all(self.twist == 0):
                    print(f"position_error: {np.max(np.abs(position_error))}")
                    print(f"velocity_error: {np.max(np.abs(velocity_error))}")
                else:
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
            # this indexing mightn't always work for freejoint
            np.clip(self.q_des, *self.model.jnt_range[self.dof_ids].T, out=self.q_des)
            print(self.i)
            # if self.debug and self.i % 10 == 0:

            if self.debug:
                # compare self.q_des's forward kinematics with target_pos
                integrated_pos: Dict[str, np.ndarray] = self.forward_kinematics(self.q_des)
                integrated_pos_np = np.array([integrated_pos[site] for site in integrated_pos])
                post_clip_error = np.linalg.norm(target_pos - integrated_pos_np)
                print(f"internal error post clip: {post_clip_error}")

            print(f"diff in error: {post_clip_error - pre_clip_error}")
            # check if we're outside the joint limits
            if False or np.any(self.q_des < self.model.jnt_range[self.dof_ids][:, 0]) or np.any(self.q_des > self.model.jnt_range[self.dof_ids][:, 1]):
                print("Joint limits exceeded! Not updating control signal.")
                # import ipdb; ipdb.set_trace()
                # get dof ids exceeding joint limits
                exceeding_ids = np.where((self.q_des < self.model.jnt_range[self.dof_ids][:, 0]) | (self.q_des > self.model.jnt_range[self.dof_ids][:, 1]))[0]
                # get joint names
                exceeding_joint_names = [self.model.joint(i).name for i in self.dof_ids[exceeding_ids]]
                for joint_name in exceeding_joint_names:
                    if "gripper" not in joint_name:
                        print(f"Joint {joint_name} has exceeded its limits.")
                        import ipdb; ipdb.set_trace()
                print(f"Exceeding joint limits for {exceeding_joint_names}")
                # import ipdb; ipdb.set_trace()
                # return

            if update_sim:
                self.data.ctrl[self.actuator_ids] = self.q_des[self.dof_ids]
            else:
                return self.q_des[self.dof_ids]


def circle(t: float, r: float, h: float, k: float, f: float, z: float) -> np.ndarray:
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    return np.array([x, y, z])  # Fixed z-coordinate

def main(cfg: Config):
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    print("Note: for GR1, using diff IK with nullspace control has residual movement, while using Panda works fine."
        "We likely want to set goals based on absolute poses from which we derived the dx values.")
    model = "gr1"
    if model == "ur5e":
        model_path = "universal_robots_ur5e/scene.xml"
        end_effector_sites = ["attachment_site"]
    elif model == "panda":
        model_path = "franka_emika_panda/scene.xml"
        end_effector_sites = ["attachment_site"]
        nullspace_joint_weights = {}
    elif model == "gr1":
        model_path = "gr1/scene.xml"
        # end_effector_sites = [ "robot0_l_wrist_site", "robot0_r_wrist_site"]  # better for mocap tracking
        end_effector_sites = [ "gripper0_left_grip_site", "gripper0_right_grip_site"]
        # end_effector_sites = [ "left_eef_site", "right_eef_site"]

        # Gains adapted from https://github.com/nvglab/GearTeleop/blob/main/configs/teleop_avp_gr1_no_head_cost_unlimited.yaml
        # Short term solution for keeping robot upright and looking okay for table top tasks.
        nullspace_joint_weights = {
            "robot0_torso_waist_yaw": 100.0,
            "robot0_torso_waist_pitch": 100.0,
            "robot0_torso_waist_roll": 500.0,
            "robot0_l_shoulder_pitch": 4.0,
            "robot0_r_shoulder_pitch": 4.0,
            "robot0_l_shoulder_roll": 3.0,
            "robot0_r_shoulder_roll": 3.0,
            "robot0_l_shoulder_yaw": 2.0,
            "robot0_r_shoulder_yaw": 2.0,
        }        
    else:
        raise ValueError(f"Invalid model {model}")
    
    model, data = load_model_and_data(model_path)

    joint_names = [model.joint(i).name for i in range(model.njnt) if model.joint(i).type != 0]  # Exclude fixed joints
    body_names = [model.body(i).name for i in range(model.nbody) if model.body(i).name not in {"world", "base", "target"}]
    Kn = get_Kn(joint_names, nullspace_joint_weights)
    # mocap_bodies = ["left_target", "right_target"]
    mocap_bodies = []
    robot_config = {
        'end_effector_sites': end_effector_sites,
        'body_names': body_names,
        'joint_names': joint_names,
        'mocap_bodies': mocap_bodies,
        'initial_keyframe': 'home',
        'nullspace_gains': Kn
    }

    print(f"Use torque actuation: {cfg.use_torque_actuation}. Ensure correct actuators are set in the XML file.")
    robot = RobotController(model, data, robot_config, input_type=cfg.input_type, debug=cfg.debug, input_file=cfg.input_file)
    
    integration_dt = 0.1
    damping = 5e-2
    dt = 0.002
    if cfg.use_torque_actuation:
        max_actuation_val = 100
        gravity_compensation = False
    else:
        gravity_compensation = True
        max_actuation_val = 1

    Kpos = 0.95
    Kori = 0.95

    robot.model.opt.timestep = dt
    
    if gravity_compensation:
        robot.apply_gravity_compensation()

    with mujoco.viewer.launch_passive(model=robot.model, data=robot.data, show_left_ui=False, show_right_ui=False) as viewer:
        robot.reset_to_initial_state()
        mujoco.mjv_defaultFreeCamera(robot.model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # step to get mass values, etc.
        mujoco.mj_step(model, data)

        while viewer.is_running():
            step_start = time.time()

            if cfg.input_type == "mocap" and cfg.use_fixed_mocap_targets:
                left_target_pos = circle(robot.data.time, 0.1, -0.33, 0.22, 0.5, 1.0)
                right_target_pos = circle(robot.data.time, 0.1, -0.33, -0.22, 0.5, 1.0)
                robot.set_target_positions([left_target_pos, right_target_pos])

            target_pos, target_ori = robot.get_targets()
            robot.solve_ik(target_pos, target_ori, damping, integration_dt, max_actuation_val, cfg.use_torque_actuation, Kpos, Kori)

            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    tyro.cli(main)
