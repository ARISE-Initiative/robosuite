"""
***********************************************************************************


NOTE: IK is only supported for the following robots:

:Baxter:
:Sawyer:
:Panda:

Attempting to run IK with any other robot will raise an error!

***********************************************************************************
"""

from typing import Dict, List, Optional, Union

import numpy as np

import mujoco
import robosuite.utils.transform_utils as T
from robosuite.controllers.generic.joint_pos import JointPositionController
from robosuite.utils.control_utils import *

from robosuite.utils.binding_utils import MjSim

# Dict of supported ik robots
SUPPORTED_IK_ROBOTS = {"Baxter", "Sawyer", "Panda", "GR1FixedLowerBody"}


class InverseKinematicsController(JointPositionController):
    """
    Controller for controlling robot arm via inverse kinematics. Allows position and orientation control of the
    robot's end effector.

    We use differential inverse kinematics with posture control in the null space posture control
    to generate joint positions (see https://github.com/kevinzakka/mjctrl) which are fed to the JointPositionController.

    NOTE: Control input actions are assumed to be relative to the current position / orientation of the end effector
    and are taken as the array (x_dpos, y_dpos, z_dpos, x_rot, y_rot, z_rot).

    However, confusingly, x_dpos, y_dpos, z_dpos are relative to the mujoco world frame, while x_rot, y_rot, z_rot are
    relative to the current end effector frame.

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        ref_name: Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        robot_name (str): Name of robot being controlled. Can be {"Sawyer", "Panda", or "Baxter"}

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        eef_rot_offset (4-array): Quaternion (x,y,z,w) representing rotational offset between the final
            robot arm link coordinate system and the end effector coordinate system (i.e: the gripper)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        ik_pos_limit (float): Limit (meters) above which the magnitude of a given action's
            positional inputs will be clipped

        ik_ori_limit (float): Limit (radians) above which the magnitude of a given action's
            orientation inputs will be clipped

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current state to
            the goal state during each timestep between inputted actions

        converge_steps (int): How many iterations to run the pybullet inverse kinematics solver to converge to a
            solution

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Unsupported robot]
    """

    def __init__(
        self,
        sim,
        ref_name: Union[List[str], str],
        joint_indexes,
        robot_name,
        actuator_range,
        eef_rot_offset=None,
        bullet_server_id=0,
        policy_freq=20,
        load_urdf=True,
        ik_pos_limit=None,
        ik_ori_limit=None,
        interpolator_pos=None,
        interpolator_ori=None,
        converge_steps=5,
        kp: int = 100,
        kv: int = 10,
        **kwargs,
    ):
        # Run sueprclass inits
        super().__init__(
            sim=sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            input_max=1,
            input_min=-1,
            output_max=1,
            output_min=-1,
            kp=kp,
            kv=kv,
            policy_freq=policy_freq,
            velocity_limits=[-1, 1],
            **kwargs,
        )

        # Verify robot is supported by IK
        assert robot_name in SUPPORTED_IK_ROBOTS, (
            "Error: Tried to instantiate IK controller for unsupported robot! "
            "Inputted robot: {}, Supported robots: {}".format(
                robot_name, SUPPORTED_IK_ROBOTS
            )
        )

        # Initialize ik-specific attributes
        self.robot_name = robot_name  # Name of robot (e.g.: "Panda", "Sawyer", etc.)

        # Override underlying control dim
        self.control_dim = 6

        self.rest_poses = None

        self.num_ref_sites = 1 if isinstance(ref_name, str) else len(ref_name)
        # Set the reference robot target pos / orientation (to prevent drift / weird ik numerical behavior over time)
        if self.num_ref_sites == 1:
            self.reference_target_pos = self.ref_pos
            self.reference_target_orn = T.mat2quat(self.ref_ori_mat)
        else:
            self.reference_target_pos = self.ref_pos
            self.reference_target_orn = np.array([T.mat2quat(self.ref_ori_mat[i]) for i in range(self.num_ref_sites)])

        # Interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # Interpolator-related attributes
        self.ori_ref = None
        self.relative_ori = None

        # Set ik limits and override internal min / max
        self.ik_pos_limit = ik_pos_limit
        self.ik_ori_limit = ik_ori_limit

        # Should be in (0, 1], smaller values mean less sensitivity.
        self.user_sensitivity = 0.3

        self.i = 0

    def get_control(self, dpos=None, rotation=None, update_targets=False):
        """
        Returns joint positions to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @rotation.
        If no arguments are provided, joint positions will be computed based
        on the previously recorded target.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): a rotation matrix of shape (3, 3) corresponding
                to the desired rotation from the current orientation of the end effector.
            update_targets (bool): whether to update ik target pos / ori attributes or not

        Returns:
            np.array: a flat array of joint position commands to apply to try and achieve the desired input control.
        """
        pos_limits = np.ones(len(self.joint_index)) * -2
        neg_limits = np.ones(len(self.joint_index)) * 2
        self.velocity_limits = [pos_limits, neg_limits]
        self.velocity_limits = [-1, 1]
        positions = InverseKinematicsController.compute_joint_positions(
            self.sim,
            self.initial_joint,
            self.joint_index,
            self.ref_name,
            self.control_freq,
            self.velocity_limits,
            dpos,
            rotation,
            jac=self.J_full,
        )

        return positions

    @staticmethod
    def compute_joint_positions(
        sim: MjSim,
        initial_joint: np.ndarray,
        joint_indices: np.ndarray,
        ref_name: Union[List[str], str],
        control_freq: float,
        velocity_limits: Optional[np.ndarray],
        dpos: Optional[np.ndarray] = None,
        drot: Optional[np.ndarray] = None,
        Kn: Optional[np.ndarray] = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
        damping_pseudo_inv: float = 0.05,
        Kpos: float = 0.95,
        Kori: float = 0.95,
        jac: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns joint positions to control the robot after the target end effector
        position and orientation are updated from arguments @dpos and @drot.
        If no arguments are provided, joint positions will be set as zero.

        Args:
            sim (MjSim): The simulation object.
            initial_joint (np.array): Initial joint positions.
            joint_indices (np.array): Indices of the joints.
            ref_name: Reference site name.
            control_freq (float): Control frequency.
            velocity_limits (np.array): Limits on joint velocities.
            dpos (Optional[np.array]): Desired change in end-effector position.
            drot (Optional[np.array]): Desired change in orientation for the reference site (e.g end effector).
            update_targets (bool): Whether to update ik target pos / ori attributes or not.
            Kpos (float): Position gain. Between 0 and 1.
                0 means no movement, 1 means move end effector to desired position in one integration step.
            Kori (float): Orientation gain. Between 0 and 1.
            jac (Optional[np.array]): Precomputed jacobian matrix.

        Returns:
            np.array: A flat array of joint position commands to apply to try and achieve the desired input control.
        """

        def get_Kn(joint_names: List[str], weight_dict: Dict[str, float]) -> np.ndarray:
            return np.array([weight_dict.get(joint, 1.0) for joint in joint_names])

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
        # Kn = get_Kn(joint_names, nullspace_joint_weights)

        if (dpos is not None) and (drot is not None):
            max_angvel = velocity_limits[1] if velocity_limits is not None else 0.7
            integration_dt: float = 1 / control_freq
            integration_dt = 0.1


            q0 = initial_joint
            dof_ids = joint_indices

            num_ref_sites = 1 if isinstance(ref_name, str) else len(ref_name)
            if num_ref_sites == 1:
                twist = np.zeros(6)
                error_quat = np.zeros(4)
                diag = damping_pseudo_inv ** 2 * np.eye(len(twist))
                eye = np.eye(len(joint_indices))

                jac = np.zeros((6, sim.model.nv), dtype=np.float64)
                twist = np.zeros(6)
                error_quat = np.zeros(4)
    
                twist[:3] = Kpos * dpos / integration_dt
                mujoco.mju_mat2Quat(error_quat, drot.reshape(-1))
                mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
                twist[3:] *= Kori / integration_dt

                mujoco.mj_jacSite(
                    sim.model._model,
                    sim.data._data,
                    jac[:3],
                    jac[3:],
                    sim.data.site(ref_name).id,
                )

                jac = jac[:, dof_ids]

                dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)
                dq += (eye - np.linalg.pinv(jac) @ jac) @ (
                    Kn * (q0 - sim.data.qpos[dof_ids])
                )

                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

                q_des = sim.data.qpos[dof_ids] + dq * integration_dt
                dq = q_des  # hack for now
                return dq
            else:
                from robosuite.scripts.diffik_nullspace import RobotController

                model = sim.model._model
                data = sim.data._data
                # TODO(klin): get the correct indexing for the torque output --- probably need the correct joint names for our case that we care about -- find how to get that?
                # find way to get correct joint names perhaps from _ref_joints_indexes_dict
                joint_names = [model.joint(i).name for i in range(model.njnt) if model.joint(i).type != 0 and ("gripper0" not in model.joint(i).name)]  # Exclude fixed joints
                body_names = [model.body(i).name for i in range(model.nbody) if model.body(i).name not in {"world", "base", "target"}]

                def get_Kn(joint_names: List[str], weight_dict: Dict[str, float]) -> np.ndarray:
                    return np.array([weight_dict.get(joint, 1.0) for joint in joint_names])

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
                Kn = get_Kn(joint_names, nullspace_joint_weights)
                end_effector_sites = [ "gripper0_left_grip_site", "gripper0_right_grip_site"]
                robot_config =  {
                    'end_effector_sites': end_effector_sites,
                    'body_names': body_names,
                    'joint_names': joint_names,
                    'mocap_bodies': [],
                    'initial_keyframe': 'home',
                    'nullspace_gains': Kn
                }
                robot = RobotController(model, data, robot_config, input_type="mocap", debug=False)
                target_pos = np.array([[-0.419,  0.28 ,  1.11 ],
                                        [-0.419, -0.279,  1.11 ]])
                target_pos[0] += dpos
                target_ori = np.array([[-0.465, -0.46 ,  0.54 , -0.53 ],
                                        [ 0.548, -0.535,  0.487,  0.419]])

                integration_dt = 0.1
                damping = 5e-2
                Kpos = 0.95
                Kori = 0.95

                max_actuation_val = 100
                robot.solve_ik(
                    target_pos=target_pos, 
                    target_ori=target_ori, 
                    damping=damping,
                    integration_dt=integration_dt, 
                    max_actuation_val=max_actuation_val,
                    Kpos=Kpos, 
                    Kori=Kori,
                    update_sim=False,
                )
                return robot.q_des

        return np.zeros(len(joint_indices))

    def set_goal(self, delta, set_ik=None):
        """
        Sets the internal goal state of this controller based on @delta

        Note that this controller wraps a VelocityController, and so determines the desired velocities
        to achieve the inputted pose, and sets its internal setpoint in terms of joint velocities

        TODO: Add feature so that using @set_ik automatically sets the target values to these absolute values

        Args:
            delta (Iterable): Desired relative position / orientation goal state
            set_ik (Iterable): If set, overrides @delta and sets the desired global position / orientation goal state
        """
        # Update state
        self.update(force=True)  # force because new_update = True only set in super().run_controller()


        try:
            (dpos, dquat) = self._clip_ik_input(delta[:3], delta[3:7])
        except:
            import ipdb; ipdb.set_trace()
            # Get requested delta inputs if we're using interpolators
            (dpos, dquat) = self._clip_ik_input(delta[:3], delta[3:7])

        # Set interpolated goals if necessary
        if self.interpolator_pos is not None:
            # Absolute position goal
            self.interpolator_pos.set_goal(
                dpos * self.user_sensitivity + self.reference_target_pos
            )

        if self.interpolator_ori is not None:
            # Relative orientation goal
            self.interpolator_ori.set_goal(
                dquat
            )  # goal is the relative change in orientation
            self.ori_ref = np.array(
                self.ref_ori_mat
            )  # reference is the current orientation at start
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

        # Run ik prepropressing to convert pos, quat ori to desired positions
        requested_control = self._make_input(delta, self.reference_target_orn)

        # Compute desired velocities to achieve eef pos / ori
        positions = self.get_control(**requested_control, update_targets=True)

        # Set the goal positions for the underlying position controller
        super().set_goal(positions)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()  # force because new_update = True only set in super().run_controller()

        # Update interpolated action if necessary
        desired_pos = None
        rotation = None
        update_velocity_goal = False

        # Update interpolated goals if active
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
            update_velocity_goal = True
        else:
            desired_pos = self.reference_target_pos

        if self.interpolator_ori is not None:
            # Linear case
            if self.interpolator_ori.order == 1:
                # relative orientation based on difference between current ori and ref
                self.relative_ori = orientation_error(self.ref_ori_mat, self.ori_ref)
                ori_error = self.interpolator_ori.get_interpolated_goal()
                rotation = T.quat2mat(ori_error)
            else:
                # Nonlinear case not currently supported
                pass
            update_velocity_goal = True
        else:
            if self.num_ref_sites == 1:
                rotation = T.mat2quat(self.ref_ori_mat)
            else:
                rotation = np.array([T.mat2quat(self.ref_ori_mat[i]) for i in range(self.num_ref_sites)])

        # Only update the velocity goals if we're interpolating
        if update_velocity_goal:
            velocities = self.get_control(
                dpos=(desired_pos - self.ref_pos), rotation=rotation
            )
            super().set_goal(velocities)

        return super().run_controller()

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # Then, update the rest pose from the initial joints
        self.rest_poses = list(self.initial_joint)

    def reset_goal(self):
        """
        Resets the goal to the current pose of the robot
        """
        if self.num_ref_sites == 1:
            self.reference_target_pos = self.ref_pos
            self.reference_target_orn = T.mat2quat(self.ref_ori_mat)
        else:
            self.reference_target_pos = self.ref_pos
            self.reference_target_orn = np.array([T.mat2quat(self.ref_ori_mat[i]) for i in range(self.num_ref_sites)])


    def _clip_ik_input(self, dpos, rotation):
        """
        Helper function that clips desired ik input deltas into a valid range.

        Args:
            dpos (np.array): a 3 dimensional array corresponding to the desired
                change in x, y, and z end effector position.
            rotation (np.array): relative rotation in scaled axis angle form (ax, ay, az)
                corresponding to the (relative) desired orientation of the end effector.

        Returns:
            2-tuple:

                - (np.array) clipped dpos
                - (np.array) clipped rotation
        """
        # scale input range to desired magnitude
        # if dpos.any():
        #     dpos, _ = T.clip_translation(dpos, self.ik_pos_limit)

        # Map input to quaternion
        rotation = T.axisangle2quat(rotation)

        # # Clip orientation to desired magnitude
        # rotation, _ = T.clip_rotation(rotation, self.ik_ori_limit)

        return dpos, rotation

    def _make_input(self, action, old_quat):
        """
        Helper function that returns a dictionary with keys dpos, rotation from a raw input
        array. The first three elements are taken to be displacement in position, and a
        quaternion indicating the change in rotation with respect to @old_quat. Additionally clips @action as well

        Args:
            action (np.array) should have form: [dx, dy, dz, ax, ay, az] (orientation in
                scaled axis-angle form)
            old_quat (np.array) the old target quaternion that will be updated with the relative change in @action
        """
        # Clip action appropriately
        dpos, rotation = self._clip_ik_input(action[:3], action[3:])


        if self.num_ref_sites == 1:
            # Update reference targets
            self.reference_target_pos += dpos * self.user_sensitivity
            self.reference_target_orn = T.quat_multiply(old_quat, rotation)
        else:
            # Update reference targets
            self.reference_target_pos += dpos * self.user_sensitivity
            self.reference_target_orn = np.array([T.quat_multiply(old_quat[i], rotation) for i in range(self.num_ref_sites)])

        return {"dpos": dpos * self.user_sensitivity, "rotation": T.quat2mat(rotation)}

    @staticmethod
    def _get_current_error(current, set_point):
        """
        Returns an array of differences between the desired joint positions and current
        joint positions. Useful for PID control.

        Args:
            current (np.array): the current joint positions
            set_point (np.array): the joint positions that are desired as a numpy array

        Returns:
            np.array: the current error in the joint positions
        """
        error = current - set_point
        return error

    @property
    def control_limits(self):
        """
        The limits over this controller's action space, as specified by self.ik_pos_limit and self.ik_ori_limit
        and overriding the superclass method

        Returns:
            2-tuple:

                - (np.array) minimum control values
                - (np.array) maximum control values
        """
        max_limit = np.concatenate(
            [self.ik_pos_limit * np.ones(3), self.ik_ori_limit * np.ones(3)]
        )
        return -max_limit, max_limit

    @property
    def name(self):
        return "IK_POSE"
