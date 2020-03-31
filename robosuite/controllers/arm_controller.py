import numpy as np
import scipy
from enum import Enum

import robosuite.utils.transform_utils as T
import mujoco_py

#from baselines.baselines import logger
import logging
logger = logging.getLogger(__name__)

from mujoco_py import load_model_from_xml, MjSim, functions
from scipy.interpolate import CubicSpline


class ControllerType(str, Enum):
    POS = 'position'
    POS_ORI = 'position_orientation'
    POS_YAW = 'position_yaw'
    JOINT_IMP = 'joint_impedance'
    JOINT_TORQUE = 'joint_torque'
    JOINT_VEL = 'joint_velocity'


class Controller():
    def __init__(self,
                 control_max,
                 control_min,
                 max_action,
                 min_action,
                 control_freq=20,
                 impedance_flag=False,
                 kp_max=None,
                 kp_min=None,
                 damping_max=None,
                 damping_min=None,
                 initial_joint=None,
                 position_limits=[[0, 0, 0], [0, 0, 0]],
                 orientation_limits=[[0, 0, 0], [0, 0, 0]],
                 interpolation=None,
                 **kwargs
                 ):

        # If the action includes impedance parameters
        self.impedance_flag = impedance_flag

        # Initial joint configuration we use for the task in the null space
        self.initial_joint = initial_joint

        # Upper and lower limits to the input action (only pos/ori)
        self.control_max = control_max
        self.control_min = control_min

        # Dimensionality of the action
        self.control_dim = self.control_max.shape[0]

        if self.impedance_flag:
            impedance_max = np.hstack((kp_max, damping_max))
            impedance_min = np.hstack((kp_min, damping_min))
            self.control_max = np.hstack((self.control_max, impedance_max))
            self.control_min = np.hstack((self.control_min, impedance_min))

        # Limits to the policy outputs
        self.input_max = max_action
        self.input_min = min_action

        # This handles when the mean of max and min control is not zero -> actions are around that mean
        self.action_scale = abs(self.control_max - self.control_min) / abs(max_action - min_action)
        self.action_output_transform = (self.control_max + self.control_min) / 2.0
        self.action_input_transform = (max_action + min_action) / 2.0

        self.control_freq = control_freq  # control steps per second

        self.interpolation = interpolation

        self.ramp_ratio = 0.20  # Percentage of the time between policy timesteps used for interpolation

        self.position_limits = position_limits
        self.orientation_limits = orientation_limits

        # Initialize the remaining attributes
        self.model_timestep = None
        self.interpolation_steps = None
        self.current_position = None
        self.current_orientation_mat = None
        self.current_lin_velocity = None
        self.current_ang_velocity = None
        self.current_joint_position = None
        self.current_joint_velocity = None
        self.Jx = None
        self.Jr = None
        self.J_full = None

    def reset(self):
        """
        Resets the internal values of the controller
        """
        pass

    def transform_action(self, action):
        """
        Scale the action to go to the right min and max
        """
        action = np.clip(action, self.input_min, self.input_max)
        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

        return transformed_action

    def untransform_action(self, action):
        """
        Unscale the action - inverse of @transform_action. If an action outside
        the valid bounds is passed in, it will rescale the action to lie withiin
        the valid action bounds (so it may not exactly correspond to the true inverse).
        """
        untransformed_action = ((action - self.action_output_transform) / self.action_scale) + self.action_input_transform
        untransformed_action = np.clip(untransformed_action, self.input_min, self.input_max)
        return untransformed_action

    def update_model(self, sim, joint_index, id_name='right_hand'):
        """
        Updates the state of the robot used to compute the control command
        """
        pos_joint_index, vel_joint_index = joint_index


        self.model_timestep = sim.model.opt.timestep
        self.interpolation_steps = np.floor(self.ramp_ratio * self.control_freq / self.model_timestep)
        self.current_position = sim.data.body_xpos[sim.model.body_name2id(id_name)]
        self.current_orientation_mat = sim.data.body_xmat[sim.model.body_name2id(id_name)].reshape([3, 3])
        self.current_lin_velocity = sim.data.body_xvelp[sim.model.body_name2id(id_name)]
        self.current_ang_velocity = sim.data.body_xvelr[sim.model.body_name2id(id_name)]

        self.current_joint_position = sim.data.qpos[pos_joint_index]
        self.current_joint_velocity = sim.data.qvel[vel_joint_index]

        self.Jx = sim.data.get_body_jacp(id_name).reshape((3, -1))[:, vel_joint_index]
        self.Jr = sim.data.get_body_jacr(id_name).reshape((3, -1))[:, vel_joint_index]
        self.J_full = np.vstack([self.Jx, self.Jr])

    def update_mass_matrix(self, sim, joint_index):
        """
        Update the mass matrix.
        sim - Mujoco simulation object
        joint_index - list of joint position indices in Mujoco
        """
        pos_joint_index, vel_joint_index = joint_index

        mass_matrix = np.ndarray(shape=(len(sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(sim.model, mass_matrix, sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(sim.data.qvel), len(sim.data.qvel)))
        self.mass_matrix = mass_matrix[vel_joint_index, :][:, vel_joint_index]

    def set_goal_impedance(self, action):
        """
        Interpret the action as the intended impedance. The impedance is not set
        directly in case interpolation is enabled.
        """
        if self.use_delta_impedance:
            # clip resulting kp and damping
            self.goal_kp = np.clip(self.impedance_kp[self.action_mask] + action[self.kp_index[0]:self.kp_index[1]],
                                   self.kp_min, self.kp_max)
            self.goal_damping = np.clip(
                self.impedance_damping[self.action_mask] + action[self.damping_index[0]:self.damping_index[1]], self.damping_min,
                self.damping_max)
        else:
            # no clipped is needed here, since the action has already been scaled
            self.goal_kp = action[self.kp_index[0]:self.kp_index[1]]
            self.goal_damping = action[self.damping_index[0]:self.damping_index[1]]

    def linear_interpolate(self, last_goal, goal):
        """
        Set self.linear to be a function interpolating between last_goal and goal based on the ramp_ratio
        """
        # We interpolate to reach the commanded desired position in self.ramp_ratio % of the time we have this goal
        delta_x_per_step = (goal - last_goal) / self.interpolation_steps
        self.linear = np.array(
            [(last_goal + i * delta_x_per_step) for i in range(1, int(self.interpolation_steps) + 1)])

    def interpolate_impedance(self, starting_kp, starting_damping, goal_kp, goal_damping):
        """
        Set self.update_impedance to be a function for generating the impedance given the timestep
        """
        delta_kp_per_step = (goal_kp - starting_kp[self.action_mask]) / self.interpolation_steps
        delta_damping_per_step = (goal_damping - starting_damping[self.action_mask]) / self.interpolation_steps

        def update_impedance(index):
            if index < self.interpolation_steps - 1:
                self.impedance_kp[self.action_mask] += delta_kp_per_step
                self.impedance_damping[self.action_mask] += delta_damping_per_step

        self.update_impedance = update_impedance

    def calculate_orientation_error(self, desired, current):
        """
        This function calculates a 3-dimensional orientation error vector for use in the
        impedance controller. It does this by computing the delta rotation between the 
        inputs and converting that rotation to exponential coordinates (axis-angle
        representation, where the 3d vector is axis * angle). 

        See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
        """
        delta_rotation_mat = desired.dot(current.T)
        delta_rotation_quat = T.mat2quat(delta_rotation_mat)
        delta_rotation_axis, delta_rotation_angle = T.quat2axisangle(delta_rotation_quat)
        orientation_error = T.axisangle2vec(axis=delta_rotation_axis, angle=delta_rotation_angle)
        return orientation_error

        # def cross_product(vec1, vec2):
        #     S = np.array(([0, -vec1[2], vec1[1]],
        #                   [vec1[2], 0, -vec1[0]],
        #                   [-vec1[1], vec1[0], 0]))

        #     return np.dot(S, vec2)

        # rc1 = current[0:3, 0]
        # rc2 = current[0:3, 1]
        # rc3 = current[0:3, 2]
        # rd1 = desired[0:3, 0]
        # rd2 = desired[0:3, 1]
        # rd3 = desired[0:3, 2]

        # orientation_error = 0.5 * (cross_product(rc1, rd1) + cross_product(rc2, rd2) + cross_product(rc3, rd3))

        # return orientation_error

    def action_to_torques(self, action, policy_step):
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Returns dimensionality of the actions
        """
        dim = self.control_dim
        if self.impedance_flag:
            # Includes (stacked) state vector, kp vector, and damping vector
            dim = dim * 3
        if self.force_control:
            assert isinstance(self, PositionOrientationController)
            if isinstance(self, PositionController):
                # can only apply translational forces to arm
                dim += 3
            else:
                # can apply both translational and rotational forces to arm
                dim += 6
        return dim

    @property
    def kp_index(self):
        """
        Indices of the kp values in the action vector
        """
        start_index = self.control_dim
        end_index = start_index + self.control_dim

        if self.impedance_flag:
            return (start_index, end_index)
        else:
            return None

    @property
    def damping_index(self):
        """
        Indices of the damping ratio values in the action vector
        """
        start_index = self.kp_index[1]
        end_index = start_index + self.control_dim

        if self.impedance_flag:
            return (start_index, end_index)
        else:
            return None

    @property
    def action_mask(self):
        raise NotImplementedError


class JointTorqueController(Controller):
    """
    Class to interpret actions as joint torques
    """

    def __init__(self,
                 control_range,
                 max_action=1,
                 min_action=-1,
                 # impedance_flag= False,  ## TODO ## : Why is this commented out?
                 inertia_decoupling=False,
                 interpolation = None,
                 **kwargs
                 ):

        super(JointTorqueController, self).__init__(
            control_max=np.array(control_range),
            control_min=-1 * np.array(control_range),
            max_action=max_action,
            min_action=min_action,
            interpolation=interpolation,
            **kwargs)

        # self.use_delta_impedance = False
        self.interpolate = True
        self.last_goal = np.zeros(self.control_dim)
        self.step = 0
        self.inertia_decoupling = inertia_decoupling

    def reset(self):
        super().reset()
        self.step = 0
        self.last_goal = np.zeros(self.control_dim)

    def action_to_torques(self, action, policy_step):
        action = self.transform_action(action)

        if policy_step:
            self.step = 0
            self.goal = np.array((action))
            if self.interpolation and self.interpolation != "linear":
                print("Only linear interpolation supported for this controller type.")
            if self.interpolation == "linear":
                self.linear_interpolate(self.last_goal, self.goal)
            else:
                self.last_goal = np.array((self.goal))

        if self.interpolation == "linear":
            self.last_goal = self.linear[self.step]

            if self.step < self.interpolation_steps - 1:
                self.step += 1

        # decoupling with mass matrix
        if self.inertia_decoupling:
            torques = self.mass_matrix.dot(self.last_goal)
        else:
            torques = np.array(self.last_goal)

        return torques

    def update_model(self, sim, joint_index, id_name='right_hand'):

        super().update_model(sim, joint_index, id_name)

        if self.inertia_decoupling:
            self.update_mass_matrix(sim, joint_index)


class JointVelocityController(Controller):
    """
    Class to interprete actions as joint velocities
    """

    def __init__(self,
                 control_range,
                 kv,
                 max_action=1,
                 min_action=-1,
                 interpolation=None,
                 ):
        super(JointVelocityController, self).__init__(
            control_max=np.array(control_range),
            control_min=-1 * np.array(control_range),
            max_action=max_action,
            min_action=min_action,
            interpolation=interpolation)

        self.kv = np.array(kv)
        self.interpolate = True

        self.last_goal = np.zeros(self.control_dim)
        self.step = 0

    def reset(self):
        super().reset()
        self.step = 0
        self.last_goal = np.zeros(self.control_dim)

    def action_to_torques(self, action, policy_step):
        action = self.transform_action(action)
        if policy_step:
            self.step = 0
            self.goal = np.array((action))
            if self.interpolation and self.interpolation != "linear":
                print("Only linear interpolation supported for this controller type.")
            if self.interpolation == "linear":
                self.linear_interpolate(self.last_goal, self.goal)
            else:
                self.last_goal = np.array((self.goal))

        if self.interpolation == "linear":
            self.last_goal = self.linear[self.step]

            if self.step < self.interpolation_steps - 1:
                self.step += 1

        # Torques for each joint are kv*(q_dot_desired - q_dot)
        torques = np.multiply(self.kv, (self.last_goal - self.current_joint_velocity))

        return torques


class JointImpedanceController(Controller):
    """
    Class to interpret actions as joint impedance values
    """

    def __init__(self,
                 control_range,
                 control_freq,
                 kp_max,
                 kp_min,
                 damping_max,
                 damping_min,
                 impedance_flag=False,
                 max_action=1,
                 min_action=-1,
                 interpolation=None,
                 **kwargs
                 ):
        # for back-compatibility interpret a single # as the same value for all joints
        if type(kp_max) != list: kp_max = [kp_max] * len(control_range)
        if type(kp_min) != list: kp_min = [kp_min] * len(control_range)
        if type(damping_max) != list: damping_max = [damping_max] * len(control_range)
        if type(damping_min) != list: damping_min = [damping_min] * len(control_range)
        super(JointImpedanceController, self).__init__(
            control_max=np.array(control_range),
            control_min=-1 * np.array(control_range),
            max_action=max_action,
            min_action=min_action,
            control_freq=control_freq,
            impedance_flag=impedance_flag,
            kp_max=np.array(kp_max),
            kp_min=np.array(kp_min),
            damping_max=np.array(damping_max),
            damping_min=np.array(damping_min),
            interpolation=interpolation,
            **kwargs
        )

        self.interpolate = True
        self.use_delta_impedance = False
        self.impedance_kp = (np.array(kp_max) + np.array(kp_min)) * 0.5
        self.impedance_damping = (np.array(damping_max) + np.array(damping_min)) * 0.5
        self.last_goal_joint = np.zeros(self.control_dim)
        self.step = 0

    def reset(self):
        super().reset()
        self.step = 0
        self.last_goal_joint = np.zeros(self.control_dim)


    def interpolate_joint(self, starting_joint, last_goal_joint, goal_joint, current_vel):
        # We interpolate to reach the commanded desired position in self.ramp_ratio % of the time we have this goal
        if self.interpolation == "cubic":
            time = [0, self.interpolation_steps]
            position = np.vstack((starting_joint, goal_joint))
            self.spline_joint = CubicSpline(time, position, bc_type=((1, current_vel), (1, (0, 0, 0, 0, 0, 0, 0))),
                                            axis=0)
        elif self.interpolation == 'linear':
            delta_x_per_step = (goal_joint - last_goal_joint) / self.interpolation_steps
            self.linear_joint = np.array([(last_goal_joint + i * delta_x_per_step)
                                          for i in range(1, int(self.interpolation_steps) + 1)])
        elif self.interpolation == None:
            pass
        else:
            logger.error("[Controller] Invalid interpolation! Please specify 'cubic' or 'linear'.")
            exit(-1)

    def action_to_torques(self, action, policy_step):

        action = self.transform_action(action)
        if policy_step == True:
            self.step = 0
            self.goal_joint_position = self.current_joint_position + action[0:self.control_dim]

            if self.impedance_flag: self.set_goal_impedance(
                action)  # this takes into account whether or not it's delta impedance

            if self.interpolation:
                if np.linalg.norm(self.last_goal_joint) == 0:
                    self.last_goal_joint = self.current_joint_position
                self.interpolate_joint(self.current_joint_position, self.last_goal_joint, self.goal_joint_position,
                                       self.current_joint_velocity)

            if self.impedance_flag:
                if self.interpolation:
                    self.interpolate_impedance(self.impedance_kp, self.impedance_damping, self.goal_kp, self.goal_damping)
                else:
                    # update impedances immediately
                    self.impedance_kp[self.action_mask] = self.goal_kp
                    self.impedance_damping[self.action_mask] = self.goal_damping

        # if interpolation is specified, then interpolate. Otherwise, pass
        if self.interpolation:
            if self.interpolation == 'cubic':
                self.last_goal_joint = self.spline_joint(self.step)
            elif self.interpolation == 'linear':
                self.last_goal_joint = self.linear_joint[self.step]
            else:
                logger.error("[Controller] Invalid interpolation! Please specify 'cubic' or 'linear'.")
                exit(-1)

            if self.step < self.interpolation_steps - 1:
                self.step += 1
            if self.impedance_flag: self.update_impedance(
                self.step)

        else:
            self.last_goal_joint = np.array(self.goal_joint_position)

            if self.impedance_flag:
                self.impedance_kp = action[self.kp_index[0]:self.kp_index[1]]
                self.impedance_damping = action[self.damping_index[0]:self.damping_index[1]]

        position_joint_error = self.last_goal_joint - self.current_joint_position

        self.impedance_kv = 2 * np.sqrt(self.impedance_kp) * self.impedance_damping

        norm = np.linalg.norm(self.current_joint_velocity)
        if norm > 7.0:
            self.current_joint_velocity /= (norm * 7.0)

        torques = np.multiply(self.impedance_kp, position_joint_error) - np.multiply(self.impedance_kv,
                                                                                     self.current_joint_velocity)

        decoupled_torques = np.dot(self.mass_matrix, torques)

        return decoupled_torques

    def update_model(self, sim, joint_index, id_name='right_hand'):
        super().update_model(sim, joint_index, id_name)
        self.update_mass_matrix(sim, joint_index)

    @property
    def action_mask(self):
        return np.arange(self.control_dim)


class PositionOrientationController(Controller):
    """
    Class to interpret actions as cartesian desired position and orientation (and impedance values)
    """

    def __init__(self,
                 control_range_pos,
                 control_range_ori,
                 kp_max,
                 kp_max_abs_delta,
                 kp_min,
                 damping_max,
                 damping_max_abs_delta,
                 damping_min,
                 use_delta_impedance,
                 initial_impedance_pos,
                 initial_impedance_ori,
                 initial_damping,
                 initial_joint=None,
                 control_freq=20,
                 max_action=1,
                 min_action=-1,
                 impedance_flag=False,
                 position_limits=[[0, 0, 0], [0, 0, 0]],
                 orientation_limits=[[0, 0, 0], [0, 0, 0]],
                 interpolation=None,
                 force_control=False,
                 axis_angle=False,
                 **kwargs
                 ):
        control_max = np.ones(3) * control_range_pos
        if control_range_ori is not None:
            control_max = np.concatenate([control_max, np.ones(3) * control_range_ori])
        control_min = -1 * control_max
        kp_max = (np.ones(6) * kp_max)[self.action_mask]
        kp_max_abs_delta = (np.ones(6) * kp_max_abs_delta)[self.action_mask]
        kp_min = (np.ones(6) * kp_min)[self.action_mask]
        damping_max = (np.ones(6) * damping_max)[self.action_mask]
        damping_max_abs_delta = (np.ones(6) * damping_max_abs_delta)[self.action_mask]
        damping_min = (np.ones(6) * damping_min)[self.action_mask]
        initial_impedance = np.concatenate([np.ones(3) * initial_impedance_pos, np.ones(3) * initial_impedance_ori])
        initial_damping = np.ones(6) * initial_damping

        self.use_delta_impedance = use_delta_impedance
        self.force_control = force_control
        self.axis_angle = axis_angle

        if self.use_delta_impedance:
            # provide range of possible delta impedances
            kp_param_max = kp_max_abs_delta
            kp_param_min = -kp_max_abs_delta
            damping_param_max = damping_max_abs_delta
            damping_param_min = -damping_max_abs_delta

            # store actual ranges for manual clipping
            self.kp_max = kp_max
            self.kp_min = kp_min
            self.damping_max = damping_max
            self.damping_min = damping_min
        else:
            # just use ranges directly
            kp_param_max = kp_max
            kp_param_min = kp_min
            damping_param_max = damping_max
            damping_param_min = damping_min

        super(PositionOrientationController, self).__init__(
            control_max=control_max,
            control_min=control_min,
            max_action=max_action,
            min_action=min_action,
            impedance_flag=impedance_flag,
            kp_max=kp_param_max,
            kp_min=kp_param_min,
            damping_max=damping_param_max,
            damping_min=damping_param_min,
            initial_joint=initial_joint,
            control_freq=control_freq,
            position_limits=position_limits,
            orientation_limits=orientation_limits,
            interpolation=interpolation,
            **kwargs
        )

        self.impedance_kp = np.array(initial_impedance).astype('float64')
        self.impedance_damping = np.array(initial_damping).astype('float64')

        self.step = 0
        self.interpolate = True

        self.last_goal_position = np.array((0, 0, 0))
        self.last_goal_orientation = np.eye(3)

    def reset(self):
        super().reset()
        self.step = 0
        self.last_goal_position = np.array((0, 0, 0))
        self.last_goal_orientation = np.eye(3)

    def interpolate_position(self, starting_position, last_goal_position, goal_position, current_vel):

        if self.interpolation == "cubic":
            # We interpolate to reach the commanded desired position in self.ramp_ratio % of the time we have this goal
            time = [0, self.interpolation_steps]
            position = np.vstack((starting_position, goal_position))
            self.spline_pos = CubicSpline(time, position, bc_type=((1, current_vel), (1, (0, 0, 0))), axis=0)
        elif self.interpolation == 'linear':
            delta_x_per_step = (goal_position - last_goal_position) / self.interpolation_steps
            self.linear_pos = np.array(
                [(last_goal_position + i * delta_x_per_step) for i in range(1, int(self.interpolation_steps) + 1)])
        elif self.interpolation == None:
            pass
        else:
            logger.error("[Controller] Invalid interpolation! Please specify 'cubic' or 'linear'.")
            exit(-1)

    def interpolate_orientation(self, starting_orientation, last_goal_orientation, goal_orientation, current_vel):
        # We interpolate to reach the commanded desired position in self.ramp_ratio % of the time we have this goal
        if self.interpolation == "cubic":
            time = [0, self.interpolation_steps]
            orientation_error = self.calculate_orientation_error(desired=goal_orientation, current=starting_orientation)
            orientation = np.vstack(([0, 0, 0], orientation_error))
            self.spline_ori = CubicSpline(time, orientation, bc_type=((1, current_vel), (1, (0, 0, 0))), axis=0)
            self.orientation_initial_interpolation = starting_orientation
        elif self.interpolation == 'linear':
            orientation_error = self.calculate_orientation_error(desired=goal_orientation,
                                                                 current=last_goal_orientation)
            delta_r_per_step = orientation_error / self.interpolation_steps
            self.linear_ori = np.array([i * delta_r_per_step for i in range(1, int(self.interpolation_steps) + 1)])
            self.orientation_initial_interpolation = last_goal_orientation
        elif self.interpolation == None:
            pass
        else:
            logger.error("[Controller] Invalid interpolation! Please specify 'cubic' or 'linear'.")
            exit(-1)

    def modify_action_with_force_perturbation(self, action, force):
        """
        Modifies the passed action vector with a pose perturbation that
        roughly corresponds to external force perturbations on the arm
        with the passed force vector.
        """

        # copies
        action = np.array(action)
        force = np.array(force)

        # get raw action from normalized actions
        action = self.transform_action(action)

        if len(self.action_mask) > 3:
            assert force.shape[0] == 6

            # this perturbation is computed assuming that the action space is
            # axis-angle - (exponential coordinates)
            force[3:6] *= 10.
            kp = np.array(self.impedance_kp[3:6])
            mr_inv = scipy.linalg.inv(self.lambda_r_matrix)
            rot_perturb = mr_inv.dot(force[3:6]) / kp

            if self.axis_angle:
                action[3:6] += rot_perturb
            else:
                # convert from euler to aa, add the force perturbation, then convert back
                r_exp = T.axisangle2vec(*T.quat2axisangle(T.mat2quat(T.euler2mat(action[3:6]))))
                r_exp += rot_perturb
                action[3:6] = T.mat2euler(T.quat2mat(T.axisangle2quat(*T.vec2axisangle(r_exp))))
        else:
            assert force.shape[0] == 3

        force[:3] *= 25.

        # delta x' = delta x + 1/kp * M^-1 * F
        kp = np.array(self.impedance_kp[0:3])
        mx_inv = scipy.linalg.inv(self.lambda_x_matrix)
        pos_perturb = mx_inv.dot(force[:3]) / kp
        action[:3] += pos_perturb
        
        # re-normalize action
        return self.untransform_action(action)

    def action_to_torques(self, action, policy_step):
        """
        Given the next action, output joint torques for the robot.
        Assumes the robot's model is updated.
        """

        force_action = None
        if self.force_control:
            if len(self.action_mask) > 3:
                # control rotation too
                force_action = np.concatenate([
                    action[:3] * 25., 
                    action[3:6] * 10.,
                ])
                action = action[6:]
            else:
                # translation only
                force_action = np.array(action[:3]) * 25.
                action = action[3:]

        action = self.transform_action(action)

        # This is computed only when we receive a new desired goal from policy
        if policy_step == True:
            self.step = 0
            self.set_goal_position(action)
            self.set_goal_orientation(action)
            if self.impedance_flag: self.set_goal_impedance(
                action)  # this takes into account whether or not it's delta impedance

            if self.interpolation:
                # The first time we interpolate we don't have a previous goal value -> We set it to the current robot position+orientation
                if np.linalg.norm(self.last_goal_position) == 0:
                    self.last_goal_position = self.current_position
                if (self.last_goal_orientation == np.eye(self.last_goal_orientation.shape[0])).all():
                    self.last_goal_orientation = self.current_orientation_mat
                # set goals for next round of interpolation - TODO rename these functions?
                self.interpolate_position(self.current_position, self.last_goal_position, self.goal_position,
                                          self.current_lin_velocity)
                self.interpolate_orientation(self.current_orientation_mat, self.last_goal_orientation,
                                             self.goal_orientation, self.current_ang_velocity)

            # handle impedances
            if self.impedance_flag:
                if self.interpolation:
                    # set goals for next round of interpolation
                    self.interpolate_impedance(self.impedance_kp, self.impedance_damping, self.goal_kp, self.goal_damping)
                else:
                    # update impedances immediately
                    self.impedance_kp[self.action_mask] = self.goal_kp
                    self.impedance_damping[self.action_mask] = self.goal_damping

        if self.interpolation:
            if self.interpolation == 'cubic':
                self.last_goal_position = self.spline_pos(self.step)
                goal_orientation_delta = self.spline_ori(self.step)
            elif self.interpolation == 'linear':
                self.last_goal_position = self.linear_pos[self.step]
                goal_orientation_delta = self.linear_ori[self.step]
            else:
                logger.error("[Controller] Invalid interpolation! Please specify 'cubic' or 'linear'.")
                exit(-1)

            if self.impedance_flag: self.update_impedance(self.step)

            self.last_goal_orientation = np.dot((T.euler2mat(-goal_orientation_delta).T),
                                                self.orientation_initial_interpolation)

            # After self.ramp_ratio % of the time we have reached the desired pose and stay constant
            if self.step < self.interpolation_steps - 1:
                self.step += 1
        else:
            self.last_goal_position = np.array((self.goal_position))
            self.last_goal_orientation = self.goal_orientation

            if self.impedance_flag:
                self.impedance_kp = action[self.kp_index[0]:self.kp_index[1]]
                self.impedance_damping = action[self.damping_index[0]:self.damping_index[1]]

        position_error = self.last_goal_position - self.current_position
        orientation_error = self.calculate_orientation_error(desired=self.last_goal_orientation,
                                                             current=self.current_orientation_mat)

        # always ensure critical damping TODO - technically this is called unneccessarily if the impedance_flag is not set
        self.impedance_kv = 2 * np.sqrt(self.impedance_kp) * self.impedance_damping

        return self.calculate_impedance_torques(position_error, orientation_error, force_action=force_action)

    def calculate_impedance_torques(self, position_error, orientation_error, force_action=None):
        """
        Given the current errors in position and orientation, return the desired torques per joint
        """
        desired_force = (np.multiply(np.array(position_error), np.array(self.impedance_kp[0:3]))
                         - np.multiply(np.array(self.current_lin_velocity), self.impedance_kv[0:3]))

        desired_torque = (np.multiply(np.array(orientation_error), np.array(self.impedance_kp[3:6]))
                          - np.multiply(np.array(self.current_ang_velocity), self.impedance_kv[3:6]))

        uncoupling = True
        if (uncoupling):
            decoupled_force = np.dot(self.lambda_x_matrix, desired_force)
            decoupled_torque = np.dot(self.lambda_r_matrix, desired_torque)
            if force_action is not None:
                assert self.force_control
                decoupled_force += force_action[:3]
                if len(force_action) > 3:
                    decoupled_torque += force_action[3:6]
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(self.lambda_matrix, desired_wrench)

        torques = np.dot(self.J_full.T, decoupled_wrench)

        if self.initial_joint is not None:
            # TODO where does 10 come from?
            joint_kp = 10
            joint_kv = np.sqrt(joint_kp) * 2
            pose_torques = np.dot(self.mass_matrix, (joint_kp * (
                        self.initial_joint - self.current_joint_position) - joint_kv * self.current_joint_velocity))
            nullspace_torques = np.dot(self.nullspace_matrix.transpose(), pose_torques)
            torques += nullspace_torques
            self.torques = torques

        return torques

    def update_model(self, sim, joint_index, id_name='right_hand'):

        super().update_model(sim, joint_index, id_name)

        self.update_mass_matrix(sim, joint_index)
        self.update_model_opspace(joint_index)

    def update_model_opspace(self, joint_index):
        """
        Updates the following:
        -Lambda matrix (full, linear, and rotational)
        -Nullspace matrix

        joint_index - list of joint position indices in Mujoco
        """
        pos_joint_index, vel_joint_index = joint_index
        
        mass_matrix_inv = scipy.linalg.inv(self.mass_matrix)

        # J M^-1 J^T
        lambda_matrix_inv = np.dot(
            np.dot(self.J_full, mass_matrix_inv),
            self.J_full.transpose()
        )

        # (J M^-1 J^T)^-1
        self.lambda_matrix = scipy.linalg.inv(lambda_matrix_inv)

        # Jx M^-1 Jx^T
        lambda_x_matrix_inv = np.dot(
            np.dot(self.Jx, mass_matrix_inv),
            self.Jx.transpose()
        )

        # Jr M^-1 Jr^T
        lambda_r_matrix_inv = np.dot(
            np.dot(self.Jr, mass_matrix_inv),
            self.Jr.transpose()
        )

        # take the inverse, but zero out elements in cases of a singularity
        svd_u, svd_s, svd_v = np.linalg.svd(lambda_x_matrix_inv)
        singularity_threshold = 0.00025
        svd_s_inv = [0 if x < singularity_threshold else 1. / x for x in svd_s]
        self.lambda_x_matrix = svd_v.T.dot(np.diag(svd_s_inv)).dot(svd_u.T)

        svd_u, svd_s, svd_v = np.linalg.svd(lambda_r_matrix_inv)
        singularity_threshold = 0.00025
        svd_s_inv = [0 if x < singularity_threshold else 1. / x for x in svd_s]
        self.lambda_r_matrix = svd_v.T.dot(np.diag(svd_s_inv)).dot(svd_u.T)

        if self.initial_joint is not None:
            Jbar = np.dot(mass_matrix_inv, self.J_full.transpose()).dot(self.lambda_matrix)
            self.nullspace_matrix = np.eye(len(vel_joint_index), len(vel_joint_index)) - np.dot(Jbar, self.J_full)

    def set_goal_position(self, action, position=None):
        if position is not None:
            self._goal_position = position
        else:
            self._goal_position = self.current_position + action[0:3]
            if np.array(self.position_limits).any():
                for idx in range(3):
                    self._goal_position[idx] = np.clip(self._goal_position[idx], self.position_limits[0][idx],
                                                       self.position_limits[1][idx])

    def set_goal_orientation(self, action, orientation=None):
        if orientation is not None:
            self._goal_orientation = orientation
        else:
            if self.axis_angle:
                # interpret input as scaled axis-angle (exponential coordinates)
                axis, angle = T.vec2axisangle(-action[3:6])
                quat_error = T.axisangle2quat(axis=axis, angle=angle)
                rotation_mat_error = T.quat2mat(quat_error)
            else:
                # interpret input as delta euler
                rotation_mat_error = T.euler2mat(-action[3:6])
            self._goal_orientation = np.dot(rotation_mat_error.T, self.current_orientation_mat)

            if np.array(self.orientation_limits).any():
                # TODO: Limit rotation!
                euler = T.mat2euler(self._goal_orientation)

                limited = False
                for idx in range(3):
                    if self.orientation_limits[0][idx] < self.orientation_limits[1][idx]:  # Normal angle sector meaning
                        if euler[idx] > self.orientation_limits[0][idx] and euler[idx] < self.orientation_limits[1][
                            idx]:
                            continue
                        else:
                            limited = True
                            dist_to_lower = euler[idx] - self.orientation_limits[0][idx]
                            if dist_to_lower > np.pi:
                                dist_to_lower -= 2 * np.pi
                            elif dist_to_lower < -np.pi:
                                dist_to_lower += 2 * np.pi

                            dist_to_higher = euler[idx] - self.orientation_limits[1][idx]
                            if dist_to_lower > np.pi:
                                dist_to_higher -= 2 * np.pi
                            elif dist_to_lower < -np.pi:
                                dist_to_higher += 2 * np.pi

                            if dist_to_lower < dist_to_higher:
                                euler[idx] = self.orientation_limits[0][idx]
                            else:
                                euler[idx] = self.orientation_limits[1][idx]
                    else:  # Inverted angle sector meaning
                        if euler[idx] > self.orientation_limits[0][idx] or euler[idx] < self.orientation_limits[1][idx]:
                            continue
                        else:
                            limited = True
                            dist_to_lower = euler[idx] - self.orientation_limits[0][idx]
                            if dist_to_lower > np.pi:
                                dist_to_lower -= 2 * np.pi
                            elif dist_to_lower < -np.pi:
                                dist_to_lower += 2 * np.pi

                            dist_to_higher = euler[idx] - self.orientation_limits[1][idx]
                            if dist_to_lower > np.pi:
                                dist_to_higher -= 2 * np.pi
                            elif dist_to_lower < -np.pi:
                                dist_to_higher += 2 * np.pi

                            if dist_to_lower < dist_to_higher:
                                euler[idx] = self.orientation_limits[0][idx]
                            else:
                                euler[idx] = self.orientation_limits[1][idx]
                if limited:
                    self._goal_orientation = T.euler2mat(np.array([euler[1], euler[0], euler[2]]))

    @property
    def action_mask(self):
        # TODO - why can't this be control_dim like the others?
        return np.array((0, 1, 2, 3, 4, 5))

    # return np.arange(self.control_dim)

    @property
    def goal_orientation(self):
        return self._goal_orientation

    @property
    def goal_position(self):
        return self._goal_position


class PositionController(PositionOrientationController):
    """
    Class to interprete actions as cartesian desired position ONLY (and impedance values)
    """

    def __init__(self,
                 control_range_pos,
                 kp_max,
                 kp_max_abs_delta,
                 kp_min,
                 damping_max,
                 damping_max_abs_delta,
                 damping_min,
                 use_delta_impedance,
                 initial_impedance_pos,
                 initial_impedance_ori,
                 initial_damping,
                 max_action=1.0,
                 min_action=-1.0,
                 impedance_flag=False,
                 initial_joint=None,
                 control_freq=20,
                 interpolation=None,
                 force_control=False,
                 axis_angle=False,
                 **kwargs
                 ):
        super(PositionController, self).__init__(
            control_range_pos=control_range_pos,
            control_range_ori=None,
            max_action=max_action,
            min_action=min_action,
            impedance_flag=impedance_flag,
            kp_max=kp_max,
            kp_max_abs_delta=kp_max_abs_delta,
            kp_min=kp_min,
            damping_max=damping_max,
            damping_max_abs_delta=damping_max_abs_delta,
            damping_min=damping_min,
            initial_joint=initial_joint,
            control_freq=control_freq,
            use_delta_impedance=use_delta_impedance,
            initial_impedance_pos=initial_impedance_pos,
            initial_impedance_ori=initial_impedance_ori,
            initial_damping=initial_damping,
            interpolation=interpolation,
            force_control=force_control,
            axis_angle=axis_angle,
            **kwargs)

        self.goal_orientation_set = False

    def reset(self):
        super().reset()

    def set_goal_orientation(self, action, orientation=None):
        if orientation is not None:
            self._goal_orientation = orientation
        elif self.goal_orientation_set == False:
            self._goal_orientation = np.array(self.current_orientation_mat)
            self.goal_orientation_set = True

    @property
    def goal_orientation(self):
        return self._goal_orientation

    @property
    def action_mask(self):
        return np.array((0, 1, 2))

    @property
    def goal_position(self):
        return self._goal_position
