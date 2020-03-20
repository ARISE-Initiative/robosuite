from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np


class EEImpController(Controller):
    """
    Controller for EE impedance
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 input_max=1,
                 input_min=-1,
                 output_max=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),
                 output_min=(-0.05, -0.05, -0.05, -0.5, -0.5, -0.5),
                 kp=50,
                 damping=1,
                 policy_freq=20,
                 position_limits=None,
                 orientation_limits=None,
                 interpolator_pos=None,
                 interpolator_ori=None,
                 control_ori=True,
                 control_delta=True,
                 uncouple_pos_ori=True,
                 **kwargs # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
        )
        # Determine whether this is pos ori or just pos
        self.use_ori = control_ori

        # Determine whether we want to use delta or absolute values as inputs
        self.use_delta = control_delta

        # Control dimension
        self.control_dim = 6 if self.use_ori else 3

        # input and output max and min
        self.input_max = input_max
        self.input_min = input_min
        self.output_max = output_max[:self.control_dim] if type(output_max) == tuple else output_max
        self.output_min = output_min[:self.control_dim] if type(output_min) == tuple else output_min

        # limits
        self.position_limits = position_limits
        self.orientation_limits = orientation_limits

        # kp kv
        self.kp = np.ones(6) * kp
        self.kv = np.ones(6) * 2 * np.sqrt(self.kp) * damping

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori
        # todo: orientation interpolators change to relative! refactor! (michelle)

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize
        self.goal_ori = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
        self.goal_pos = np.array([0,0,0])

        self.relative_ori = np.zeros(3)
        self.ori_ref = None

    def set_goal(self, delta, set_pos=None, set_ori=None):
        self.update()

        # If we're using deltas, interpret actions as such
        if self.use_delta:
            if delta is not None:
                scaled_delta = self.scale_action(delta)
                if not self.use_ori:
                    # Set default control for ori since user isn't actively controlling ori
                    set_ori = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            else:
                scaled_delta = []
        # Else, interpret actions as absolute values
        else:
            set_pos = self.initial_ee_pos + delta[:3]
            # Set default control for ori if we're only using position control
            set_ori = self.initial_ee_ori_mat.T.dot(T.euler2mat(delta[3:])) if self.use_ori \
                else np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            scaled_delta = []

        # We only want to update goal orientation if there is a valid delta ori value
        if not np.isclose(scaled_delta[3:], 0).all():
            self.goal_ori = set_goal_orientation(scaled_delta[3:],
                                                 self.ee_ori_mat,
                                                 orientation_limit=self.orientation_limits,
                                                 set_ori=set_ori)
        self.goal_pos = set_goal_position(scaled_delta[:3],
                                          self.ee_pos,
                                          position_limit=self.position_limits,
                                          set_pos=set_pos)

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)
            print("Got here!")

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat)  # reference is the current orientation at start
            self.interpolator_ori.set_goal(orientation_error(self.goal_ori, self.ori_ref))  # goal is the total orientation error
            self.relative_ori = np.zeros(3)  # relative orientation always starts at 0

    def run_controller(self, action=None):
        """
        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        Note: Does not account for gravity compensation!

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        """
        # First, update goal if action is not set to none
        # Action will be interpreted as delta value from current
        if action is not None:
            self.set_goal(action)
        else:
            self.update()

        desired_pos = None
        # Only linear interpolator is currently supported
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal(self.ee_pos)
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_pos = np.array(self.goal_pos)

        if self.interpolator_ori is not None:
            # relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)

            interpolated_results = self.interpolator_ori.get_interpolated_goal(self.relative_ori)
            ori_error = interpolated_results[0:3]
        else:
            desired_ori = np.array(self.goal_ori)
            ori_error = orientation_error(desired_ori, self.ee_ori_mat)

        # Compute desired force and torque based on errors
        position_error = desired_pos - self.ee_pos
        vel_pos_error = -self.ee_pos_vel

        # F_r = kp * pos_err + kv * vel_err
        desired_force = (np.multiply(np.array(position_error), np.array(self.kp[0:3]))
                         + np.multiply(vel_pos_error, self.kv[0:3]))

        vel_ori_error = -self.ee_ori_vel

        # Tau_r = kp * ori_err + kv * vel_err
        desired_torque = (np.multiply(np.array(ori_error), np.array(self.kp[3:6]))
                          + np.multiply(vel_ori_error, self.kv[3:6]))

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(self.mass_matrix,
                                                                                 self.J_full,
                                                                                 self.J_pos,
                                                                                 self.J_ori)

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F + compensations
        self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation

        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_0 = desired nullspace pose torques, assumed to be positional joint control relative
        #                   to the initial joint positions
        self.torques += nullspace_torques(self.mass_matrix, nullspace_matrix,
                                          self.initial_joint, self.joint_pos, self.joint_vel)

        return self.torques

    @property
    def name(self):
        return 'ee_imp'
