from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np

class EEImpController(Controller):
    """

    """

    def __init__(self,
                 sim,
                 robot_id,
                 joint_indexes,
                 input_max=1,
                 input_min=-1,
                 output_max=0.05,
                 output_min=-0.05,
                 kp=50,
                 damping=1,
                 policy_freq=20,
                 initial_joint=None,
                 position_limits=None,
                 orientation_limits=None,
                 interpolator_pos=None,
                 interpolator_ori=None,
                 uncouple_pos_ori=True,
                 **kwargs # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super(EEImpController, self).__init__(
            sim,
            robot_id,
            joint_indexes,
            initial_joint
        )
        # Control dimension
        self.control_dim = 6 if orientation_limits else 3

        # input and output max and min
        self.input_max = input_max
        self.input_min = input_min
        self.output_max = output_max
        self.output_min = output_min

        # limits
        self.position_limits = np.array(position_limits)
        self.orientation_limits = np.array(orientation_limits)

        # kp kv
        self.kp = np.ones(6) * kp
        self.kv = np.ones(6) * 2 * np.sqrt(self.kp) * damping

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori
        # self.interpolator_ori = None
        # todo: orientation interpolators change to relative! refactor!

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori

        # initialize
        self.goal_ori = None
        self.goal_pos = None

        self.relative_ori = np.zeros(3)

        self.set_goal(np.zeros(6))

    def set_goal(self, delta, set_pos=None, set_ori=None):
        self.update()
        if delta is not None:
            scaled_delta = self.scale_action(delta)
        else:
            scaled_delta = None

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

        if self.interpolator_ori is not None:
            self.ori_ref = np.array(self.ee_ori_mat) #reference is the current orientation at start
            self.interpolator_ori.set_goal(orientation_error(self.goal_ori, self.ori_ref)) #goal is the total orientation error
            self.relative_ori = np.zeros(3) #relative orientation always starts at 0

    def run_controller(self):
        # TODO: check if goal has been set.
        self.update()

        desired_vel_pos = 0
        desired_acc_pos = 0
        desired_vel_ori = 0
        desired_acc_ori = 0

        if self.interpolator_pos is not None:
            if self.interpolator_pos.order == 4:
                interpolated_results = self.interpolator_pos.get_interpolated_goal(self.ee_pos)
                desired_pos = interpolated_results[0:3]
                desired_vel_pos = interpolated_results[3:6]
                desired_acc_pos = interpolated_results[6:]
            else:
                desired_pos = self.interpolator_pos.get_interpolated_goal(self.ee_pos)
        else:
            desired_pos = np.array(self.goal_pos)

        if self.interpolator_ori is not None:
            #relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(self.ee_ori_mat, self.ori_ref)

            interpolated_results = self.interpolator_ori.get_interpolated_goal(self.relative_ori)
            ori_error = interpolated_results[0:3]

            if self.interpolator_ori.order == 4:
                desired_vel_ori = interpolated_results[3:6]
                desired_acc_ori = interpolated_results[6:]
                self.ori_interpolate_started = True

        else:
            desired_ori = np.array(self.goal_ori)
            ori_error = orientation_error(desired_ori, self.ee_ori_mat)

        position_error = desired_pos - self.ee_pos
        vel_pos_error = desired_vel_pos - self.ee_pos_vel
        desired_force = (np.multiply(np.array(position_error), np.array(self.kp[0:3]))
                         + np.multiply(vel_pos_error, self.kv[0:3])) + desired_acc_pos

        vel_ori_error = desired_vel_ori - self.ee_ori_vel
        desired_torque = (np.multiply(np.array(ori_error), np.array(self.kp[3:6]))
                          + np.multiply(vel_ori_error, self.kv[3:6])) + desired_acc_ori

        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(self.mass_matrix,
                                                                                 self.J_full,
                                                                                 self.J_pos,
                                                                                 self.J_ori)
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate([decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        self.torques = np.dot(self.J_full.T, decoupled_wrench) + self.torque_compensation


        # Calculate nullspace torques if necessary
        # TODO: Cleanup / verify with michelle
        if self.initial_joint:
            self.torques += nullspace_torques(self.mass_matrix, nullspace_matrix,
                                              self.initial_joint, self.joint_pos, self.joint_vel)

        return self.torques
