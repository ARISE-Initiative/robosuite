from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import numpy as np


class JointPositionController(Controller):
    """
    Controller for controlling robot arm via impedance control. Allows position control of the robot's joints.

    NOTE: Control input actions assumed to be taken relative to the current joint positions. A given action to this
    controller is assumed to be of the form: (dpos_j0, dpos_j1, ... , dpos_jn-1) for an n-joint robot

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:
            "joints" : list of indexes to relevant robot joints
            "qpos" : list of indexes to relevant robot joint positions
            "qvel" : list of indexes to relevant robot joint velocities

        input_max (float or list of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or list of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or list of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or list of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or list of float): positional gain for determining desired torques based upon the joint pos errors.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping (float or list of float): used in conjunction with kp to determine the velocity gain for determining
            desired torques based upon the joint pos errors. Can be either be a scalar (same value for all action dims),
            or a list (specific values for each dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        qpos_limits (2-list of float or 2-list of list of floats): Limits (rad) below and above which the magnitude
            of a calculated goal joint position will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint position to
            the goal joint position during each timestep between inputted actions

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    """

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 input_max=1,
                 input_min=-1,
                 output_max=0.05,
                 output_min=-0.05,
                 kp=50,
                 damping=1,
                 policy_freq=20,
                 qpos_limits=None,
                 interpolator=None,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes
        )

        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min
        self.input_max = input_max
        self.input_min = input_min
        self.output_max = output_max
        self.output_min = output_min

        # limits
        self.position_limits = qpos_limits

        # kp kv
        self.kp = np.ones(self.joint_dim) * kp
        self.kv = np.ones(self.joint_dim) * 2 * np.sqrt(self.kp) * damping

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize
        self.goal_qpos = None

    def set_goal(self, delta, set_qpos=None):
        self.update()

        # Check to make sure delta is size self.joint_dim
        assert len(delta) == self.control_dim, "Delta qpos must be equal to the robot's joint dimension space!"

        if delta is not None:
            scaled_delta = self.scale_action(delta)
        else:
            scaled_delta = None

        self.goal_qpos = set_goal_position(scaled_delta,
                                           self.joint_pos,
                                           position_limit=self.position_limits,
                                           set_pos=set_qpos)

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_qpos)

    def run_controller(self):
        # Make sure goal has been set
        if not self.goal_qpos.any():
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        desired_qpos = None

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                desired_qpos = self.interpolator.get_interpolated_goal(self.joint_pos)
            else:
                # Nonlinear case not currently supported
                pass
        else:
            desired_qpos = np.array(self.goal_qpos)

        # torques = pos_err * kp + vel_err * kv
        position_error = desired_qpos - self.joint_pos
        vel_pos_error = -self.joint_vel
        desired_torque = (np.multiply(np.array(position_error), np.array(self.kp))
                          + np.multiply(vel_pos_error, self.kv))

        # Return desired torques plus gravity compensations
        self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation

        return self.torques

    @property
    def name(self):
        return 'JOINT_POSITION'
