from robosuite.controllers.base_controller import Controller
from robosuite.utils.buffers import RingBuffer
import numpy as np


class JointVelocityController(Controller):
    """
    Controller for controlling the robot arm's joint velocities. This is simply a P controller with desired torques
    (pre gravity compensation) taken to be proportional to the velocity error of the robot joints.

    NOTE: Control input actions assumed to be taken as absolute joint velocities. A given action to this
    controller is assumed to be of the form: (vel_j0, vel_j1, ... , vel_jn-1) for an n-joint robot

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

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

        kp (float or list of float): velocity gain for determining desired torques based upon the joint vel errors.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        velocity_limits (2-list of float or 2-list of list of floats): Limits (m/s) below and above which the magnitude
            of a calculated goal joint velocity will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint velocities
            to the goal joint velocities during each timestep between inputted actions

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
        input_max=1,
        input_min=-1,
        output_max=1,
        output_min=-1,
        kp=0.25,
        policy_freq=20,
        velocity_limits=None,
        interpolator=None,
        **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
            actuator_range,
        )
        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.joint_dim)
        self.input_min = self.nums2array(input_min, self.joint_dim)
        self.output_max = self.nums2array(output_max, self.joint_dim)
        self.output_min = self.nums2array(output_min, self.joint_dim)

        # gains and corresopnding vars
        self.kp = self.nums2array(kp, self.joint_dim)
        # if kp is a single value, map wrist gains accordingly (scale down x10 for final two joints)

        if type(kp) is float or type(kp) is int:
            # Scale kpp according to how wide the actuator range is for this robot
            low, high = self.actuator_limits
            self.kp = kp * (high - low)
        self.ki = self.kp * 0.005
        self.kd = self.kp * 0.001
        self.last_err = np.zeros(self.joint_dim)
        self.derr_buf = RingBuffer(dim=self.joint_dim, length=5)
        self.summed_err = np.zeros(self.joint_dim)
        self.saturated = False
        self.last_joint_vel = np.zeros(self.joint_dim)

        # limits
        self.velocity_limits = np.array(velocity_limits) if velocity_limits is not None else None

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize torques and goal velocity
        self.goal_vel = None  # Goal velocity desired, pre-compensation
        self.current_vel = np.zeros(self.joint_dim)  # Current velocity setpoint, pre-compensation
        self.torques = None  # Torques returned every time run_controller is called

    def set_goal(self, velocities):
        """
        Sets goal based on input @velocities.

        Args:
            velocities (Iterable): Desired joint velocities

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()

        # Otherwise, check to make sure velocities is size self.joint_dim
        assert (
            len(velocities) == self.joint_dim
        ), "Goal action must be equal to the robot's joint dimension space! Expected {}, got {}".format(
            self.joint_dim, len(velocities)
        )

        self.goal_vel = self.scale_action(velocities)
        if self.velocity_limits is not None:
            self.goal_vel = np.clip(self.goal_vel, self.velocity_limits[0], self.velocity_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_vel)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        if self.goal_vel is None:
            self.set_goal(np.zeros(self.joint_dim))

        # Update state
        self.update()

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            if self.interpolator.order == 1:
                # Linear case
                self.current_vel = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            self.current_vel = np.array(self.goal_vel)

        # We clip the current joint velocity to be within a reasonable range for stability
        joint_vel = np.clip(self.joint_vel, self.output_min, self.output_max)

        # Compute necessary error terms for PID velocity controller
        err = self.current_vel - joint_vel
        derr = err - self.last_err
        self.last_err = err
        self.derr_buf.push(derr)

        # Only add to I component if we're not saturated (anti-windup)
        if not self.saturated:
            self.summed_err += err

        # Compute command torques via PID velocity controller plus gravity compensation torques
        torques = (
            self.kp * err
            + self.ki * self.summed_err
            + self.kd * self.derr_buf.average
            + self.torque_compensation
        )

        # Clip torques
        self.torques = self.clip_torques(torques)

        # Check if we're saturated
        self.saturated = False if np.sum(np.abs(self.torques - torques)) == 0 else True

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Return final torques
        return self.torques

    def reset_goal(self):
        """
        Resets joint velocity goal to be all zeros
        """
        self.goal_vel = np.zeros(self.joint_dim)

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_vel)

    @property
    def name(self):
        return "JOINT_VELOCITY"
