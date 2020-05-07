from robosuite.controllers.base_controller import Controller
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

        kv (float or list of float): velocity gain for determining desired torques based upon the joint vel errors.
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

    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
                 input_max=1,
                 input_min=-1,
                 output_max=1,
                 output_min=-1,
                 kv=4.0,
                 policy_freq=20,
                 velocity_limits=None,
                 interpolator=None,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            eef_name,
            joint_indexes,
        )
        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # kv
        self.kv = self.nums2array(kv, self.control_dim)

        # limits
        self.velocity_limits = velocity_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize torques and goal velocity
        self.goal_vel = None                            # Goal velocity desired, pre-compensation
        self.current_vel = np.zeros(self.joint_dim)     # Current velocity setpoint, pre-compensation
        self.torques = None                             # Torques returned every time run_controller is called

    def set_goal(self, velocities):
        # Update state
        self.update()

        # Otherwise, check to make sure velocities is size self.joint_dim
        assert len(velocities) == self.joint_dim, \
            "Goal action must be equal to the robot's joint dimension space! Expected {}, got {}".format(
                self.joint_dim, len(velocities)
            )

        self.goal_vel = velocities
        if self.velocity_limits is not None:
            self.goal_vel = np.clip(velocities, self.velocity_limits[0], self.velocity_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_vel)

    def run_controller(self):
        # Make sure goal has been set
        if not self.goal_vel.any():
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            if self.interpolator.order == 1:
                # Linear case
                self.current_vel = self.interpolator.get_interpolated_goal(self.current_vel)
            else:
                # Nonlinear case not currently supported
                pass
        else:
            self.current_vel = np.array(self.goal_vel)

        # Compute torques plus gravity compensation
        self.torques = np.multiply(self.kv, (self.current_vel - self.joint_vel)) + self.torque_compensation

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Return final torques
        return self.torques

    def reset_goal(self):
        """
        Resets joint velocity goal to be all zeros
        """
        self.goal_vel = np.zeros(self.control_dim)

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_vel)

    @property
    def name(self):
        return 'JOINT_VELOCITY'
