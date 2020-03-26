from robosuite.controllers.base_controller import Controller
import numpy as np


class JointTorqueController(Controller):
    """
    Controller for controlling the robot arm's joint torques. As the actuators at the mujoco sim level are already
    torque actuators, this "controller" usually simply "passes through" desired torques, though it also includes the
    typical input / output scaling and clipping, as well as interpolator features seen in other controllers classes
    as well

    NOTE: Control input actions assumed to be taken as absolute joint torques. A given action to this
    controller is assumed to be of the form: (torq_j0, torq_j1, ... , torq_jn-1) for an n-joint robot

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

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        torque_limits (2-list of float or 2-list of list of floats): Limits (N-m) below and above which the magnitude
            of a calculated goal joint torque will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint torques to
            the goal joint torques during each timestep between inputted actions

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
                 policy_freq=20,
                 torque_limits=None,
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
        self.torque_limits = torque_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize torques
        self.goal_torque = None                           # Goal torque desired, pre-compensation
        self.current_torque = np.zeros(self.control_dim)  # Current torques being outputted, pre-compensation
        self.torques = None                               # Torques returned every time run_controller is called

    def set_goal(self, torques):
        self.update()

        # Check to make sure torques is size self.joint_dim
        assert len(torques) == self.control_dim, "Delta torque must be equal to the robot's joint dimension space!"

        self.goal_torque = torques
        if self.torque_limits is not None:
            self.goal_torque = np.clip(self.goal_torque, self.torque_limits[0], self.torque_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    def run_controller(self, action=None):
        # Make sure goal has been set
        if not self.goal_torque.any():
            self.set_goal(np.zeros(self.control_dim))

        # Then, update goal if action is not set to none
        # Action will be interpreted as delta value from current
        if action is not None:
            self.set_goal(action)
        else:
            self.update()

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                self.current_torque = self.interpolator.get_interpolated_goal(self.current_torque)
            else:
                # Nonlinear case not currently supported
                pass
        else:
            self.current_torque = np.array(self.goal_torque)

        # Add gravity compensation
        self.torques = self.current_torque + self.torque_compensation

        # Return final torques
        return self.torques

    @property
    def name(self):
        return 'joint_tor'
