import numpy as np

from robosuite.controllers.controller import Controller


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

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        torque_limits (2-list of float or 2-list of list of floats): Limits (N-m) below and above which the magnitude
            of a calculated goal joint torque will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)
            If not specified, will automatically set the limits to the actuator limits for this robot arm

        interpolator (Interpolator): Interpolator object to be used for interpolating from the current joint torques to
            the goal joint torques during each timestep between inputted actions

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    """

    def __init__(
        self,
        sim,
        joint_indexes,
        actuator_range,
        ref_name=None,
        input_max=1,
        input_min=-1,
        output_max=0.05,
        output_min=-0.05,
        policy_freq=20,
        lite_physics=False,
        torque_limits=None,
        interpolator=None,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):

        super().__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),
            lite_physics=lite_physics,
        )

        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)

        # limits (if not specified, set them to actuator limits by default)
        self.torque_limits = np.array(torque_limits) if torque_limits is not None else self.actuator_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize torques
        self.goal_torque = None  # Goal torque desired, pre-compensation
        self.current_torque = np.zeros(self.control_dim)  # Current torques being outputted, pre-compensation
        self.torques = None  # Torques returned every time run_controller is called

    def set_goal(self, torques):
        """
        Sets goal based on input @torques.

        Args:
            torques (Iterable): Desired joint torques

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        # Update state
        self.update()

        # Check to make sure torques is size self.joint_dim
        assert len(torques) == self.control_dim, "Delta torque must be equal to the robot's joint dimension space!"

        self.goal_torque = np.clip(self.scale_action(torques), self.torque_limits[0], self.torque_limits[1])

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """
        # Make sure goal has been set
        if self.goal_torque is None:
            self.set_goal(np.zeros(self.control_dim))

        # Update state
        self.update()

        # Only linear interpolator is currently supported
        if self.interpolator is not None:
            # Linear case
            if self.interpolator.order == 1:
                self.current_torque = self.interpolator.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            self.current_torque = np.array(self.goal_torque)

        # Add gravity compensation
        self.torques = self.current_torque + self.torque_compensation

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        # Return final torques
        return self.torques

    def reset_goal(self):
        """
        Resets joint torque goal to be all zeros (pre-compensation)
        """
        self.goal_torque = np.zeros(self.control_dim)

        # Reset interpolator if required
        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    @property
    def name(self):
        return "JOINT_TORQUE"
