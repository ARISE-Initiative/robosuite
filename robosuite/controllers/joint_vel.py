from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np


class JointVelController(Controller):
    """
    Controller for joint velocity
    """

    def __init__(self,
                 sim,
                 robot_id,
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
            robot_id,
            joint_indexes,
        )
        # Control dimension
        self.control_dim = len(joint_indexes["joints"])

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = input_max
        self.input_min = input_min
        self.output_max = output_max
        self.output_min = output_min

        # kv
        self.kv = np.array(kv) if type(kv) == list else np.array([kv] * self.control_dim)

        # limits
        self.velocity_limits = np.array(velocity_limits) if type(velocity_limits[0]) == list else \
            np.array([[velocity_limits[0]] * self.control_dim, [velocity_limits[1]] * self.control_dim])

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize torques and goal velocity
        self.goal_vel = None                            # Goal velocity desired, pre-compensation
        self.current_vel = np.zeros(self.joint_dim)     # Current velocity setpoint, pre-compensation
        self.torques = None                             # Torques returned every time run_controller is called

    def set_goal(self, delta, set_velocity=None):
        self.update()

        if delta is not None:
            # Check to make sure delta is size self.joint_dim
            assert len(delta) == self.joint_dim,\
                "Delta length must be equal to the robot's joint dimension space! Expected {}, got {}".format(
                    self.joint_dim, len(delta)
                )
            scaled_delta = self.scale_action(delta)
        else:
            # Otherwise, check to make sure set_velocity is size self.joint_dim
            assert len(set_velocity) == self.joint_dim,\
                "Goal action must be equal to the robot's joint dimension space! Expected {}, got {}".format(
                    self.joint_dim, len(set_velocity)
                )
            scaled_delta = None

        self.goal_vel = set_goal_position(scaled_delta,
                                          self.current_vel,
                                          position_limit=self.velocity_limits,
                                          set_pos=set_velocity)

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_vel)

    def run_controller(self, action=None):
        # Make sure goal has been set
        if not self.goal_vel.any():
            self.set_goal(np.zeros(self.control_dim))

        # First, update goal if action is not set to none
        # Action will be interpreted as delta value from current
        if action is not None:
            self.set_goal(action)
        else:
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

        # Compute torques (pre-compensation)
        self.torques = np.multiply(self.kv, (self.current_vel - self.joint_vel)) + self.torque_compensation

        # Return final torques
        return self.torques

    @property
    def name(self):
        return 'joint_vel'
