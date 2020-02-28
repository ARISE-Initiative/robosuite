from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np


class JointTorqueController(Controller):
    """
    Controller for joint torque
    """

    def __init__(self,
                 sim,
                 robot_id,
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

        super(JointTorqueController, self).__init__(
            sim,
            robot_id,
            joint_indexes
        )

        # Control dimension
        self.control_dim = len(joint_indexes)

        # input and output max and min
        self.input_max = input_max
        self.input_min = input_min
        self.output_max = output_max
        self.output_min = output_min

        # limits
        self.torque_limits = np.array(torque_limits)

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator = interpolator

        # initialize torques
        self.goal_torque = None                           # Goal torque desired, pre-compensation
        self.current_torque = np.zeros(self.control_dim)  # Current torques being outputted, pre-compensation
        self.torques = None                               # Torques returned every time run_controller is called

    def set_goal(self, delta, set_torque=None):
        self.update()

        # Check to make sure delta is size self.joint_dim
        assert len(delta) == self.control_dim, "Delta torque must be equal to the robot's joint dimension space!"

        if delta is not None:
            scaled_delta = self.scale_action(delta)
        else:
            scaled_delta = None

        self.goal_torque = set_goal_position(scaled_delta,
                                             self.current_torque,
                                             position_limit=self.torque_limits,
                                             set_pos=set_torque)

        if self.interpolator is not None:
            self.interpolator.set_goal(self.goal_torque)

    def run_controller(self, action=None):
        # Make sure goal has been set
        if not self.goal_torque.all():
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

        # Add torque compensation
        self.torques = self.current_torque + self.torque_compensation

        # Return final torques
        return self.torques
