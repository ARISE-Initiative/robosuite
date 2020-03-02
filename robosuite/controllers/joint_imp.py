from robosuite.controllers.base_controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np


class JointImpController(Controller):
    """
    Controller for joint impedance
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
                 qpos_limits=None,
                 interpolator=None,
                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super(JointImpController, self).__init__(
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
        self.position_limits = np.array(qpos_limits)

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

    def run_controller(self, action=None):
        # Make sure goal has been set
        if not self.goal_qpos.all():
            self.set_goal(np.zeros(self.control_dim))

        # Then, update goal if action is not set to none
        # Action will be interpreted as delta value from current
        if action is not None:
            self.set_goal(action)
        else:
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

        position_error = desired_qpos - self.joint_pos
        vel_pos_error = -self.joint_vel
        desired_torque = (np.multiply(np.array(position_error), np.array(self.kp))
                          + np.multiply(vel_pos_error, self.kv))

        self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation

        return self.torques

    @property
    def name(self):
        return 'joint_imp'
