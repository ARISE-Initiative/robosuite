import abc
import numpy as np
import mujoco_py
from collections.abc import Iterable


class Controller(object, metaclass=abc.ABCMeta):
    """
    General controller interface.

    Requires reference to mujoco sim object, eef_name of specific robot, relevant joint_indexes to that robot, and
    whether an initial_joint is used for nullspace torques or not

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:
            "joints" : list of indexes to relevant robot joints
            "qpos" : list of indexes to relevant robot joint positions
            "qvel" : list of indexes to relevant robot joint velocities
    """
    def __init__(self,
                 sim,
                 eef_name,
                 joint_indexes,
    ):

        # Attributes for scaling / clipping inputs to outputs
        self.action_scale = None
        self.action_input_transform = None
        self.action_output_transform = None

        # Private property attributes
        self._control_dim = None
        self._output_min = None
        self._output_max = None
        self._input_min = None
        self._input_max = None

        # mujoco simulator state
        self.sim = sim
        self.model_timestep = self.sim.model.opt.timestep
        self.eef_name = eef_name
        self.joint_index = joint_indexes["joints"]
        self.qpos_index = joint_indexes["qpos"]
        self.qvel_index = joint_indexes["qvel"]

        # robot states
        self.ee_pos = None
        self.ee_ori_mat = None
        self.ee_pos_vel = None
        self.ee_ori_vel = None
        self.joint_pos = None
        self.joint_vel = None

        # dynamics and kinematics
        self.J_pos = None
        self.J_ori = None
        self.J_full = None
        self.mass_matrix = None

        # Joint dimension
        self.joint_dim = len(joint_indexes["joints"])

        # Torques being outputted by the controller
        self.torques = None

        # Move forward one timestep to propagate updates before taking first update
        self.sim.forward()

        # Initialize controller by updating internal state and setting the initial joint, pos, and ori
        self.update()
        self.initial_joint = self.joint_pos
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat

    @abc.abstractmethod
    def run_controller(self):
        """
        Abstract method that should be implemented in all subclass controllers
        Converts a given action into torques (pre gravity compensation) to be executed on the robot
        """
        raise NotImplementedError

    def scale_action(self, action):
        """
        Scale the action based on max and min of action
        """

        if self.action_scale is None:
            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
        action = np.clip(action, self.input_min, self.input_max)
        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

        return transformed_action

    def update(self):
        """
        Updates the state of the robot arm, including end effector pose / orientation / velocity, joint pos/vel,
        jacobian, and mass matrix
        """
        self.ee_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(self.eef_name)])
        self.ee_ori_mat = np.array(self.sim.data.body_xmat[self.sim.model.body_name2id(self.eef_name)].reshape([3, 3]))
        self.ee_pos_vel = np.array(self.sim.data.body_xvelp[self.sim.model.body_name2id(self.eef_name)])
        self.ee_ori_vel = np.array(self.sim.data.body_xvelr[self.sim.model.body_name2id(self.eef_name)])

        self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
        self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

        self.J_pos = np.array(self.sim.data.get_body_jacp(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        self.J_ori = np.array(self.sim.data.get_body_jacr(self.eef_name).reshape((3, -1))[:, self.qvel_index])
        self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

        mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        self.mass_matrix = mass_matrix[self.joint_index, :][:, self.joint_index]

    @property
    def input_min(self):
        """Returns input minimum below which an inputted action will be clipped"""
        return self._input_min

    @input_min.setter
    def input_min(self, input_min):
        """Sets the minimum input"""
        self._input_min = np.array(input_min) if isinstance(input_min, Iterable) \
            else np.array([input_min]*self.control_dim)

    @property
    def input_max(self):
        """Returns input maximum above which an inputted action will be clipped"""
        return self._input_max

    @input_max.setter
    def input_max(self, input_max):
        """Sets the maximum input"""
        self._input_max = np.array(input_max) if isinstance(input_max, Iterable) \
            else np.array([input_max]*self.control_dim)

    @property
    def output_min(self):
        """Returns output minimum which defines lower end of scaling range when scaling an input action"""
        return self._output_min

    @output_min.setter
    def output_min(self, output_min):
        """Set the minimum output"""
        self._output_min = np.array(output_min) if isinstance(output_min, Iterable) \
            else np.array([output_min]*self.control_dim)

    @property
    def output_max(self):
        """Returns output maximum which defines upper end of scaling range when scaling an input action"""
        return self._output_max

    @output_max.setter
    def output_max(self, output_max):
        """Set the maximum output"""
        self._output_max = np.array(output_max) if isinstance(output_max, Iterable) \
            else np.array([output_max]*self.control_dim)

    @property
    def control_dim(self):
        """Returns the control dimension for this controller (specifies size of action space)"""
        return self._control_dim

    @control_dim.setter
    def control_dim(self, control_dim):
        """Sets the control dimension for this controller"""
        self._control_dim = control_dim

    @property
    def torque_compensation(self):
        """Returns gravity compensation torques for the robot arm"""
        return self.sim.data.qfrc_bias[self.qvel_index]

    @property
    def name(self):
        """Returns the name of this controller"""
        raise NotImplementedError


