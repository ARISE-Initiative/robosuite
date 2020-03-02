import abc
import numpy as np
import mujoco_py


class Controller(object, metaclass=abc.ABCMeta):
    """
    General controller interface.

    Requires reference to mujoco sim object, id_name of specific robot, relevant joint_indexes to that robot, and
    whether an initial_joint is used for nullspace torques or not
    """
    def __init__(self,
                 sim,
                 id_name,
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
        self.id_name = id_name
        self.joint_index = joint_indexes

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
        self.joint_dim = len(joint_indexes)

        # Torques being outputted by the controller
        self.torques = None

        # Move forward one timestep to propagate updates before taking first update
        self.sim.forward()

        # Initialize controller by updating internal state and setting the initial joint, pos, and ori
        self.update()
        self.initial_joint = self.joint_pos
        self.initial_ee_pos = self.ee_pos
        self.initial_ee_ori_mat = self.ee_ori_mat
        self.torque_compensation = np.zeros(self.joint_dim)

    @abc.abstractmethod
    def run_controller(self, action):
        """
        Go from actions to torques
        """
        pass

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

        self.ee_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id(self.id_name)])
        self.ee_ori_mat = np.array(self.sim.data.body_xmat[self.sim.model.body_name2id(self.id_name)].reshape([3, 3]))
        self.ee_pos_vel = np.array(self.sim.data.body_xvelp[self.sim.model.body_name2id(self.id_name)])
        self.ee_ori_vel = np.array(self.sim.data.body_xvelr[self.sim.model.body_name2id(self.id_name)])

        self.joint_pos = np.array(self.sim.data.qpos[self.joint_index])
        self.joint_vel = np.array(self.sim.data.qvel[self.joint_index])

        self.J_pos = np.array(self.sim.data.get_body_jacp(self.id_name).reshape((3, -1))[:, self.joint_index])
        self.J_ori = np.array(self.sim.data.get_body_jacr(self.id_name).reshape((3, -1))[:, self.joint_index])
        self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

        mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        self.mass_matrix = mass_matrix[self.joint_index, :][:, self.joint_index]

    @property
    def input_min(self):
        return self._input_min

    @input_min.setter
    def input_min(self, input_min):
        self._input_min = np.array(input_min) if type(input_min) == list or type(input_min) == tuple \
            else np.array([input_min]*self.control_dim)

    @property
    def input_max(self):
        return self._input_max

    @input_max.setter
    def input_max(self, input_max):
        self._input_max = np.array(input_max) if type(input_max) == list or type(input_max) == tuple \
            else np.array([input_max]*self.control_dim)

    @property
    def output_min(self):
        return self._output_min

    @output_min.setter
    def output_min(self, output_min):
        self._output_min = np.array(output_min) if type(output_min) == list or type(output_min) == tuple \
            else np.array([output_min]*self.control_dim)

    @property
    def output_max(self):
        return self._output_max

    @output_max.setter
    def output_max(self, output_max):
        self._output_max = np.array(output_max) if type(output_max) == list or type(output_max) == tuple \
                else np.array([output_max]*self.control_dim)

    @property
    def control_dim(self):
        return self._control_dim

    @control_dim.setter
    def control_dim(self, control_dim):
        self._control_dim = control_dim

    @property
    def name(self):
        raise NotImplementedError


