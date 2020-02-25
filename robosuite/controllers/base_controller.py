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
                 initial_joint
    ):
        self.action_scale = None

        # Note: the following attributes must be specified in extended controller subclasses during the init call:
        # self.output_min
        # self.output_max
        # self.input_min
        # self.input_max

        # mujoco simulator state
        self.sim = sim
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

        self.update()
        self.initial_joint = initial_joint
        print("initial_joint: {}".format(self.initial_joint))       # TODO: Clean post-debugging
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

        self.model_timestep = self.sim.model.opt.timestep

        self.ee_pos = self.sim.data.body_xpos[self.sim.model.body_name2id(self.id_name)]
        self.ee_ori_mat = self.sim.data.body_xmat[self.sim.model.body_name2id(self.id_name)].reshape([3, 3])
        self.ee_pos_vel = self.sim.data.body_xvelp[self.sim.model.body_name2id(self.id_name)]
        self.ee_ori_vel = self.sim.data.body_xvelr[self.sim.model.body_name2id(self.id_name)]

        self.joint_pos = self.sim.data.qpos[self.joint_index]
        self.joint_vel = self.sim.data.qvel[self.joint_index]

        self.J_pos = self.sim.data.get_body_jacp(self.id_name).reshape((3, -1))[:, self.joint_index]
        self.J_ori = self.sim.data.get_body_jacr(self.id_name).reshape((3, -1))[:, self.joint_index]
        self.J_full = np.vstack([self.J_pos, self.J_ori])

        mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel) ** 2,), dtype=np.float64, order='C')
        mujoco_py.cymj._mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        self.mass_matrix = mass_matrix[self.joint_index, :][:, self.joint_index]




