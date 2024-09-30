import abc  # for abstract base class definitions


class Device(metaclass=abc.ABCMeta):
    """
    Base class for all robot controllers.
    Defines basic interface for all controllers to adhere to.
    Also contains shared logic for managing multiple and/or multiarmed robots.
    """

    def __init__(self, env):
        """
        Args:
            env (RobotEnv): The environment which contains the robot(s) to control
                            using this device.
        """
        self.env = env
        self.all_robot_arms = [robot.arms for robot in self.env.robots]
        self.num_robots = len(self.all_robot_arms)

    def _reset_internal_state(self):
        """
        Resets internal state related to robot control
        """
        self.grasp_states = [[False] * len(self.all_robot_arms[i]) for i in range(self.num_robots)]
        self.active_arm_indices = [0] * len(self.all_robot_arms)
        self.active_robot = 0
        self.base_modes = [False] * len(self.all_robot_arms)

    @property
    def active_arm(self):
        return self.all_robot_arms[self.active_robot][self.active_arm_index]

    @property
    def grasp(self):
        return self.grasp_states[self.active_robot][self.active_arm_index]

    @property
    def active_arm_index(self):
        return self.active_arm_indices[self.active_robot]

    @property
    def base_mode(self):
        return self.base_modes[self.active_robot]

    @active_arm_index.setter
    def active_arm_index(self, value):
        self.active_arm_indices[self.active_robot] = value

    @abc.abstractmethod
    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_controller_state(self) -> dict:
        """Returns the current state of the device, a dictionary of pos, orn, grasp, and reset."""
        raise NotImplementedError
