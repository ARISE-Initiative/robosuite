import abc  # for abstract base class definitions


class Controller(metaclass=abc.ABCMeta):
    """
    Base class for all robot controllers.
    Defines basic interface for all controllers to adhere to.
    """

    def __init__(self, bullet_data_path, robot_jpos_getter):
        """
        Args:
            bullet_data_path (str): base path to bullet data.

            robot_jpos_getter (function): function that returns the position of the joints
                as a numpy array of the right dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_control(self, *args, **kwargs):
        """
        Retrieve a control input from the controller.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sync_state(self):
        """
        This function does internal bookkeeping to maintain
        consistency between the robot being controlled and
        the controller state.
        """
        raise NotImplementedError
