import abc  # for abstract base class definitions


class Device(metaclass=abc.ABCMeta):
    """
    Base class for all robot controllers.
    Defines basic interface for all controllers to adhere to.
    """

    @abc.abstractmethod
    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_controller_state(self):
        """Returns the current state of the device, a dictionary of pos, orn, grasp, and reset."""
        raise NotImplementedError
