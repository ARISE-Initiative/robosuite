import abc
import numpy as np


class Interpolator(object, metaclass=abc.ABCMeta):
    """
    General interpolator interface.
    """

    @abc.abstractmethod
    def get_interpolated_goal(self, goal):
        """
        Go from actions to torques
        """
        pass





