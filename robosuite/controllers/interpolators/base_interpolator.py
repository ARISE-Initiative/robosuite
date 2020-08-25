import abc


class Interpolator(object, metaclass=abc.ABCMeta):
    """
    General interpolator interface.
    """

    @abc.abstractmethod
    def get_interpolated_goal(self, x):
        """
        Takes the current state and provides the next step in interpolation given
            the remaining steps.

        Args:
            x (np.array): Current state
        Returns:
            x_current (np.array): Next interpolated step
        """
        raise NotImplementedError





