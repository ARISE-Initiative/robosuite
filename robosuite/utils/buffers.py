"""
Collection of Buffer objects with general functionality
"""


import numpy as np


class Buffer(object):
    """
    Abstract class for different kinds of data buffers. Minimum API should have a "push" and "clear" method
    """

    def push(self, value):
        """
        Pushes a new @value to the buffer

        Args:
            value: Value to push to the buffer
        """
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class RingBuffer(Buffer):
    """
    Simple RingBuffer object to hold values to average (useful for, e.g.: filtering D component in PID control)

    Note that the buffer object is a 2D numpy array, where each row corresponds to
    individual entries into the buffer

    Args:
        dim (int): Size of entries being added. This is, e.g.: the size of a state vector that is to be stored
        length (int): Size of the ring buffer
    """
    def __init__(self, dim, length):
        # Store input args
        self.dim = dim
        self.length = length

        # Variable so that initial average values are accurate
        self._size = 0

        # Save pointer to current place in the buffer
        self.ptr = 0

        # Construct ring buffer
        self.buf = np.zeros((length, dim))

    def push(self, value):
        """
        Pushes a new value into the buffer

        Args:
            value (int or float or array): Value(s) to push into the array (taken as a single new element)
        """
        # Add value, then increment pointer (and size if necessary)
        self.buf[self.ptr] = np.array(value)
        self.ptr = (self.ptr + 1) % self.length
        if self._size < self.length:
            self._size += 1

    def clear(self):
        """
        Clears buffer and reset pointer
        """
        self.buf = np.zeros((self.length, self.dim))
        self.ptr = 0
        self._size = 0

    @property
    def average(self):
        """
        Gets the average of components in buffer

        Returns:
            float or np.array: Averaged value of all elements in buffer
        """
        return np.mean(self.buf[:self._size], axis=0)


class DeltaBuffer(Buffer):
    """
    Simple 2-length buffer object to streamline grabbing delta values between "current" and "last" values

    Constructs delta object.

    Args:
        dim (int): Size of numerical arrays being inputted
        init_value (None or Iterable): Initial value to fill "last" value with initially.
            If None (default), last array will be filled with zeros
    """
    def __init__(self, dim, init_value=None):
        # Setup delta object
        self.dim = dim
        self.last = np.zeros(self.dim) if init_value is None else np.array(init_value)
        self.current = np.zeros(self.dim)

    def push(self, value):
        """
        Pushes a new value into the buffer; current becomes last and @value becomes current

        Args:
            value (int or float or array): Value(s) to push into the array (taken as a single new element)
        """
        self.last = self.current
        self.current = np.array(value)

    def clear(self):
        """
        Clears last and current value
        """
        self.last, self.current = np.zeros(self.dim), np.zeros(self.dim)

    @property
    def delta(self, abs_value=False):
        """
        Returns the delta between last value and current value. If abs_value is set to True, then returns
        the absolute value between the values

        Args:
            abs_value (bool): Whether to return absolute value or not

        Returns:
            float or np.array: difference between current and last value
        """
        return self.current - self.last if not abs_value else np.abs(self.current - self.last)

    @property
    def average(self):
        """
        Returns the average between the current and last value

        Returns:
            float or np.array: Averaged value of all elements in buffer
        """
        return (self.current + self.last) / 2.0


class DelayBuffer(RingBuffer):
    """
    Modified RingBuffer that returns delayed values when polled
    """

    def get_delayed_value(self, delay):
        """
        Returns value @delay increments behind most recent value.

        Args:
            delay (int): How many steps backwards from most recent value to grab value. Note that this should not be
                greater than the buffer's length

        Returns:
            np.array: delayed value
        """
        # First make sure that the delay is valid
        assert delay < self.length, "Requested delay must be less than buffer's length!"
        # Grab delayed value
        return self.buf[(self.ptr - delay) % self.length]
