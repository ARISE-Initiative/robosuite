import numpy as np
from robosuite.utils.buffers import DelayBuffer


class Observable:
    """
    Base class for all observables -- defines interface for interacting with sensors

    Args:
        name (str): Name for this observable
        sensor (function): Method to grab raw sensor data for this observable. Should take in
            no arguments and return the raw sensor data for the current timestep
        corrupter (function): Method to corrupt the raw sensor data for this observable. Should take in
            the output of @sensor and return the same type (corrupted data)
        delayer (function): Method to delay the raw sensor data when polling this observable. Should take in
            no arguments and return an integer, for the number of timesteps to delay
        data_size (int): Size of data array per sensor reading
        history_size (int): Size of the internal history buffer to store recent values
    """
    def __init__(self, name, sensor, corrupter, delayer, data_size, history_size=50):
        # Set all internal variables and methods
        self.name = name
        self.sensor = sensor
        self.corrupter = corrupter
        self.delayer = delayer
        self._current_observed_value = None          # Will be modified later
        self._data_size = data_size
        self._history_size = history_size
        self._history = DelayBuffer(dim=data_size, length=history_size)

        # Enabled by default
        self._enabled = True

    def update(self):
        """
        Updates internal values for this observable, if enabled
        """
        if self._enabled:
            # Get newest value, corrupt it, and store it in the history buffer
            self._history.push(self.corrupter(self.sensor()))
            # Update current observed value
            obs = self._history.get_delayed_value(delay=self.delayer())
            # Make sure to convert to single number if data_size is 1
            self._current_observed_value = obs[0] if self._data_size == 1 else obs

    def set_enabled(self, enabled):
        """
        Sets whether this observable is active or not

        Args:
            enabled (bool): True if this observable should be enabled
        """
        self._enabled = enabled

    def set_sensor(self, sensor):
        """
        Sets the sensor for this observable.

        Args:
            sensor (function): Method to grab raw sensor data for this observable. Should take in
                no arguments and return the raw sensor data for the current timestep
        """
        self.sensor = sensor

    def set_corrupter(self, corrupter):
        """
        Sets the corrupter for this observable.

        Args:
             corrupter (function): Method to corrupt the raw sensor data for this observable. Should take in
            the output of self.sensor and return the same type (corrupted data)
        """
        self.corrupter = corrupter

    def set_delayer(self, delayer):
        """
        Sets the delayer for this observable.

        Args:
            delayer (function): Method to delay the raw sensor data when polling this observable. Should take in
                no arguments and return an integer, for the number of timesteps to delay
        """
        self.delayer = delayer

    @property
    def observation(self):
        """
        Current observation from this observable

        Returns:
            float or np.array: Current observed value from this observable
        """
        return self._current_observed_value


class ImageObservable(Observable):
    """
    Class for images (multi-dimensional sensor readings)

    Args:
        name (str): Name for this observable
        sensor (function): Method to grab raw sensor data for this observable. Should take in
            no arguments and return the raw sensor data for the current timestep
        corrupter (function): Method to corrupt the raw sensor data for this observable. Should take in
            the output of @sensor and return the same type (corrupted data)
        delayer (function): Method to delay the raw sensor data when polling this observable. Should take in
            no arguments and return an integer, for the number of timesteps to delay
        image_shape (tuple): Shape of the image, e.g. for RGB, could be (H x W x 3)
        history_size (int): Size of the internal history buffer to store recent values
    """
    def __init__(self, name, sensor, corrupter, delayer, image_shape, history_size=50):
        # Store image shape and find flattened image_shape to store data
        self.image_shape = np.array(image_shape)
        flattened_shape = np.product(self.image_shape)
        # Run super init
        super().__init__(name=name, sensor=sensor, corrupter=corrupter, delayer=delayer,
                         data_size=flattened_shape, history_size=history_size)

    def update(self):
        """
        Updates internal values for this observable, if enabled. Overrides superclass method to make sure
        images are returned appropriately in their original multi-dimensional form
        """
        if self._enabled:
            # Get newest value, corrupt it, flatten it, and store it in the history buffer
            self._history.push(self.corrupter(self.sensor()).flatten())
            # Update current observed value
            obs = self._history.get_delayed_value(delay=self.delayer())
            # Convert back to multi-dimensional size
            self._current_observed_value = obs.reshape(self.image_shape)


def create_uniform_noise_corrupter(low, high):
    """
    Creates a corrupter that applies uniform noise to a given input within range @low to @high

    Args:
        low (float): Low-end of noise to apply
        high (float): High-end of noise to apply
    """
    def corrupter(inp):
        inp = np.array(inp)
        noise = (high - low) * np.random.random_sample(inp.shape) + low
        return inp + noise
    return corrupter


def create_gaussian_noise_corrupter(mean, std):
    """
    Creates a corrupter that applies gaussian noise to a given input with mean @mean and std dev @std

    Args:
        mean (float): Mean of the noise to apply
        std (float): Standard deviation of the noise to apply
    """
    def corrupter(inp):
        inp = np.array(inp)
        noise = mean + std * np.random.randn(*inp.shape)
        return inp + noise
    return corrupter


def create_determinstic_delayer(delay):
    """
    Create a deterministic delayer that always returns the same delay value

    Args:
        delay (int): Delay value to return
    """
    return lambda: delay


def create_uniform_sampled_delayer(low, high):
    """
    Creates uniformly sampled delayer, with minimum delay @low and maximum delay @high, both inclusive

    Args:
        low (int): Minimum possible delay
        high (int): Maxmimum possible delay
    """
    return lambda: np.random.randint(low=low, high=high+1)


def create_gaussian_sampled_delayer(mean, std):
    """
    Creates a gaussian sampled delayer, with average delay @mean which varies by standard deviation @std

    Args:
        mean (float): Average delay
        std (float): Standard deviation of the delay variation
    """
    return lambda: int(np.round(mean + std * np.random.randn()))
