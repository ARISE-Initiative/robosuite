import numpy as np


def sensor(modality):
    """
    Decorator that should be added to any sensors that will be an observable.

    Decorated functions should have signature:

        any = func(obs_cache)

    Args:
        modality (str): Modality for this sensor
    """
    # Define standard decorator (with no args)
    def decorator(func):
        # Add modality attribute
        func.__modality__ = modality
        # Return function
        return func
    return decorator


def create_deterministic_corrupter(corruption, low=-np.inf, high=np.inf):
    """
    Creates a deterministic corrupter that applies the same corrupted value to all sensor values

    Args:
        corruption (float): Corruption to apply
        low (float): Minimum value for output for clipping
        high (float): Maximum value for output for clipping
    """
    def corrupter(inp):
        inp = np.array(inp)
        return np.clip(inp + corruption, low, high)
    return corrupter


def create_uniform_noise_corrupter(min_noise, max_noise, low=-np.inf, high=np.inf):
    """
    Creates a corrupter that applies uniform noise to a given input within range @low to @high

    Args:
        min_noise (float): Minimum noise to apply
        max_noise (float): Maximum noise to apply
        low (float): Minimum value for output for clipping
        high (float): Maxmimum value for output for clipping
    """
    def corrupter(inp):
        inp = np.array(inp)
        noise = (max_noise - min_noise) * np.random.random_sample(inp.shape) + min_noise
        return np.clip(inp + noise, low, high)
    return corrupter


def create_gaussian_noise_corrupter(mean, std, low=-np.inf, high=np.inf):
    """
    Creates a corrupter that applies gaussian noise to a given input with mean @mean and std dev @std

    Args:
        mean (float): Mean of the noise to apply
        std (float): Standard deviation of the noise to apply
        low (float): Minimum value for output for clipping
        high (float): Maxmimum value for output for clipping
    """
    def corrupter(inp):
        inp = np.array(inp)
        noise = mean + std * np.random.randn(*inp.shape)
        return np.clip(inp + noise, low, high)
    return corrupter


def create_deterministic_delayer(delay):
    """
    Create a deterministic delayer that always returns the same delay value

    Args:
        delay (float): Delay value to return
    """
    assert delay >= 0, "Inputted delay must be non-negative!"
    return lambda: delay


def create_uniform_sampled_delayer(min_delay, max_delay):
    """
    Creates uniformly sampled delayer, with minimum delay @low and maximum delay @high, both inclusive

    Args:
        min_delay (float): Minimum possible delay
        max_delay (float): Maxmimum possible delay
    """
    assert min(min_delay, max_delay) >= 0, "Inputted delay must be non-negative!"
    return lambda: min_delay + (max_delay - min_delay) * np.random.random()


def create_gaussian_sampled_delayer(mean, std):
    """
    Creates a gaussian sampled delayer, with average delay @mean which varies by standard deviation @std

    Args:
        mean (float): Average delay
        std (float): Standard deviation of the delay variation
    """
    assert mean >= 0, "Inputted mean delay must be non-negative!"
    return lambda: max(0.0, int(np.round(mean + std * np.random.randn())))


# Common defaults to use
NO_CORRUPTION = lambda inp: inp
NO_FILTER = lambda inp: inp
NO_DELAY = lambda: 0.0


class Observable:
    """
    Base class for all observables -- defines interface for interacting with sensors

    Args:
        name (str): Name for this observable
        sensor (function with sensor decorator): Method to grab raw sensor data for this observable. Should take in a
            single dict argument (observation cache if a pre-computed value is required) and return the raw sensor data
            for the current timestep. Must handle case if inputted argument is empty ({}), and should have `sensor`
            decorator when defined
        corrupter (None or function): Method to corrupt the raw sensor data for this observable. Should take in
            the output of @sensor and return the same type (corrupted data). If None, results in default no corruption
        filter (None or function): Method to filter the outputted reading for this observable. Should take in the output
            of @corrupter and return the same type (filtered data). If None, results in default no filter
        delayer (None or function): Method to delay the raw sensor data when polling this observable. Should take in
            no arguments and return a float, for the number of seconds to delay the measurement by. If None, results in
            default no delayer
        sampling_rate (float): Sampling rate for this observable (Hz)
        enabled (bool): Whether this sensor is enabled or not. If enabled, this observable's values
            are continually computed / updated every time update() is called.
        active (bool): Whether this sensor is active or not. If active, this observable's current
            observed value is returned from self.obs, otherwise self.obs returns None.
    """
    def __init__(
            self,
            name,
            sensor,
            corrupter=None,
            filter=None,
            delayer=None,
            sampling_rate=20,
            enabled=True,
            active=True,
    ):
        # Set all internal variables and methods
        self.name = name
        self._sensor = sensor
        self._corrupter = corrupter if corrupter is not None else NO_CORRUPTION
        self._filter = filter if filter is not None else NO_FILTER
        self._delayer = delayer if delayer is not None else NO_DELAY
        self._sampling_timestep = 1. / sampling_rate
        self._enabled = enabled
        self._active = active
        self._is_number = False                                     # filled in during sensor check call
        self._data_shape = (1,)                                     # filled in during sensor check call

        # Make sure sensor is working
        self._check_sensor_validity()

        # These values will be modified during update() call
        self._time_since_last_sample = 0.0                          # seconds
        self._current_delay = self._delayer()                       # seconds
        self._current_observed_value = 0 if self._is_number else np.zeros(self._data_shape)
        self._sampled = False

    def update(self, timestep, obs_cache):
        """
        Updates internal values for this observable, if enabled.

        Args:
            timestep (float): Amount of simulation time (in sec) that has passed since last call.
            obs_cache (dict): Observation cache mapping observable names to pre-computed values to pass to sensor. This
                will be updated in-place during this call.
        """
        if self._enabled:
            # Increment internal time counter
            self._time_since_last_sample += timestep

            # If the delayed sampling time has been passed and we haven't sampled yet for this sampling period,
            # we should grab a new measurement
            if not self._sampled and self._sampling_timestep - self._current_delay >= self._time_since_last_sample:
                # Get newest raw value, corrupt it, filter it, and set it as our current observed value
                obs = np.array(self._filter(self._corrupter(self._sensor(obs_cache))))
                self._current_observed_value = obs[0] if len(obs.shape) == 1 and obs.shape[0] == 1 else obs
                # Update cache entry as well
                obs_cache[self.name] = np.array(self._current_observed_value)
                # Toggle sampled and re-sample next time delay
                self._sampled = True
                self._current_delay = self._delayer()

            # If our total time since last sample has surpassed our sampling timestep,
            # then we reset our timer and sampled flag
            if self._time_since_last_sample >= self._sampling_timestep:
                if not self._sampled:
                    # If we still haven't sampled yet, sample immediately and warn user that sampling rate is too low
                    print("Warning: sampling rate is either too low or delay is too high. Please adjust one (or both)")
                    # Get newest raw value, corrupt it, filter it, and set it as our current observed value
                    obs = np.array(self._filter(self._corrupter(self._sensor(obs_cache))))
                    self._current_observed_value = obs[0] if len(obs.shape) == 1 and obs.shape[0] == 1 else obs
                    # Update cache entry as well
                    obs_cache[self.name] = np.array(self._current_observed_value)
                    # Re-sample next time delay
                    self._current_delay = self._delayer()
                self._time_since_last_sample %= self._sampling_timestep
                self._sampled = False

    def reset(self):
        """
        Resets this observable's internal values (but does not reset its sensor, corrupter, delayer, or filter)
        """
        self._time_since_last_sample = 0.0
        self._current_delay = self._delayer()
        self._current_observed_value = 0 if self._is_number else np.zeros(self._data_shape)

    def is_enabled(self):
        """
        Determines whether observable is enabled or not. This observable is considered enabled if its values
        are being continually computed / updated during each update() call.

        Returns:
            bool: True if this observable is enabled
        """
        return self._enabled

    def is_active(self):
        """
        Determines whether observable is active or not. This observable is considered active if its current observation
            value is being returned in self.obs.

        Returns:
            bool: True if this observable is active
        """
        return self._active

    def set_enabled(self, enabled):
        """
        Sets whether this observable is enabled or not. If enabled, this observable's values
        are continually computed / updated every time update() is called.

        Args:
            enabled (bool): True if this observable should be enabled
        """
        self._enabled = enabled
        # Reset values
        self.reset()

    def set_active(self, active):
        """
        Sets whether this observable is active or not. If active, this observable's current
            observed value is returned from self.obs, otherwise self.obs returns None.

        Args:
            active (bool): True if this observable should be active
        """
        self._active = active

    def set_sensor(self, sensor):
        """
        Sets the sensor for this observable.

        Args:
            sensor (function with sensor decorator): Method to grab raw sensor data for this observable. Should take in
                a single dict argument (observation cache if a pre-computed value is required) and return the raw
                sensor data for the current timestep. Must handle case if inputted argument is empty ({}), and should
                have `sensor` decorator when defined
        """
        self._sensor = sensor
        self._check_sensor_validity()

    def set_corrupter(self, corrupter):
        """
        Sets the corrupter for this observable.

        Args:
             corrupter (None or function): Method to corrupt the raw sensor data for this observable. Should take in
                the output of self.sensor and return the same type (corrupted data).
                If None, results in default no corruption
        """
        self._corrupter = corrupter if corrupter is not None else NO_CORRUPTION

    def set_filter(self, filter):
        """
        Sets the filter for this observable.

        Args:
             filter (None or function): Method to filter the outputted reading for this observable. Should take in
                the output of @corrupter and return the same type (filtered data).
                If None, results in default no filter
        """
        self._filter = filter if filter is not None else NO_FILTER

    def set_delayer(self, delayer):
        """
        Sets the delayer for this observable.

        Args:
            delayer (None or function): Method to delay the raw sensor data when polling this observable. Should take
                in no arguments and return a float, for the number of seconds to delay the measurement by.
                If None, results in default no filter
        """
        self._delayer = delayer if delayer is not None else NO_DELAY

    def set_sampling_rate(self, rate):
        """
        Sets the sampling rate for this observable.

        Args:
            rate (int): New sampling rate for this observable (Hz)
        """
        self._sampling_timestep = 1. / rate

    def _check_sensor_validity(self):
        """
        Internal function that checks the validity of this observable's sensor. It does the following:

            - Asserts that the inputted sensor has its __modality__ attribute defined from the sensor decorator
            - Asserts that the inputted sensor can handle the empty dict {} arg case
            - Updates the corresponding name, and data-types for this sensor
        """
        try:
            _ = self.modality
            self._data_shape = np.array(self._sensor({})).shape
            self._is_number = len(self._data_shape) == 1 and self._data_shape[0] == 1
        except:
            raise ValueError("Current sensor for observable {} is invalid.".format(self.name))

    @property
    def obs(self):
        """
        Current observation from this observable

        Returns:
            None or float or np.array: If active, current observed value from this observable. Otherwise, None
        """
        return self._current_observed_value if self._active else None

    @property
    def modality(self):
        """
        Modality of this sensor

        Returns:
            str: Modality name for this observable
        """
        return self._sensor.__modality__
