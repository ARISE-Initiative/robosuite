"""
This file implements a wrapper for visualizing important sites in a given environment.

By default, this visualizes all sites possible for the environment. Visualization options
for a given environment can be found by calling `get_visualization_settings()`, and can
be set individually by calling `set_visualization_setting(setting, visible)`.
"""
from robosuite.wrappers import Wrapper


class VisualizationWrapper(Wrapper):
    def __init__(self, env):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to visualize
        """
        super().__init__(env)

        # Create internal dict to store visualization settings (set to True by default)
        self._vis_settings = {vis: True for vis in self.env._visualizations}

        # Update visualizations for this environment
        self.env.visualize(vis_settings=self._vis_settings)

    def get_visualization_settings(self):
        """
        Gets all settings for visualizing this environment

        Returns:
            list: Visualization keywords for this environment.
        """
        return self._vis_settings.keys()

    def set_visualization_setting(self, setting, visible):
        """
        Sets the specified @setting to have visibility = @visible.

        Args:
            setting (str): Visualization keyword to set
            visible (bool): True if setting should be visualized.
        """
        assert setting in self._vis_settings, "Invalid visualization setting specified. Valid options are {}, got {}".\
            format(self._vis_settings.keys(), setting)
        self._vis_settings[setting] = visible

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        # Update any visualization
        self.env.visualize(vis_settings=self._vis_settings)
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ret = super().step(action)

        # Update any visualization
        self.env.visualize(vis_settings=self._vis_settings)

        return ret

