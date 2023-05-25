"""
This file implements a wrapper for visualizing important sites in a given environment.

By default, this visualizes all sites possible for the environment. Visualization options
for a given environment can be found by calling `get_visualization_settings()`, and can
be set individually by calling `set_visualization_setting(setting, visible)`.
"""
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np

from robosuite.utils.mjcf_utils import new_body, new_geom, new_site
from robosuite.wrappers import Wrapper

DEFAULT_INDICATOR_SITE_CONFIG = {
    "type": "sphere",
    "size": [0.03],
    "rgba": [1, 0, 0, 0.5],
}


class VisualizationWrapper(Wrapper):
    def __init__(self, env, indicator_configs=None):
        """
        Initializes the data collection wrapper. Note that this automatically conducts a (hard) reset initially to make
        sure indicators are properly added to the sim model.

        Args:
            env (MujocoEnv): The environment to visualize

            indicator_configs (None or str or dict or list): Configurations to use for indicator objects.

                If None, no indicator objects will be used

                If a string, this should be `'default'`, which corresponds to single default spherical indicator

                If a dict, should specify a single indicator object config

                If a list, should specify specific indicator object configs to use for multiple indicators (which in
                turn can either be `'default'` or a dict)

                As each indicator object is essentially a site element, each dict should map site attribute keywords to
                values. Note that, at the very minimum, the `'name'` attribute MUST be specified for each indicator. See
                http://www.mujoco.org/book/XMLreference.html#site for specific site attributes that can be specified.
        """
        super().__init__(env)

        # Make sure that the environment is NOT using segmentation sensors, since we cannot use segmentation masks
        # with visualization sites simultaneously
        assert all(
            seg is None for seg in env.camera_segmentations
        ), "Cannot use camera segmentations with visualization wrapper!"

        # Standardize indicator configs
        self.indicator_configs = None
        if indicator_configs is not None:
            self.indicator_configs = []
            if type(indicator_configs) in {str, dict}:
                indicator_configs = [indicator_configs]
            for i, indicator_config in enumerate(indicator_configs):
                if indicator_config == "default":
                    indicator_config = deepcopy(DEFAULT_INDICATOR_SITE_CONFIG)
                    indicator_config["name"] = f"indicator{i}"
                # Make sure name attribute is specified
                assert "name" in indicator_config, "Name must be specified for all indicator object configurations!"
                # Add this configuration to the internal array
                self.indicator_configs.append(indicator_config)

        # Create internal dict to store visualization settings (set to True by default)
        self._vis_settings = {vis: True for vis in self.env._visualizations}

        # Add the post-processor to make sure indicator objects get added to model before it's actually loaded in sim
        self.env.set_xml_processor(processor=self._add_indicators_to_model)

        # Conduct a (hard) reset to make sure visualization changes propagate
        reset_mode = self.env.hard_reset
        self.env.hard_reset = True
        self.reset()
        self.env.hard_reset = reset_mode

    def get_indicator_names(self):
        """
        Gets all indicator object names for this environment.

        Returns:
            list: Indicator names for this environment.
        """
        return (
            [ind_config["name"] for ind_config in self.indicator_configs] if self.indicator_configs is not None else []
        )

    def set_indicator_pos(self, indicator, pos):
        """
        Sets the specified @indicator to the desired position @pos

        Args:
            indicator (str): Name of the indicator to set
            pos (3-array): (x, y, z) Cartesian world coordinates to set the specified indicator to
        """
        # Make sure indicator is valid
        indicator_names = set(self.get_indicator_names())
        assert indicator in indicator_names, "Invalid indicator name specified. Valid options are {}, got {}".format(
            indicator_names, indicator
        )
        # Set the specified indicator
        self.env.sim.model.body_pos[self.env.sim.model.body_name2id(indicator + "_body")] = np.array(pos)

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
        assert (
            setting in self._vis_settings
        ), "Invalid visualization setting specified. Valid options are {}, got {}".format(
            self._vis_settings.keys(), setting
        )
        self._vis_settings[setting] = visible

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate visualization

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        # Update any visualization
        self.env.visualize(vis_settings=self._vis_settings)
        return ret

    def step(self, action):
        """
        Extends vanilla step() function call to accommodate visualization

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

    def _add_indicators_to_model(self, xml):
        """
        Adds indicators to the mujoco simulation model

        Args:
            xml (string): MJCF model in xml format, for the current simulation to be loaded
        """
        if self.indicator_configs is not None:
            root = ET.fromstring(xml)
            worldbody = root.find("worldbody")

            for indicator_config in self.indicator_configs:
                config = deepcopy(indicator_config)
                indicator_body = new_body(name=config["name"] + "_body", pos=config.pop("pos", (0, 0, 0)))
                indicator_body.append(new_site(**config))
                worldbody.append(indicator_body)

            xml = ET.tostring(root, encoding="utf8").decode("utf8")

        return xml
