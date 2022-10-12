"""
This file contains a wrapper for sampling environment states
from a set of demonstrations on every reset. The main use case is for 
altering the start state distribution of training episodes for 
learning RL policies.
"""

import os
import random
import time

import h5py
import numpy as np

from robosuite.wrappers import Wrapper


class DemoSamplerWrapper(Wrapper):
    """
    Initializes a wrapper that provides support for resetting the environment
    state to one from a demonstration. It also supports curriculums for
    altering how often to sample from demonstration vs. sampling a reset
    state from the environment.

    Args:
        env (MujocoEnv): The environment to wrap.

        demo_path (str): The path to the folder containing the demonstrations.
            There should be a `demo.hdf5` file and a folder named `models` with
            all of the stored model xml files from the demonstrations.

        need_xml (bool): If True, the mujoco model needs to be reloaded when
            sampling a state from a demonstration. This could be because every
            demonstration was taken under varied object properties, for example.
            In this case, every sampled state comes with a corresponding xml to
            be used for the environment reset.

        num_traj (int): If provided, subsample @number demonstrations from the
            provided set of demonstrations instead of using all of them.

        sampling_schemes (list of str): A list of sampling schemes
            to be used. The following strings are valid schemes:

                `'random'`: sample a reset state directly from the wrapped environment

                `'uniform'`: sample a state from a demonstration uniformly at random

                `'forward'`: sample a state from a window that grows progressively from
                    the start of demonstrations

                `'reverse'`: sample a state from a window that grows progressively from
                    the end of demonstrations

        scheme_ratios (list of float --> np.array): A list of probability values to
            assign to each member of @sampling_schemes. Must be non-negative and
            sum to 1.

        open_loop_increment_freq (int): How frequently to increase
            the window size in open loop schemes ("forward" and "reverse"). The
            window size will increase by @open_loop_window_increment every
            @open_loop_increment_freq samples. Only samples that are generated
            by open loop schemes contribute to this count.

        open_loop_initial_window_width (int): The width of the initial sampling
            window, in terms of number of demonstration time steps, for
            open loop schemes.

        open_loop_window_increment (int): The window size will increase by
            @open_loop_window_increment every @open_loop_increment_freq samples.
            This number is in terms of number of demonstration time steps.

    Raises:
        AssertionError: [Incompatible envs]
        AssertionError: [Invalid sampling scheme]
        AssertionError: [Invalid scheme ratio]
    """

    def __init__(
        self,
        env,
        demo_path,
        need_xml=False,
        num_traj=-1,
        sampling_schemes=("uniform", "random"),
        scheme_ratios=(0.9, 0.1),
        open_loop_increment_freq=100,
        open_loop_initial_window_width=25,
        open_loop_window_increment=25,
    ):
        super().__init__(env)

        self.demo_path = demo_path
        hdf5_path = os.path.join(self.demo_path, "demo.hdf5")
        self.demo_file = h5py.File(hdf5_path, "r")

        # ensure that wrapped env matches the env on which demonstrations were collected
        env_name = self.demo_file["data"].attrs["env"]
        assert (
            env_name == self.unwrapped.__class__.__name__
        ), "Wrapped env {} does not match env on which demos were collected ({})".format(
            env.__class__.__name__, env_name
        )

        # list of all demonstrations episodes
        self.demo_list = list(self.demo_file["data"].keys())

        # subsample a selection of demonstrations if requested
        if num_traj > 0:
            random.seed(3141)  # ensure that the same set is sampled every time
            self.demo_list = random.sample(self.demo_list, num_traj)

        self.need_xml = need_xml
        self.demo_sampled = 0

        self.sample_method_dict = {
            "random": "_random_sample",
            "uniform": "_uniform_sample",
            "forward": "_forward_sample_open_loop",
            "reverse": "_reverse_sample_open_loop",
        }

        self.sampling_schemes = sampling_schemes
        self.scheme_ratios = np.asarray(scheme_ratios)

        # make sure the list of schemes is valid
        schemes = self.sample_method_dict.keys()
        assert np.all([(s in schemes) for s in self.sampling_schemes])

        # make sure the distribution is the correct size
        assert len(self.sampling_schemes) == len(self.scheme_ratios)

        # make sure the distribution lies in the probability simplex
        assert np.all(self.scheme_ratios > 0.0)
        assert sum(self.scheme_ratios) == 1.0

        # open loop configuration
        self.open_loop_increment_freq = open_loop_increment_freq
        self.open_loop_window_increment = open_loop_window_increment

        # keep track of window size
        self.open_loop_window_size = open_loop_initial_window_width

    def reset(self):
        """
        Logic for sampling a state from the demonstration and resetting
        the simulation to that state.

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        state = self.sample()
        if state is None:
            # None indicates that a normal env reset should occur
            return self.env.reset()
        else:
            if self.need_xml:
                # reset the simulation from the model if necessary
                state, xml = state
                self.env.reset_from_xml_string(xml)

            if isinstance(state, tuple):
                state = state[0]

            # force simulator state to one from the demo
            self.sim.set_state_from_flattened(state)
            self.sim.forward()

            return self.env._get_observation()

    def sample(self):
        """
        This is the core sampling method. Samples a state from a
        demonstration, in accordance with the configuration.

        Returns:
            None or np.array or 2-tuple: If np.array, is the state sampled from a demo file. If 2-tuple, additionally
                includes the model xml file
        """

        # chooses a sampling scheme randomly based on the mixing ratios
        seed = random.uniform(0, 1)
        ratio = np.cumsum(self.scheme_ratios)
        ratio = ratio > seed
        for i, v in enumerate(ratio):
            if v:
                break

        sample_method = getattr(self, self.sample_method_dict[self.sampling_schemes[i]])
        return sample_method()

    def _random_sample(self):
        """
        Sampling method.

        Return None to indicate that the state should be sampled directly
        from the environment.
        """
        return None

    def _uniform_sample(self):
        """
        Sampling method.

        First uniformly sample a demonstration from the set of demonstrations.
        Then uniformly sample a state from the selected demonstration.

        Returns:
            np.array or 2-tuple: If np.array, is the state sampled from a demo file. If 2-tuple, additionally
                includes the model xml file
        """

        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # select a flattened mujoco state uniformly from this episode
        states = self.demo_file["data/{}/states".format(ep_ind)][()]
        state = random.choice(states)

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = self.env.postprocess_model_xml(model_xml)
            return state, xml
        return state

    def _reverse_sample_open_loop(self):
        """
        Sampling method.

        Open loop reverse sampling from demonstrations. Starts by
        sampling from states near the end of the demonstrations.
        Increases the window backwards as the number of calls to
        this sampling method increases at a fixed rate.

        Returns:
            np.array or 2-tuple: If np.array, is the state sampled from a demo file. If 2-tuple, additionally
                includes the model xml file
        """

        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # sample uniformly in a window that grows backwards from the end of the demos
        states = self.demo_file["data/{}/states".format(ep_ind)][()]
        eps_len = states.shape[0]
        index = np.random.randint(max(eps_len - self.open_loop_window_size, 0), eps_len)
        state = states[index]

        # increase window size at a fixed frequency (open loop)
        self.demo_sampled += 1
        if self.demo_sampled >= self.open_loop_increment_freq:
            if self.open_loop_window_size < eps_len:
                self.open_loop_window_size += self.open_loop_window_increment
            self.demo_sampled = 0

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = self.env.postprocess_model_xml(model_xml)
            return state, xml

        return state

    def _forward_sample_open_loop(self):
        """
        Sampling method.

        Open loop forward sampling from demonstrations. Starts by
        sampling from states near the beginning of the demonstrations.
        Increases the window forwards as the number of calls to
        this sampling method increases at a fixed rate.

        Returns:
            np.array or 2-tuple: If np.array, is the state sampled from a demo file. If 2-tuple, additionally
                includes the model xml file
        """

        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # sample uniformly in a window that grows forwards from the beginning of the demos
        states = self.demo_file["data/{}/states".format(ep_ind)][()]
        eps_len = states.shape[0]
        index = np.random.randint(0, min(self.open_loop_window_size, eps_len))
        state = states[index]

        # increase window size at a fixed frequency (open loop)
        self.demo_sampled += 1
        if self.demo_sampled >= self.open_loop_increment_freq:
            if self.open_loop_window_size < eps_len:
                self.open_loop_window_size += self.open_loop_window_increment
            self.demo_sampled = 0

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = self.env.postprocess_model_xml(model_xml)
            return state, xml

        return state

    def _xml_for_episode_index(self, ep_ind):
        """
        Helper method to retrieve the corresponding model xml string
        for the passed episode index.

        Args:
            ep_ind (int): Episode index to pull from demo file

        Returns:
            str: model xml as a string
        """

        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = self.demo_file["data/{}".format(ep_ind)].attrs["model_file"]
        model_path = os.path.join(self.demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()
        return model_xml
