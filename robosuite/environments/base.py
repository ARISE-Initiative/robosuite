from collections import OrderedDict
from mujoco_py import MjSim, MjRenderContextOffscreen
from mujoco_py import load_model_from_xml

from robosuite.utils import SimulationError, XMLError, MujocoPyRenderer
import robosuite.utils.macros as macros
from robosuite.models.base import MujocoModel

import numpy as np

REGISTERED_ENVS = {}


def register_env(target_class):
    REGISTERED_ENVS[target_class.__name__] = target_class


def make(env_name, *args, **kwargs):
    """
    Instantiates a robosuite environment.

    This method attempts to mirror the equivalent functionality of gym.make in a somewhat sloppy way.

    Args:
        env_name (str): Name of the robosuite environment to initialize
        *args: Additional arguments to pass to the specific environment class initializer
        **kwargs: Additional arguments to pass to the specific environment class initializer

    Returns:
        MujocoEnv: Desired robosuite environment

    Raises:
        Exception: [Invalid environment name]
    """
    if env_name not in REGISTERED_ENVS:
        raise Exception(
            "Environment {} not found. Make sure it is a registered environment among: {}".format(
                env_name, ", ".join(REGISTERED_ENVS)
            )
        )
    return REGISTERED_ENVS[env_name](*args, **kwargs)


class EnvMeta(type):
    """Metaclass for registering environments"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["MujocoEnv", "RobotEnv", "ManipulationEnv", "SingleArmEnv", "TwoArmEnv"]

        if cls.__name__ not in _unregistered_envs:
            register_env(cls)
        return cls


class MujocoEnv(metaclass=EnvMeta):
    """
    Initializes a Mujoco Environment.

    Args:
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering.

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes
            in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes
            in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive
            in every simulated second. This sets the amount of simulation time
            that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

    Raises:
        ValueError: [Invalid renderer selection]
    """

    def __init__(
        self,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True
    ):
        # First, verify that both the on- and off-screen renderers are not being used simultaneously
        if has_renderer is True and has_offscreen_renderer is True:
            raise ValueError("the onscreen and offscreen renderers cannot be used simultaneously.")

        # Rendering-specific attributes
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.render_camera = render_camera
        self.render_collision_mesh = render_collision_mesh
        self.render_visual_mesh = render_visual_mesh
        self.render_gpu_device_id = render_gpu_device_id
        self.viewer = None

        # Simulation-specific attributes
        self._observables = {}                      # Maps observable names to observable objects
        self._obs_cache = {}                        # Maps observable names to pre-/partially-computed observable values
        self.control_freq = control_freq
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.hard_reset = hard_reset
        self._model_postprocessor = None            # Function to post-process model after load_model() call
        self.model = None
        self.cur_time = None
        self.model_timestep = None
        self.control_timestep = None
        self.deterministic_reset = False            # Whether to add randomized resetting of objects / robot joints

        # Load the model
        self._load_model()

        # Post-process model
        self._postprocess_model()

        # Initialize the simulation
        self._initialize_sim()

        # Run all further internal (re-)initialization required
        self._reset_internal()

        # Load observables
        self._observables = self._setup_observables()

    def initialize_time(self, control_freq):
        """
        Initializes the time constants used for simulation.

        Args:
            control_freq (float): Hz rate to run control loop at within the simulation
        """
        self.cur_time = 0
        self.model_timestep = macros.SIMULATION_TIMESTEP
        if self.model_timestep <= 0:
            raise ValueError("Invalid simulation timestep defined!")
        self.control_freq = control_freq
        if control_freq <= 0:
            raise SimulationError("Control frequency {} is invalid".format(control_freq))
        self.control_timestep = 1. / control_freq

    def set_model_postprocessor(self, postprocessor):
        """
        Sets the post-processor function that self.model will be passed to after load_model() is called during resets.

        Args:
            postprocessor (None or function): If set, postprocessing method should take in a Task-based instance and
                return no arguments.
        """
        self._model_postprocessor = postprocessor

    def _load_model(self):
        """Loads an xml model, puts it in self.model"""
        pass

    def _postprocess_model(self):
        """
        Post-processes model after load_model() call. Useful for external objects (e.g.: wrappers) to
        be able to modify the sim model before it is actually loaded into the simulation
        """
        if self._model_postprocessor is not None:
            self._model_postprocessor(self.model)

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        pass

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        return OrderedDict()

    def _initialize_sim(self, xml_string=None):
        """
        Creates a MjSim object and stores it in self.sim. If @xml_string is specified, the MjSim object will be created
        from the specified xml_string. Else, it will pull from self.model to instantiate the simulation

        Args:
            xml_string (str): If specified, creates MjSim object from this filepath
        """
        # if we have an xml string, use that to create the sim. Otherwise, use the local model
        self.mjpy_model = load_model_from_xml(xml_string) if xml_string else self.model.get_model(mode="mujoco_py")

        # Create the simulation instance and run a single step to make sure changes have propagated through sim state
        self.sim = MjSim(self.mjpy_model)
        self.sim.forward()

        # Setup sim time based on control frequency
        self.initialize_time(self.control_freq)

    def reset(self):
        """
        Resets simulation.

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        # TODO(yukez): investigate black screen of death
        # Use hard reset if requested
        if self.hard_reset and not self.deterministic_reset:
            self._destroy_viewer()
            self._load_model()
            self._postprocess_model()
            self._initialize_sim()
        # Else, we only reset the sim internally
        else:
            self.sim.reset()
        # Reset necessary robosuite-centric variables
        self._reset_internal()
        self.sim.forward()
        # Setup observables, reloading if
        self._obs_cache = {}
        if self.hard_reset:
            # If we're using hard reset, must re-update sensor object references
            _observables = self._setup_observables()
            for obs_name, obs in _observables.items():
                self.modify_observable(observable_name=obs_name, attribute="sensor", modifier=obs._sensor)
        # Make sure that all sites are toggled OFF by default
        self.visualize(vis_settings={vis: False for vis in self._visualizations})
        # Return new observations
        return self._get_observations(force_update=True)

    def _reset_internal(self):
        """Resets simulation internal configurations."""

        # create visualization screen or renderer
        if self.has_renderer and self.viewer is None:
            self.viewer = MujocoPyRenderer(self.sim)
            self.viewer.viewer.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
            self.viewer.viewer.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)

            # hiding the overlay speeds up rendering significantly
            self.viewer.viewer._hide_overlay = True

            # make sure mujoco-py doesn't block rendering frames
            # (see https://github.com/StanfordVL/robosuite/issues/39)
            self.viewer.viewer._render_every_frame = True

            # Set the camera angle for viewing
            if self.render_camera is not None:
                self.viewer.set_camera(camera_id=self.sim.model.camera_name2id(self.render_camera))

        elif self.has_offscreen_renderer:
            if self.sim._render_context_offscreen is None:
                render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)
                self.sim.add_render_context(render_context)
            self.sim._render_context_offscreen.vopt.geomgroup[0] = (1 if self.render_collision_mesh else 0)
            self.sim._render_context_offscreen.vopt.geomgroup[1] = (1 if self.render_visual_mesh else 0)

        # additional housekeeping
        self.sim_state_initial = self.sim.get_state()
        self._setup_references()
        self.cur_time = 0
        self.timestep = 0
        self.done = False

        # Empty observation cache and reset all observables
        self._obs_cache = {}
        for observable in self._observables.values():
            observable.reset()

    def _update_observables(self, force=False):
        """
        Updates all observables in this environment

        Args:
            force (bool): If True, will force all the observables to update their internal values to the newest
                value. This is useful if, e.g., you want to grab observations when directly setting simulation states
                without actually stepping the simulation.
        """
        for observable in self._observables.values():
            observable.update(timestep=self.model_timestep, obs_cache=self._obs_cache, force=force)

    def _get_observations(self, force_update=False):
        """
        Grabs observations from the environment.

        Args:
            force_update (bool): If True, will force all the observables to update their internal values to the newest
                value. This is useful if, e.g., you want to grab observations when directly setting simulation states
                without actually stepping the simulation.

        Returns:
            OrderedDict: OrderedDict containing observations [(name_string, np.array), ...]

        """
        observations = OrderedDict()
        obs_by_modality = OrderedDict()

        # Force an update if requested
        if force_update:
            self._update_observables(force=True)

        # Loop through all observables and grab their current observation
        for obs_name, observable in self._observables.items():
            if observable.is_enabled() and observable.is_active():
                obs = observable.obs
                observations[obs_name] = obs
                modality = observable.modality + "-state"
                if modality not in obs_by_modality:
                    obs_by_modality[modality] = []
                # Make sure all observations are numpy arrays so we can concatenate them
                array_obs = [obs] if type(obs) in {int, float} or not obs.shape else obs
                obs_by_modality[modality].append(np.array(array_obs))

        # Add in modality observations
        for modality, obs in obs_by_modality.items():
            # To save memory, we only concatenate the image observations if explicitly requested
            if modality == "image-state" and not macros.CONCATENATE_IMAGES:
                continue
            observations[modality] = np.concatenate(obs, axis=-1)

        return observations

    def step(self, action):
        """
        Takes a step in simulation with control command @action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information

        Raises:
            ValueError: [Steps past episode termination]

        """
        if self.done:
            raise ValueError("executing action in terminated episode")

        self.timestep += 1

        # Since the env.step frequency is slower than the mjsim timestep frequency, the internal controller will output
        # multiple torque commands in between new high level action commands. Therefore, we need to denote via
        # 'policy_step' whether the current step we're taking is simply an internal update of the controller,
        # or an actual policy update
        policy_step = True

        # Loop through the simulation at the model timestep rate until we're ready to take the next policy step
        # (as defined by the control frequency specified at the environment level)
        for i in range(int(self.control_timestep / self.model_timestep)):
            self.sim.forward()
            self._pre_action(action, policy_step)
            self.sim.step()
            self._update_observables()
            policy_step = False

        # Note: this is done all at once to avoid floating point inaccuracies
        self.cur_time += self.control_timestep

        reward, done, info = self._post_action(action)
        return self._get_observations(), reward, done, info

    def _pre_action(self, action, policy_step=False):
        """
        Do any preprocessing before taking an action.

        Args:
            action (np.array): Action to execute within the environment
            policy_step (bool): Whether this current loop is an actual policy step or internal sim update step
        """
        self.sim.data.ctrl[:] = action

    def _post_action(self, action):
        """
        Do any housekeeping after taking an action.

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:

                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) empty dict to be filled with information by subclassed method

        """
        reward = self.reward(action)

        # done if number of elapsed timesteps is greater than horizon
        self.done = (self.timestep >= self.horizon) and not self.ignore_done

        return reward, self.done, {}

    def reward(self, action):
        """
        Reward should be a function of state and action

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            float: Reward from environment
        """
        raise NotImplementedError

    def render(self):
        """
        Renders to an on-screen window.
        """
        self.viewer.render()

    def observation_spec(self):
        """
        Returns an observation as observation specification.

        An alternative design is to return an OrderedDict where the keys
        are the observation names and the values are the shapes of observations.
        We leave this alternative implementation commented out, as we find the
        current design is easier to use in practice.

        Returns:
            OrderedDict: Observations from the environment
        """
        observation = self._get_observations()
        return observation

    def clear_objects(self, object_names):
        """
        Clears objects with the name @object_names out of the task space. This is useful
        for supporting task modes with single types of objects, as in
        @self.single_object_mode without changing the model definition.

        Args:
            object_names (str or list of str): Name of object(s) to remove from the task workspace
        """
        object_names = {object_names} if type(object_names) is str else set(object_names)
        for obj in self.model.mujoco_objects:
            if obj.name in object_names:
                self.sim.data.set_joint_qpos(obj.joints[0], np.array((10, 10, 10, 1, 0, 0, 0)))

    def visualize(self, vis_settings):
        """
        Do any needed visualization here

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "env" keyword as well as any other relevant
                options specified.
        """
        # Set visuals for environment objects
        for obj in self.model.mujoco_objects:
            obj.set_sites_visibility(sim=self.sim, visible=vis_settings["env"])

    def reset_from_xml_string(self, xml_string):
        """
        Reloads the environment from an XML description of the environment.

        Args:
            xml_string (str): Filepath to the xml file that will be loaded directly into the sim
        """

        # if there is an active viewer window, destroy it
        self.close()

        # Since we are reloading from an xml_string, we are deterministically resetting
        self.deterministic_reset = True

        # initialize sim from xml
        self._initialize_sim(xml_string=xml_string)

        # Now reset as normal
        self.reset()

        # Turn off deterministic reset
        self.deterministic_reset = False

    def check_contact(self, geoms_1, geoms_2=None):
        """
        Finds contact between two geom groups.

        Args:
            geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
                a MujocoModel is specified, the geoms checked will be its contact_geoms
            geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
                If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
                any collision with @geoms_1 to any other geom in the environment

        Returns:
            bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
        """
        # Check if either geoms_1 or geoms_2 is a string, convert to list if so
        if type(geoms_1) is str:
            geoms_1 = [geoms_1]
        elif isinstance(geoms_1, MujocoModel):
            geoms_1 = geoms_1.contact_geoms
        if type(geoms_2) is str:
            geoms_2 = [geoms_2]
        elif isinstance(geoms_2, MujocoModel):
            geoms_2 = geoms_2.contact_geoms
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # check contact geom in geoms
            c1_in_g1 = self.sim.model.geom_id2name(contact.geom1) in geoms_1
            c2_in_g2 = self.sim.model.geom_id2name(contact.geom2) in geoms_2 if geoms_2 is not None else True
            # check contact geom in geoms (flipped)
            c2_in_g1 = self.sim.model.geom_id2name(contact.geom2) in geoms_1
            c1_in_g2 = self.sim.model.geom_id2name(contact.geom1) in geoms_2 if geoms_2 is not None else True
            if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
                return True
        return False

    def get_contacts(self, model):
        """
        Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
        geom names currently in contact with that model (excluding the geoms that are part of the model itself).

        Args:
            model (MujocoModel): Model to check contacts for.

        Returns:
            set: Unique geoms that are actively in contact with this model.

        Raises:
            AssertionError: [Invalid input type]
        """
        # Make sure model is MujocoModel type
        assert isinstance(model, MujocoModel), \
            "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
        contact_set = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # check contact geom in geoms; add to contact set if match is found
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
            if g1 in model.contact_geoms and g2 not in model.contact_geoms:
                contact_set.add(g2)
            elif g2 in model.contact_geoms and g1 not in model.contact_geoms:
                contact_set.add(g1)
        return contact_set

    def add_observable(self, observable):
        """
        Adds an observable to this environment.

        Args:
            observable (Observable): Observable instance.
        """
        assert observable.name not in self._observables,\
            "Observable name {} is already associated with an existing observable! Use modify_observable(...) " \
            "to modify a pre-existing observable.".format(observable.name)
        self._observables[observable.name] = observable

    def modify_observable(self, observable_name, attribute, modifier):
        """
        Modifies observable with associated name @observable_name, replacing the given @attribute with @modifier.

        Args:
             observable_name (str): Observable to modify
             attribute (str): Observable attribute to modify.
                Options are {`'sensor'`, `'corrupter'`,`'filter'`,  `'delayer'`, `'sampling_rate'`,
                `'enabled'`, `'active'`}
             modifier (any): New function / value to replace with for observable. If a function, new signature should
                match the function being replaced.
        """
        # Find the observable
        assert observable_name in self._observables, "No valid observable with name {} found. Options are: {}".\
            format(observable_name, self.observation_names)
        obs = self._observables[observable_name]
        # replace attribute accordingly
        if attribute == "sensor":
            obs.set_sensor(modifier)
        elif attribute == "corrupter":
            obs.set_corrupter(modifier)
        elif attribute == "filter":
            obs.set_filter(modifier)
        elif attribute == "delayer":
            obs.set_delayer(modifier)
        elif attribute == "sampling_rate":
            obs.set_sampling_rate(modifier)
        elif attribute == "enabled":
            obs.set_enabled(modifier)
        elif attribute == "active":
            obs.set_active(modifier)
        else:
            # Invalid attribute specified
            raise ValueError("Invalid observable attribute specified. Requested: {}, valid options are {}".
                             format(attribute, {"sensor", "corrupter", "filter", "delayer",
                                                "sampling_rate", "enabled", "active"}))

    def _check_success(self):
        """
        Checks if the task has been completed. Should be implemented by subclasses

        Returns:
            bool: True if the task has been completed
        """
        raise NotImplementedError

    def _destroy_viewer(self):
        """
        Destroys the current mujoco renderer instance if it exists
        """
        # if there is an active viewer window, destroy it
        if self.viewer is not None:
            self.viewer.close()  # change this to viewer.finish()?
            self.viewer = None

    def close(self):
        """Do any cleanup necessary here."""
        self._destroy_viewer()

    @property
    def observation_modalities(self):
        """
        Modalities for this environment's observations

        Returns:
            set: All observation modalities
        """
        return set([observable.modality for observable in self._observables.values()])

    @property
    def observation_names(self):
        """
        Grabs all names for this environment's observables

        Returns:
            set: All observation names
        """
        return set(self._observables.keys())

    @property
    def enabled_observables(self):
        """
        Grabs all names of enabled observables for this environment. An observable is considered enabled if its values
        are being continually computed / updated at each simulation timestep.

        Returns:
            set: All enabled observation names
        """
        return set([name for name, observable in self._observables.items() if observable.is_enabled()])

    @property
    def active_observables(self):
        """
        Grabs all names of active observables for this environment. An observable is considered active if its value is
        being returned in the observation dict from _get_observations() call or from the step() call (assuming this
        observable is enabled).

        Returns:
            set: All active observation names
        """
        return set([name for name, observable in self._observables.items() if observable.is_active()])

    @property
    def _visualizations(self):
        """
        Visualization keywords for this environment

        Returns:
            set: All components that can be individually visualized for this environment
        """
        return {"env"}

    @property
    def action_spec(self):
        """
        Action specification should be implemented in subclasses.

        Action space is represented by a tuple of (low, high), which are two numpy
        vectors that specify the min/max action limits per dimension.
        """
        raise NotImplementedError

    @property
    def action_dim(self):
        """
        Size of the action space

        Returns:
            int: Action space dimension
        """
        raise NotImplementedError
