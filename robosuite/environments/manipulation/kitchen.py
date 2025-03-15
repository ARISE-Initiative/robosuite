from collections import OrderedDict
import numpy as np
from copy import deepcopy
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

import robosuite.utils.transform_utils as T
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, BoxObject

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements, add_material
from robosuite.utils.buffers import RingBuffer

from robosuite.models.objects import (
    BreadObject,
    MilkObject,
    AppleObject,
    PotObject,
    StoveObject,
    ButtonObject,
    CabinetObject,
    ServingRegionObject
)

class Kitchen(SingleArmEnv):
    """
    Kitchen Env: The task is: place plate on the stove, cook with different ingradients and place the plate on the serving region.
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            use_latch=False,
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            contact_threshold=2.0,
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (1.0, 0.8, 0.05)
        self.table_offset = (-0.2, 0, 0.90)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        # ee resets
        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)

        # Thresholds
        self.contact_threshold = contact_threshold

        # History observations
        self._history_force_torque = None
        self._recent_force_torque = None

        self.stoves = {}
        self.buttons = {}
        self.buttons_on = {}
        self.button_qpos_addrs = {}
        self.num_stoves = 0
        self.has_stove_turned_on = False

        self.food_items_to_id = {"milk": 0, "bread": 1, "apple": 2}
        self.food_items_names = ["Milk", "Bread", "Apple"]

        self.objects = []

        self.eef_bounds = np.array([
            [-0.28, -0.32, 0.90],
            [0.15, 0.32, 1.16]
        ])

        self.data_eef_bounds = np.array([
            [-0.26, -0.31, 0.90],
            [0.14, 0.31, 1.15]
        ])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the drawer is opened

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 0.25], proportional to the distance between drawer handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by drawer handled
              - Note that this component is only relevant if the environment is using the locked drawer version

        Note that a successfully completed task (drawer opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards

        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414,
                  0.49641484022140503, 0.6968992352485657]
        )

        # bread = CustomMaterial(
        #     texture="Bread",
        #     tex_name="bread",
        #     mat_name="MatBread",
        #     tex_attrib={"type": "cube"},
        #     mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        # )
        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4", "shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        tex_attrib = {
            "type": "cube"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="MatBlueWood",
            # tex_attrib=tex_attrib,
            # mat_attrib=mat_attrib,
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        self.stove_object_1 = StoveObject(
            name="Stove1",
            joints=None,
        )
        stove_body = self.stove_object_1.get_obj()
        stove_body.set("pos", array_to_string((0.23, 0.18, 0.02)))
        mujoco_arena.table_body.append(stove_body)

        self.button_object_1 = ButtonObject(
            name="Button1",
        )

        button_body = self.button_object_1.get_obj()
        button_body.set("quat", array_to_string((0., 0., 0., 1.)))
        button_body.set("pos", array_to_string((0.06, 0.20, 0.02)))
        mujoco_arena.table_body.append(button_body)

        self.serving_region = ServingRegionObject(
            name="ServingRegionRed"
        )
        serving_region_object = self.serving_region.get_obj()
        serving_region_object.set("pos", array_to_string((0.35, -0.02, 0.003)))
        mujoco_arena.table_body.append(serving_region_object)

        self.pot_object = PotObject(
            name="PotObject",
        )

        self.cabinet_object = CabinetObject(
            name="CabinetObject")
        cabinet_object = self.cabinet_object.get_obj()
        cabinet_object.set("quat", array_to_string((0., 0., 0., 1.)))
        cabinet_object.set("pos", array_to_string((0.15, -0.35, 0.03)))
        mujoco_arena.table_body.append(cabinet_object)

        for obj_body in [
            self.button_object_1,
            self.stove_object_1,
            self.serving_region,
            self.cabinet_object,
        ]:
            for material in [darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                 naming_prefix=obj_body.naming_prefix,
                                                                 custom_material=deepcopy(material))
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        # ingredient_size = [0.015, 0.025, 0.02]
        #
        # self.bread_ingredient = BoxObject(
        #     name="cube_bread",
        #     size_min=ingredient_size,
        #     size_max=ingredient_size,
        #     rgba=[1, 0, 0, 1],
        #     material=bread,
        #     density=500.,
        # )
        mujoco_objects = [self.pot_object]
        self.food_item_objects = []

        for obj_cls, obj_name in zip(
            (MilkObject, BreadObject, AppleObject),
            self.food_items_names,
        ):
            obj = obj_cls(name=obj_name)
            mujoco_objects.append(obj)
            self.food_item_objects.append(obj)

        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=mujoco_objects,
        )
        self.stoves = {1: self.stove_object_1,
                       # 2: self.stove_object_2
                       }

        self.num_stoves = len(self.stoves.keys())

        self.buttons = {1: self.button_object_1,
                        # 2: self.button_object_2,
                        }
        self.buttons_on = {1: False,
                           # 2: False
                           }

        self.objects = [
            *(mujoco_objects),
            self.stove_object_1,
            self.serving_region,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.button_object_1)
        self.model.merge_assets(self.stove_object_1)
        self.model.merge_assets(self.serving_region)
        self.model.merge_assets(self.cabinet_object)


    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-food",
                mujoco_objects=self.food_item_objects,
                x_range=[-0.15, 0.12],
                y_range=[-0.18, 0.12],
                rotation=(-np.pi / 6., np.pi / 6.),
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.02,
            ))

        # Create placement initializer
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-pot",
                mujoco_objects=self.pot_object,
                x_range=[0.18, 0.22],
                y_range=[-0.02, -0.02],
                rotation=(-0.1, 0.1),
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=-0.1,
            ))


    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.object_body_ids = dict()
        self.object_body_ids["stove_1"] = self.sim.model.body_name2id(self.stove_object_1.root_body)
        # self.object_body_ids["stove_2"] = self.sim.model.body_name2id(self.stove_object_2.root_body)

        self.pot_object_id = self.sim.model.body_name2id(self.pot_object.root_body)
        self.button_qpos_addrs.update({1: self.sim.model.get_joint_qpos_addr(self.button_object_1.joints[0])})
        # self.button_qpos_addrs.update({2: self.sim.model.get_joint_qpos_addr(self.button_object_2.joints[0])})

        self.serving_region_id = self.sim.model.body_name2id(self.serving_region.root_body)

        self.cabinet_qpos_addrs = self.sim.model.get_joint_qpos_addr(self.cabinet_object.joints[0])

        self.sim.data.set_joint_qpos(self.button_object_1.joints[0], np.array([-0.4]))

        self.obj_body_id = {}
        for obj in self.objects:
            self.obj_body_id[obj.name] = self.sim.model.body_name2id(obj.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        observables["robot0_joint_pos"]._active = True

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"
            sensors = []
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked drawer version with rotatable handle

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix
        modality = f"{pf}proprio"

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if \
                f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)

        sensors.append(world_pose_in_gripper)
        names.append("world_pose_in_gripper")

        for (i, obj) in enumerate(self.objects):
            obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj.name, modality="object")

            sensors += obj_sensors
            names += obj_sensor_names

        @sensor(modality=modality)
        def gripper_contact(obs_cache):
            return self._has_gripper_contact

        @sensor(modality=modality)
        def force_norm(obs_cache):
            return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias)

        sensors += [gripper_contact, force_norm]
        names += [f"{pf}contact", f"{pf}eef_force_norm"]

        for name, s in zip(names, sensors):
            if name == "world_pose_in_gripper":
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=True,
                    active=False,
                )
            else:
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []

    def _check_success(self):
        """
        Check if drawer has been opened.

        Returns:
            bool: True if drawer has been opened
        """
        return False
        # pot_pos = self.sim.data.body_xpos[self.pot_object_id]
        # serving_region_pos = self.sim.data.body_xpos[self.serving_region_id]
        # dist_serving_pot = serving_region_pos - pot_pos
        # pot_in_serving_region = np.abs(dist_serving_pot[0]) < 0.05 and np.abs(dist_serving_pot[1]) < 0.10 and np.abs(
        #     dist_serving_pot[2]) < 0.05
        # stove_turned_off = not self.buttons_on[1]
        # if not stove_turned_off:
        #     self.has_stove_turned_on = True
        # object_in_pot = self.check_contact(self.bread_ingredient, self.pot_object)
        # return pot_in_serving_region and stove_turned_off and object_in_pot and self.has_stove_turned_on

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drawer handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4:
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)

        self._recent_force_torque = []
        obs, reward, done, info = super().step(action)
        info["history_ft"] = np.clip(np.copy(self._history_force_torque.buf), a_min=None, a_max=2)
        info["recent_ft"] = np.array(self._recent_force_torque)
        done = self._check_success()
        return obs, reward, done, info

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

        self._history_force_torque.push(
            np.hstack((self.robots[0].ee_force - self.ee_force_bias, self.robots[0].ee_torque - self.ee_torque_bias)))
        self._recent_force_torque.append(
            np.hstack((self.robots[0].ee_force - self.ee_force_bias, self.robots[0].ee_torque - self.ee_torque_bias)))

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        # Check if stove is turned on or not
        self._post_process()

        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.robots[0].ee_force
            self.ee_torque_bias = self.robots[0].ee_torque

        return reward, done, info

    def _post_process(self):
        stoves_on = {1: False,
                     2: False}

        for i in range(1, self.num_stoves + 1):
            if self.buttons_on[i]:
                if self.sim.data.qpos[self.button_qpos_addrs[i]] < 0.0:
                    self.buttons_on[i] = False
            else:
                if self.sim.data.qpos[self.button_qpos_addrs[i]] >= 0.0:
                    self.buttons_on[i] = True

        for stove_num, stove_status in self.buttons_on.items():
            self.stoves[stove_num].set_sites_visibility(sim=self.sim, visible=stove_status)

    @property
    def _has_gripper_contact(self):
        """
        Determines whether the gripper is making contact with an object, as defined by the eef force surprassing
        a certain threshold defined by self.contact_threshold

        Returns:
            bool: True if contact is surpasses given threshold magnitude
        """
        # return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias) > self.contact_threshold

        return np.linalg.norm(self.robots[0].ee_force - self.ee_force_bias) > self.contact_threshold

    def get_state_vector(self, obs):
        return np.concatenate([obs["robot0_gripper_qpos"],
                               obs["robot0_eef_pos"],
                               obs["robot0_eef_quat"]])
