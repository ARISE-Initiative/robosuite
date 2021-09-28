import numpy as np
import robosuite as suite

try:
    import igibson
except ImportError:
    raise Exception("Use `pip install igibson` to install iGibson renderer. "
                    "Follow instructions here: http://svl.stanford.edu/igibson/docs/installation.html "
                    "to download the iG assets and dataset."
                    )

from robosuite.wrappers import Wrapper
from robosuite.utils import transform_utils as T
from robosuite.renderers.igibson.parser import Parser

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, Instance, Robot
from igibson.utils.mesh_util import xyzw2wxyz, quat2rotmat, ortho
from igibson.render.viewer import Viewer
from robosuite.utils.observables import sensor
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
import robosuite.utils.macros as macros
from robosuite.renderers.igibson.igibson_utils import TensorObservable, adjust_convention
from robosuite.renderers import load_renderer_config
from robosuite.renderers.base import Renderer
from collections import OrderedDict
from igibson.utils.constants import MAX_CLASS_COUNT
import os


from robosuite.utils import macros


try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


AVAILABLE_MODALITIES = {'rgb', 'seg', 'normal', '3d'}
AVAILABLE_MODES = {'gui', 'headless'}

def check_modes(modes):
    for m in modes:
        if m not in AVAILABLE_MODALITIES:
            raise ValueError(f'`vision_modalities` can only be from the {AVAILABLE_MODALITIES}, got `{m}`')

def check_render_mode(render_mode):
    if render_mode not in AVAILABLE_MODES:
        raise ValueError(f'`render_mode` can only be from the {AVAILABLE_MODES}, got {render_mode}')

def check_camera_obs(use_camera_obs):
    if use_camera_obs:
        raise ValueError("Cannot set `use_camera_obs` in the environment, instead set 'camera_obs' flag in iGibson initialization.")

def check_render2tensor(render2tensor, render_mode):
    if render2tensor and not HAS_TORCH:
        raise AssertionError("`render2tensor` requires PyTorch to be installed.") 

    if render_mode == 'gui' and render2tensor:
        raise ValueError('render2tensor can only be set to true in `headless` mode. ')

def check_segmentation_type(segmentation_type):

    assert len(segmentation_type) == 1, \
            ValueError('Only one segmentation type can be used for all cameras. Set `camera_segmentations` to be' \
            'either `class`, `instance` or `element`')
    if segmentation_type[0] is not None:
        assert len(segmentation_type[0]) == 1, \
                ValueError('Only one sensor per camera can be used.' \
                'Set `camera_segmentations` to be either `class`, `instance` or `element`')

    if segmentation_type[0] is not None:
        return segmentation_type[0][0] 


class iGibsonRenderer(Renderer):
    def __init__(self,
                 env,
                 render_mode='gui',
                 width=1280,
                 height=720,
                 enable_pbr=True,
                 enable_shadow=True,
                 msaa=True,
                 render2tensor=False,
                 optimized=False,
                 light_dimming_factor=1.0,
                 device_idx=0,
                 camera_obs=False,
                 vision_modalities=['rgb']):
        """
        Initializes the iGibson renderer.

        Args:
            env (MujocoEnv instance): The environment to wrap.

            render_mode (string, optional): The rendering mode. Could be set either to `gui`
                or `headless`. If set to `gui` the viewer window
                will open up.

            width (int, optional): Width of the rendered image. Defaults to 1280.
                All the cameras will be rendered at same resolution.
                This width will be given preference over `camera_widths`
                specified at the initialization of the environment.

            height (int, optional): Height of the rendered image. Defaults to 720.
                All the cameras will be rendered at same resolution.
                This width will be given preference over `camera_heights`
                specified at the initialization of the environment.

            enable_pbr (bool, optional): Whether to enable physically-based rendering. Defaults to True.

            enable_shadow (bool, optional): Whether to render shadows. Defaults to True.

            msaa (bool, optional): Whether to use multisample anti-aliasing. Defaults to True.
                If using `seg` in modes, please turn off msaa because of
                segmentation artefacts.

            render2tensor (bool, optional): Whether to render images as torch cuda tensors. Defaults to False.
                Images are rendered as numpy arrays by default. If `render2tensor`
                is True, it will render as torch tensors on gpu.

            optimized (bool, optional): Whether to use optimized renderer, which will be faster. Defaults to False.
                Turning this will speed up the rendering at the cost of slight loss in
                visuals of the renderered image.

            light_dimming_factor (float, optional): Controls the intensity of light. Defaults to 1.0.
                Increasing this will increase the intensity of the light.

            device_idx (int, optional): GPU index to render on. Defaults to 0.

            camera_obs (bool, optional): Whether to use camera observation or not. The orientation of the image
                array will follow `macros.IMAGE_CONVENTION` which is by default set to
                `opengl`.

            vision_modalities (list, optional): Types of modality to render. Defaults to ['rgb'].
                Available modalities are `['rgb', 'seg', '3d', 'normal']`
                If `camd` is set to True while initializing the environment
                `3d` mode is automatically added in the list of modes.
                If `camera_segmentation` is set, that camera segmentation sensor is used.

        
        """
        super().__init__(env)

        check_modes(vision_modalities)
        check_render_mode(render_mode)
        # environment use camera obs must be false and iG camera_obs must be true.
        # This makes robosuite not initialize mujoco sensors
        # which was behaving strangely.
        check_camera_obs(self.env.use_camera_obs)
        check_render2tensor(render2tensor, render_mode)
        segmentation_type = check_segmentation_type(self.env.camera_segmentations)
            


        self.mode = render_mode
        self.env = env
        self.render2tensor = render2tensor
        self.width = width
        self.height = height
        self.enable_pbr = enable_pbr
        self.enable_shadow = enable_shadow
        self.msaa = msaa
        self.optimized = optimized
        self.light_dimming_factor = light_dimming_factor
        self.device_idx = device_idx
        self.modes = vision_modalities
        self.camera_obs = camera_obs
        self.segmentation_type = segmentation_type
        
        # setup references so that segmentation mappings are created.
        env._setup_references()

        # if camd is set in env, add the depth mode in iG
        if True in self.env.camera_depths and '3d' not in self.modes:
            self.modes += ['3d']

        if self.segmentation_type is not None and 'seg' not in self.modes:
            self.modes += ['seg']
        
        self.mrs = MeshRendererSettings(msaa=msaa, 
                                        env_texture_filename=os.path.join(suite.models.assets_root,"light_map","photo_studio_01_2k.hdr"),
                                        env_texture_filename2=os.path.join(suite.models.assets_root,"light_map","photo_studio_01_2k.hdr"),
                                        env_texture_filename3=os.path.join(suite.models.assets_root,"light_map","photo_studio_01_2k.hdr"),
                                        enable_pbr=enable_pbr, 
                                        enable_shadow=enable_shadow,
                                        optimized=optimized,
                                        light_dimming_factor=light_dimming_factor,
                                        is_robosuite=True)

        if render2tensor:
            self.renderer_class = MeshRendererG2G
        else:
            self.renderer_class = MeshRenderer

        self.renderer = self.renderer_class(width=width,
                                            height=height,
                                            vertical_fov=45,
                                            device_idx=device_idx,
                                            rendering_settings=self.mrs)
        
        # load all the textures, materials, geoms, cameras
        self._load()

        # set camera for renderer
        self._switch_camera(camera_name=self.env.render_camera)

        # add viewer
        self.viewer = self._add_viewer(initial_pos=self.camera_position, 
                                       initial_view_direction=self.view_direction)      

    def _setup_observables(self):
        observables = self.env._setup_observables()
        
        # Loop through cameras and change the sensor
        if self.camera_obs:
            
            sensors = []
            names = []

            for (cam_name, cam_d) in \
                zip(self.env.camera_names, self.env.camera_depths):

                # Add cameras associated to our arrays
                cam_sensors, cam_sensor_names = self._create_camera_sensors(
                    cam_name, cam_w=self.width, cam_h=self.height, cam_d=cam_d, modality="image")
                sensors += cam_sensors
                names += cam_sensor_names    

            # Create observables for these cameras
            for name, s in zip(names, sensors):
                observables[name] = TensorObservable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.env.control_freq,
                )       

        return observables

    def _create_camera_sensors(self, cam_name, cam_w, cam_h, cam_d, modality="image"):
        """
        Helper function to create sensors for a given camera. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            cam_name (str): Name of camera to create sensors for
            cam_w (int): Width of camera
            cam_h (int): Height of camera
            cam_d (bool): Whether to create a depth sensor as well
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given camera
                names (list): array of corresponding observable names
        """
        # Make sure we get correct convention
        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]

        # Create sensor information
        sensors = []
        names = []

        # Add camera observables to the dict
        rgb_sensor_name = f"{cam_name}_image"
        depth_sensor_name = f"{cam_name}_depth"
        seg_sensor_name = f"{cam_name}_seg"
        normal_sensor_name = f"{cam_name}_normal"

        @sensor(modality=modality)
        def camera_rgb(obs_cache):
            
            # Switch to correct camera
            if self is not None and self.camera_name != cam_name:
                self._switch_camera(cam_name)

            rendered_imgs = self.renderer.render(modes=self.modes)
            rendered_mapping = {k: val for k, val in zip(self.modes, rendered_imgs)}

            # in np array image received is of correct orientation
            # in torch tensor image is flipped upside down
            # adjusting the np image in a way so that return statement stays same
            # flipping torch tensor when opencv coordinate system is required.
            # using torch.flipud because negative strides do not work in torch.

            if 'rgb' in self.modes:
                img = rendered_mapping['rgb'][:,:,:3]
                if isinstance(img, np.ndarray):
                    # np array is in range 0-1
                    img = (img * 255).astype(np.uint8)
                    img = adjust_convention(img, convention)
                elif convention == -1:
                    img = torch.flipud(img)
            else:
                img = np.zeros((cam_h, cam_w), np.uint8)

            if 'seg' in self.modes:
                # 0th channel contains segmentation
                seg_map = rendered_mapping['seg'][:,:,0]
                if isinstance(seg_map, np.ndarray):
                    # np array is in range 0-1
                    # round function is important here otherwise trippy looking segmaps
                    seg_map = (seg_map * MAX_CLASS_COUNT).round().astype(np.int64)
                    # flip the image upside down if required
                    seg_map = adjust_convention(seg_map, convention)
                elif convention == -1:
                    # flip in Y direction if torch tensor
                    seg_map = torch.flipud(seg_map)  
                obs_cache[seg_sensor_name] = seg_map

            if '3d' in self.modes:
                # 2nd channel contains correct depth map
                depth_map = rendered_mapping['3d'][:,:,2]
                if isinstance(depth_map, np.ndarray):
                    depth_map = adjust_convention(depth_map, convention)
                elif convention == -1:
                    # flip in Y direction if torch tensor
                    depth_map = torch.flipud(depth_map)
                
                obs_cache[depth_sensor_name] = depth_map

            if 'normal' in self.modes:
                normal_map = rendered_mapping['normal'][:,:,:3]
                if isinstance(normal_map, np.ndarray):
                    # np array is in range 0-1
                    normal_map = (normal_map * 255).astype(np.uint8)
                    normal_map = adjust_convention(normal_map, convention)
                elif convention == -1:
                    normal_map = torch.flipud(normal_map)                                                  
                obs_cache[normal_sensor_name] = normal_map
            
            return img

        sensors.append(camera_rgb)
        names.append(rgb_sensor_name)

        # Below modes are only applicable for iG renderer.
        if 'seg' in self.modes:
            @sensor(modality=modality)
            def camera_seg(obs_cache):
                return obs_cache[seg_sensor_name] if seg_sensor_name in obs_cache else \
                    np.zeros((cam_h, cam_w, 1))

            sensors.append(camera_seg)
            names.append(seg_sensor_name)

        if '3d' in self.modes or cam_d:
            @sensor(modality=modality)
            def camera_depth(obs_cache):
                return obs_cache[depth_sensor_name] if depth_sensor_name in obs_cache else \
                    np.zeros((cam_h, cam_w, 1))  

            sensors.append(camera_depth)
            names.append(depth_sensor_name)                     

        if 'normal' in self.modes:
            @sensor(modality=modality)
            def camera_normal(obs_cache):
                return obs_cache[normal_sensor_name] if normal_sensor_name in obs_cache else \
                    np.zeros((cam_h, cam_w, 1))  

            sensors.append(camera_normal)
            names.append(normal_sensor_name)               

        return sensors, names


    def _switch_camera(self, camera_name):
        """
        Change renderer camera to one of the available cameras of the environment
        using its name.

        Args:
            camera_name (string): name of the camera
        """
        self.camera_name = camera_name
        self.camera_position, self.view_direction, fov = self._get_camera_pose(camera_name)
        self.renderer.set_camera(self.camera_position,
                                 self.camera_position + self.view_direction,
                                 [0, 0, 1])
        self.renderer.lightP = ortho(-3, 3, -3, 3, -10, 25.0)
        self.renderer.set_fov(fov)

    def _get_camera_pose(self, camera_name):
        """
        Get position and orientation of the camera given the name

        Args:
            camera_name (string): name of the camera
        """
        for instance in self.renderer.instances:
            if isinstance(instance, Robot):
                for cam in instance.robot.cameras:
                    if cam.camera_name == camera_name:
                        camera_pose = cam.get_pose()
                        camera_pos = camera_pose[:3]
                        camera_ori = camera_pose[3:]
                        camera_ori_mat = quat2rotmat([camera_ori[-1], camera_ori[0], camera_ori[1], camera_ori[2]])[:3, :3]
                        # Mujoco camera points in -z
                        camera_view_dir = camera_ori_mat.dot(np.array([0, 0, -1])) 
                        return camera_pos, camera_view_dir, cam.fov

        raise Exception(f"Camera {self.env.render_camera} not present")

    def _set_camera_pos_quat(self, pos, quat):
        """Set renderer position and quaternion.

        Args:
            pos ([list]): pos array of the camera.
            quat ([list]): quat array of the camera.
        """
        # to wxyz
        mat = quat2rotmat([quat[-1], quat[0], quat[1], quat[2]])[:3, :3]
        mat = mat.dot(np.array([0, 0, 1]))
        self.renderer.set_camera(pos, [0,0,1], [0,0,1])


    def _add_viewer(self, 
                 initial_pos = [0,0,1], 
                 initial_view_direction = [1,0,0], 
                 initial_up = [0,0,1]):
        """
        Initialize iGibson viewer.

        Args:
            initial_pos (list, optional): Initial position of the viewer camera. Defaults to [0,0,1].
            initial_view_direction (list, optional): Initial direction of the camera. Defaults to [1,0,0].
            initial_up (list, optional): Vertical direction. Defaults to [0,0,1].
        """
        if self.mode == 'gui' and not self.render2tensor:
            # in case of robosuite viewer, we open only one window.
            # Later use the numpad to activate additional cameras            
            return Viewer(initial_pos = initial_pos,
                                initial_view_direction=initial_view_direction,
                                initial_up=initial_up,
                                renderer=self.renderer)

    def _load(self):        
        parser = Parser(self.renderer, self.env, self.segmentation_type)
        parser.parse_cameras()        
        parser.parse_textures()
        parser.parse_materials()
        parser.parse_geometries()
        self.visual_objects = parser.visual_objects
        self.max_elements = parser.max_elements
        self.max_instances = parser.max_instances
        self.max_classes = parser.max_classes

    def render(self):
        """
        Update the viewer of the iGibson renderer.
        Call to render is made inside viewer's update.
        """
        if self.mode == 'gui' and self.viewer is not None:
            self.viewer.update()

    def update(self):
        """Updates the states for the wrapper given a certain action
        Args:
            action (np-array): the action the robot should take
        """
        for instance in self.renderer.instances:
            if instance.dynamic:
                self._update_position(instance, self.env)

    @staticmethod
    def _update_position(instance, env):
        """
        Update position for an object or a robot in renderer.

        :param instance: Instance in the renderer
        """
        if isinstance(instance, Instance):
            if instance.parent_body != 'worldbody':
                pos_body_in_world = env.sim.data.get_body_xpos(instance.parent_body)
                rot_body_in_world = env.sim.data.get_body_xmat(instance.parent_body).reshape((3, 3))
                pose_body_in_world = T.make_pose(pos_body_in_world, rot_body_in_world)
                pose_geom_in_world = pose_body_in_world
                pos, orn = T.mat2pose(pose_geom_in_world) #xyzw
            else:
                pos = [0,0,0]
                orn = [0,0,0,1] #xyzw

            instance.set_position(pos)
            instance.set_rotation(quat2rotmat(xyzw2wxyz(orn)))

    def reset(self):
        self.renderer.release()
        self.__init__(env=self.env,
                 render_mode=self.mode,
                 width=self.width,
                 height=self.height,
                 vision_modalities=self.modes,
                 enable_pbr=self.enable_pbr,
                 enable_shadow=self.enable_shadow,
                 msaa=self.msaa,
                 render2tensor=self.render2tensor,
                 optimized=self.optimized,
                 light_dimming_factor=self.light_dimming_factor,
                 camera_obs=self.camera_obs,
                 device_idx=self.device_idx)

        return self.env._get_observations()
                
    def close(self):
        """
        Releases the iGibson renderer.
        """        
        self.renderer.release()

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
            self.env._update_observables(force=True)

        # Loop through all observables and grab their current observation
        for obs_name, observable in self.env._observables.items():
            if observable.is_enabled() and observable.is_active():
                obs = observable.obs
                observations[obs_name] = obs
                modality = observable.modality + "-state"
                if modality not in obs_by_modality:
                    obs_by_modality[modality] = []
                # Make sure all observations are numpy arrays so we can concatenate them
                array_obs = [obs] if type(obs) in {int, float} or not obs.shape else obs
                if HAS_TORCH and isinstance(array_obs, torch.Tensor):
                    obs_by_modality[modality].append(array_obs)
                else:
                    obs_by_modality[modality].append(np.array(array_obs))

        # Add in modality observations
        for modality, obs in obs_by_modality.items():
            # To save memory, we only concatenate the image observations if explicitly requested
            if modality == "image-state" and not macros.CONCATENATE_IMAGES:
                continue
            if HAS_TORCH and isinstance(obs[0], torch.Tensor):
                observations[modality] = torch.cat(obs, axis=-1)
            else:
                observations[modality] = np.concatenate(obs, axis=-1)

        return observations

if __name__ == '__main__':

    # Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
    #                          PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal, 
    #                          PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    # Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e

    config = load_renderer_config('igibson')
    # config['optimized'] = True

    # config['vision_modalities'] = ['rgb', 'normal', '3d']
    # config['vision_modalities'] = ['seg']
    # config['camera_obs'] = True
    # config['render_mode'] = 'headless'
    # config['msaa'] = False          


    env = suite.make(
            "Door",
            robots = ["Jaco"],
            reward_shaping=True,
            has_renderer=False,           # no on-screen renderer
            has_offscreen_renderer=False, # no off-screen renderer
            ignore_done=True,
            use_object_obs=True,          # use object-centric feature
            use_camera_obs=False,
            render_camera='frontview',         
            control_freq=10,
            renderer="igibson",
            camera_segmentations='element',
            renderer_config=config,
            camera_names=['frontview']            
        )    

    env.reset()

    for i in range(10000):
        action = np.random.randn(8)
        obs, reward, done, _ = env.step(action)
        # print(obs.keys())        
        env.render()

    env.close()
    
    print('Done.')
