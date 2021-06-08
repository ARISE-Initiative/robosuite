from os import fwalk
import numpy as np
import robosuite as suite

from robosuite.wrappers import Wrapper
from robosuite.utils import transform_utils as T
from robosuite.renderers.igibson.parser import Parser

from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, Instance, Robot
from gibson2.utils.mesh_util import xyzw2wxyz, quat2rotmat, ortho
from gibson2.render.viewer import Viewer

AVAILABLE_MODALITIES = ['rgb', 'seg', 'normal', '3d']

def check_modes(modes):
    for m in modes:
        if m not in AVAILABLE_MODALITIES:
            raise Exception(f'`modes` can only be from the {AVAILABLE_MODALITIES}, got {m}')

class iGibsonWrapper(Wrapper):
    def __init__(self,
                 env,
                 width=1280,
                 height=720,
                 enable_pbr=True,
                 enable_shadow=True,
                 msaa=True,
                 render2tensor=False,
                 optimized=False,
                 light_dimming_factor=1.0,
                 device_idx=0,
                 modes=['rgb']):
        """
        Initializes the iGibson wrapper. Wrapping any MuJoCo environment in this 
        wrapper will use the iGibson wrapper for rendering.

        Args:
            env (MujocoEnv instance): The environment to wrap.

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

            modes (list, optional): Types of modality to render. Defaults to ['rgb'].
                                    Available modalities are `['rgb', 'seg', '3d', 'normal']`
                                    If `camd` is set to True while initializing the environment
                                    `3d` mode is automatically added in the list of modes.
        
        """
        super().__init__(env)

        check_modes(modes)        
        if not self.env.render_with_igibson:
            raise Exception("Set `render_with_igibson=True` while initializing the environment.")


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
        self.modes = modes

        # if camd is et in env, add the depth mode in iG
        if True in self.env.camera_depths and '3d' not in self.modes:
            self.modes += ['3d']

        if not self.env.has_renderer:
            self.mode = 'headless'
        else:
            self.mode = 'gui'
        
        self.mrs = MeshRendererSettings(msaa=msaa, 
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
        self._add_viewer(initial_pos=self.camera_position, 
                        initial_view_direction=self.view_direction)

        # set parameters which will be used inside rovobosuite
        # when use_camera_obs=True
        self.env.ig_renderer_params = {'renderer':self.renderer,
                                       'modes': modes,
                                       'ig_wrapper_obj':self,
                                       }

        # Setup observables again after setting the iG parameters
        self.env._observables = self.env._setup_observables()

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
        self.renderer.lightP = ortho(-2, 2, -2, 2, -10, 25.0)
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

        raise Exception("Camera {self.env.render_camera} not present")


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
            self.viewer = Viewer(initial_pos = initial_pos,
                                initial_view_direction=initial_view_direction,
                                initial_up=initial_up,
                                renderer=self.renderer)
                        
        else:
            self.viewer = None

    def _load(self):        
        parser = Parser(self.renderer, self.env)
        parser.parse_cameras()        
        parser.parse_textures()
        parser.parse_materials()
        parser.parse_geometries()
        self.visual_objects = parser.visual_objects
        self.max_instances = parser.max_instances

    def render(self):
        """
        Update the viewer of the iGibson renderer.
        Call to render is made inside viewer's update.
        """
        if self.mode == 'gui' and self.viewer is not None:
            self.viewer.update()

    def step(self, action):
        """Updates the states for the wrapper given a certain action
        Args:
            action (np-array): the action the robot should take
        """        
        ret_val = super().step(action)
        for instance in self.renderer.instances:
            if instance.dynamic:
                self._update_position(instance, self.env)
        return ret_val

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
        super().reset()
        self.renderer.release()
        self.__init__(env=self.env,
                 width=self.width,
                 height=self.height,
                 enable_pbr=self.enable_pbr,
                 enable_shadow=self.enable_shadow,
                 msaa=self.msaa,
                 render2tensor=self.render2tensor,
                 optimized=self.optimized,
                 light_dimming_factor=self.light_dimming_factor,
                 device_idx=self.device_idx)
                
    def close(self):
        """
        Releases the iGibson renderer.
        """        
        self.renderer.release()

if __name__ == '__main__':

    # Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
    #                          PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal, 
    #                          PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    # Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e

    env = iGibsonWrapper(
        env = suite.make(
                "PickPlace",
                robots = ["Jaco"],
                reward_shaping=True,
                has_renderer=True,           
                has_offscreen_renderer=False,
                ignore_done=True,
                use_object_obs=True,
                use_camera_obs=False,  
                render_camera='frontview',
                control_freq=20, 
                camera_names=['frontview', 'agentview'],
                render_with_igibson=True
            ),
            enable_pbr=True,
            enable_shadow=True,
            modes=('rgb', 'seg', '3d', 'normal'),
            render2tensor=False,
            optimized=False,
    )

    # env.reset()

    for i in range(10000):
        action = np.random.randn(8)
        obs, reward, done, _ = env.step(action)
        env.render()

    env.close()
    
    print('Done.')



        

        


