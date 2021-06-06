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
        """[summary]

        Args:
            env ([type]): [description]
            width (int, optional): [description]. Defaults to 1280.
            height (int, optional): [description]. Defaults to 720.
            enable_pbr (bool, optional): [description]. Defaults to True.
            enable_shadow (bool, optional): [description]. Defaults to True.
            msaa (bool, optional): [description]. Defaults to True.
            render2tensor (bool, optional): [description]. Defaults to False.
            optimized (bool, optional): [description]. Defaults to False.
            light_dimming_factor (float, optional): [description]. Defaults to 1.0.
            device_idx (int, optional): [description]. Defaults to 0.
            modes (list, optional): [description]. Defaults to ['rgb'].
        """
        super().__init__(env)

        check_modes(modes)

        # expose all params
        
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
        self.load()

        # set camera for renderer
        self.switch_camera(camera_name=self.env.render_camera)

        # add viewer
        self.add_viewer(initial_pos=self.camera_position, 
                        initial_view_direction=self.view_direction)

        # set parameters which will be used inside rovobosuite
        # when use_camera_obs=True
        self.env.ig_renderer_params = {'renderer':self.renderer,
                                       'modes': modes,
                                       'ig_wrapper_obj':self,
                                       }

        # Setup observables again after setting the iG parameters
        self.env._observables = self.env._setup_observables()

    def switch_camera(self, camera_name):
        self.camera_name = camera_name
        self.camera_position, self.view_direction, fov = self.get_camera_pose(camera_name)
        self.renderer.set_camera(self.camera_position,
                                 self.camera_position + self.view_direction,
                                 [0, 0, 1])
        self.renderer.lightP = ortho(-2, 2, -2, 2, -10, 25.0)
        self.renderer.set_fov(fov)

    def get_camera_pose(self, camera_name):
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


    def add_viewer(self, 
                 initial_pos = [0,0,1], 
                 initial_view_direction = [1,0,0], 
                 initial_up = [0,0,1]):
        """[summary]

        Args:
            initial_pos (list, optional): [description]. Defaults to [0,0,1].
            initial_view_direction (list, optional): [description]. Defaults to [1,0,0].
            initial_up (list, optional): [description]. Defaults to [0,0,1].
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

    def load(self):        
        parser = Parser(self.renderer, self.env)
        parser.parse_cameras()        
        parser.parse_textures()
        parser.parse_materials()
        parser.parse_geometries()
        self.visual_objects = parser.visual_objects
        self.max_instances = parser.max_instances

    def render(self, **kwargs):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.update_position(instance, self.env)
        if self.mode == 'gui' and self.viewer is not None:
            self.viewer.update()


    @staticmethod
    def update_position(instance, env):
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



        

        


