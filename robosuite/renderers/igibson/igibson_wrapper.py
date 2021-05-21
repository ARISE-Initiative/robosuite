import numpy as np
import robosuite as suite

from robosuite.wrappers import Wrapper
from robosuite.utils import transform_utils as T
from parser import Parser

from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, Instance
from gibson2.utils.mesh_util import xyzw2wxyz, quat2rotmat
from gibson2.render.viewer import Viewer

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
                 device_idx=0):
        """[summary]

        Args:
            env ([type]): [description]
            width (int, optional): [description]. Defaults to 512.
            height (int, optional): [description]. Defaults to 512.
            enable_pbr (bool, optional): [description]. Defaults to True.
            enable_shadow (bool, optional): [description]. Defaults to True.
            msaa (bool, optional): [description]. Defaults to True.
            render2tensor (bool, optional): [description]. Defaults to False.
            optimized (bool, optional): [description]. Defaults to False.
            light_dimming_factor (float, optional): [description]. Defaults to 1.0.
        """
        super().__init__(env)

        self.env = env
        self.render2tensor = render2tensor

        if self.env.use_camera_obs:
            self.mode = 'headless'
        else:
            self.mode = 'gui'
        
        self.mrs = MeshRendererSettings(msaa=msaa, 
                                        enable_pbr=enable_pbr, 
                                        enable_shadow=enable_shadow,
                                        optimized=optimized,
                                        light_dimming_factor=light_dimming_factor)

        if render2tensor:
            self.renderer_class = MeshRendererG2G
        else:
            self.renderer_class = MeshRenderer

        self.renderer = self.renderer_class(width=width,
                                            height=height,
                                            #TODO: fix vertical fov
                                            vertical_fov=45,
                                            device_idx=device_idx,
                                            rendering_settings=self.mrs)
        
        #TODO: Check this setting of camera
        camera_position = np.array([1.6,  0.,   1.45])
        view_direction = -np.array([0.9632, 0, 0.2574])
        self.renderer.set_camera(camera_position, camera_position + view_direction, [0, 0, 1])

        # add viewer
        self.add_viewer(initial_pos=camera_position, 
                        initial_view_direction=view_direction)

        self.load()



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
            self.viewer = Viewer(initial_pos = initial_pos,
                                initial_view_direction=initial_view_direction,
                                initial_up=initial_up)    
        else:
            self.viewer = None

    def load(self):
        parser = Parser(self.renderer, self.env)
        parser.parse_textures()
        parser.parse_materials()
        parser.parse_cameras()
        parser.parse_geometries()
        self.visual_objects = parser.visual_objects

    def render(self, **kwargs):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        # print(kwargs)
        # import pdb; pdb.set_trace();
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

    def close(self):
        self.renderer.release()

if __name__ == '__main__':

    # Registered environments: Lift, Stack, NutAssembly, NutAssemblySingle, NutAssemblySquare, NutAssemblyRound,
    #                          PickPlace, PickPlaceSingle, PickPlaceMilk, PickPlaceBread, PickPlaceCereal, 
    #                          PickPlaceCan, Door, Wipe, TwoArmLift, TwoArmPegInHole, TwoArmHandover

    # Possible robots: Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e

    env = iGibsonWrapper(
        env = suite.make(
                "NutAssembly",
                robots = ["Panda"],
                reward_shaping=True,
                has_renderer=False,           # no on-screen renderer
                has_offscreen_renderer=False, # no off-screen renderer
                ignore_done=True,
                use_object_obs=True,          # use object-centric feature
                use_camera_obs=False,         # no camera observations
                control_freq=10, 
            )
    )

    env.reset()

    for i in range(500):
        action = np.random.randn(8)
        obs, reward, done, info = env.step(action)

        if i%100 == 0:
            env.render(render_type = "png")

    env.close()
    
    print('Done.')



        

        


