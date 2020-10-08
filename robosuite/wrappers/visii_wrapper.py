"""
This file implements a wrapper for using the ViSII imaging interface
for robosuite. 
"""

import numpy as np
import sys
import visii
import robosuite as suite
from mujoco_py import MjSim
from mujoco_py import load_model_from_path
from robosuite.wrappers import Wrapper
from robosuite.models.robots import Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e
from robosuite.models.robots import create_robot
import open3d as o3d

class VirtualWrapper(Wrapper):
	"""
	Initializes the ViSII wrapper.
	Args:
		env (MujocoEnv instance): The environment to wrap.
		width (int, optional)
		height (int, optional)
		use_noise (bool, optional): Use noise or denoise
		debug_mode (bool, optional): Use debug mode for visii
	"""
	def __init__(self,
				 env,
				 width=500,
				 height=500,
				 use_noise=True,
				 debug_mode=False):

		super().__init__(env)

		# Camera variables
		self.width  = width
		self.height = height

		if debug_mode:
			visii.initialize_interactive()
		else:
			visii.initialize_headless()

		if not use_noise: 
			visii.enable_denoiser()

		light_1 = visii.entity.create(
			name      = "light_1",
			mesh      = visii.mesh.create_plane("light_1"),
			transform = visii.transform.create("light_1"),
		)

		light_1.set_light(
			visii.light.create("light_1")
		)

		light_1.get_light().set_intensity(20000)
		light_1.get_transform().set_scale(visii.vec3(0.3))
		light_1.get_transform().set_position(visii.vec3(-3, -3, 2))
		
		floor = visii.entity.create(
			name      = "floor",
			mesh      = visii.mesh.create_plane("plane"),
			material  = visii.material.create("plane"),
			transform = visii.transform.create("plane")
		)
		floor.get_transform().set_scale(visii.vec3(10))
		floor.get_transform().set_position(visii.vec3(0, 0, -5))
		floor.get_material().set_base_color(visii.vec3(0.8, 1, 0.8))
		floor.get_material().set_roughness(0.4)
		floor.get_material().set_specular(0)

		#self.static_obj_init()
		self.camera_init()

		# Sets the primary camera to the renderer to the camera entity
		visii.set_camera_entity(self.camera) 

		# TODO (yifeng): 1. Put camera intiialization to a seprate
		# method - DONE
		# TODO (yifeng): 2. Create another function to configure the camera
		# parameters when needed - DONE

		self.camera_configuration(pos_vec = visii.vec3(0, 0, 1), 
								  at_vec  = visii.vec3(0,0,0), 
								  up_vec  = visii.vec3(0,0,1),
								  eye_vec = visii.vec3(1.5, 1.5, 1.5))
		
		# Environment configuration
		self._dome_light_intensity = 1
		visii.set_dome_light_intensity(self._dome_light_intensity)

		visii.set_max_bounce_depth(2)

		self.model = None
		
		self.robots = []
		self.robots_names = self.env.robot_names

		idNum = 1
		for robot in self.robot_names:
			robot_model = create_robot(robot, idn = idNum) # only taking the first robot for now
			self.robots.append(robot_model)
			idNum+=1

		### PASS IN XML FILE BASED ON ROBOT
		robot_xml_filepath = '../models/assets/robots/sawyer/robot.xml'
		self.initalize_simulation(robot_xml_filepath)

		self.base_pos = self.sim.data.body_xpos[self.sim.model.body_name2id("right_l1")]
		print(self.base_pos)

	def close(self):
		visii.deinitialize()

	def static_obj_init(self):

		# create the tables and walls
		raise NotImplementedError

	def camera_init(self):

		# intializes the camera
		self.camera = visii.entity.create(
			name = "camera",
			transform = visii.transform.create("camera_transform"),
		)

		self.camera.set_camera(
			visii.camera.create_perspective_from_fov(
				name = "camera_camera", 
				field_of_view = 1, 
				aspect = float(self.width)/float(self.height)
			)
		)

	def camera_configuration(self, pos_vec, at_vec, up_vec, eye_vec):

		# configures the camera
		self.camera.get_transform().set_position(pos_vec)

		self.camera.get_transform().look_at(
			at  = at_vec, # look at (world coordinate)
			up  = up_vec, # up vector
			eye = eye_vec,
			previous = False
		)

	def initalize_simulation(self, xml_file = None):
		
		self.mjpy_model = load_model_from_path(xml_file) if xml_file else self.model.get_model(mode="mujoco_py")

		# Creates the simulation
		self.sim = MjSim(self.mjpy_model)

	def str_to_class(self, str):
		return getattr(sys.modules[__name__], str)

	def reset(self):
		self.obs_dict = self.env.reset()

	def step(self, action):

		"""
		Updates the states for the wrapper given a certain action
		"""

		obs_dict, reward, done, info = self.env.step(action)
		self.update(obs_dict, reward, done, info)

		return obs_dict, reward, done, info

	def update(self, obs_dict, reward, done, info):
		self.obs_dict = obs_dict
		self.reward   = reward
		self.done     = done
		self.info     = info

		# call the render function to update the states in the window

	def render(self, render_type):
		"""
		Renders the scene
		Arg:
			render_type: tells the method whether to save to png or save to hdr
		"""

		# For now I am only rendering it as a png

		# stl file extension
		mesh = o3d.io.read_triangle_mesh('../models/assets/robots/sawyer/meshes/base.stl')
		link_name = 'base'

		normals  = np.array(mesh.vertex_normals).flatten().tolist()
		vertices = np.array(mesh.vertices).flatten().tolist() 

		mesh = visii.mesh.create_from_data(f'{link_name}_mesh', positions=vertices, normals=normals)

		link_entity = visii.entity.create(
			name      = link_name,
			mesh      = mesh,
			transform = visii.transform.create(link_name),
			material  = visii.material.create(link_name)
		)

		link_entity.get_transform().set_position(visii.vec3(0, 0, 0.2))

		link_entity.get_material().set_base_color(visii.vec3(0.2, 0.2, 0.2))
		link_entity.get_material().set_metallic(0)
		link_entity.get_material().set_transmission(0)
		link_entity.get_material().set_roughness(0.3)

		visii.render_to_png(
			width             = self.width,
			height            = self.height, 
			samples_per_pixel = 500,   
			image_path        = 'temp.png'
		)
		
	#def parse_mjcf_files(self):


	def printState(self): # For testing purposes
		print(self.obs_dict)

	def get_camera_intrinsics(self):
		"""Get camera intrinsics matrix
		"""
		raise NotImplementedError

	def get_camera_extrinsics(self):
		"""Get camera extrinsics matrix
		"""
		raise NotImplementedError

	def set_camera_intrinsics(self):
		"""Set camera intrinsics matrix
		"""
		raise NotImplementedError

	def set_camera_extrinsics(self):
		"""Set camera extrinsics matrix
		"""
		raise NotImplementedError
		
		
if __name__ == '__main__':

	env = VirtualWrapper(
		env = suite.make(
				"Lift",
				robots = "Sawyer",
				reward_shaping=True,
				has_renderer=False,       # no on-screen renderer
				has_offscreen_renderer=False, # no off-screen renderer
				ignore_done=True,
				use_object_obs=True,      # use object-centric feature
				use_camera_obs=False,     # no camera observations
				control_freq=10,
			),
		use_noise=False,
	)

	env.reset()

	action = np.random.randn(8)
	obs, reward, done, info = env.step(action)

	#env.printState() # For testing purposes

	#env.render(render_type = "png")

	env.close()

	print('Done.')
