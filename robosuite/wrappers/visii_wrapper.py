"""

This file implements a wrapper for using the ViSII imaging interface
for robosuite. 

"""

import numpy as np
import sys
import visii
import robosuite as suite
from robosuite.wrappers import Wrapper
from robosuite.models.robots import Baxter, IIWA, Jaco, Kinova3, Panda, Sawyer, UR5e

class VirtualWrapper(Wrapper):

	env = None

	# stores the current state
	obs_dict = {}
	reward = 0
	done = False
	info = ""


	def __init__(self, env):

		"""

		Initializes the ViSII wrapper.

		Args:
			env (MujocoEnv instance): The environment to wrap.

		"""

		self.env = env

		# Camera variables
		self.width  = 500
		self.height = 500

		self.denoiser = True

		visii.initialize_headless()

		if self.denoiser is True: 
			visii.enable_denoiser()

		# Creates an entity
		self.camera = visii.entity.create(
			name = "camera",
			transform = visii.transform.create("camera_transform"),
		)

		# Adds a camera to the entity
		self.camera.set_camera(
			visii.camera.create_perspective_from_fov(
				name = "camera_camera", 
				field_of_view = 1, 
				aspect = float(self.width)/float(self.height)
			)
		)

		# Sets the primary camera to the renderer to the camera entity
		visii.set_camera_entity(self.camera)

		# Lets place our camera to look at the scene
		# all the position are defined by visii.vector3  

		self.camera.get_transform().set_position(visii.vec3(0, 0, 1))

		self.camera.get_transform().look_at(
			at = visii.vec3(-50,0,80) , # look at (world coordinate)
			up = visii.vec3(0,0,1), # up vector
			eye = visii.vec3(-500,500,100 + 80),
			previous = False
		)

		self.camera.get_camera().set_aperture_diameter(2000)
		self.camera.get_camera().set_focal_distance(500)

		self.robots = []
		self.robots_names = self.env.robot_names

		for robot in self.robot_names:
			self.robots.append(self.str_to_class(robot)())

	def close(self):
		visii.deinitialize()

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

	def render(self, render_type):
		"""
		Renders the scene
		Arg:
			render_type: tells the method whether to save to png or save to hdr
		"""

		# For now I am only rendering it as a png
		
		floor = visii.entity.create(
			name = "floor",
			mesh = visii.mesh.create_plane("plane"),
			material = visii.material.create("plane"),
			transform = visii.transform.create("plane")
		)
		floor.get_transform().set_scale(visii.vec3(10000))
		floor.get_transform().set_position(visii.vec3(0, 0, -5))
		floor.get_material().set_base_color(visii.vec3(.0))
		floor.get_material().set_roughness(1)
		floor.get_material().set_specular(0)

		# adds the object to the scene 
		obj = visii.import_obj(
			"obj", # prefix name
			'../models/assets/robots/sawyer/meshes/head.obj', #obj path
			'../models/assets/robots/sawyer/meshes/', # mtl folder 
			visii.vec3(0,0,0), # translation 
			visii.vec3(1), # scale here
			visii.angleAxis(3.14 * .5, visii.vec3(1,0,0)) #rotation here
		)

		#obj[0].get_transform().set_position(visii.vec3(0, 0, 1))
		#obj[0].get_material().set_base_color(visii.vec3(0.9, 0.9, 0.9))

		obj[0].get_transform().set_position(visii.vec3(0,0,1))
		obj[0].get_transform().set_scale(visii.vec3(0.4))
		obj[0].get_material().set_base_color(visii.vec3(0.9, 0.9, 0.9)) 

		visii.render_to_png(
			width = self.width,
			height = self.height, 
			samples_per_pixel = 500,   
			image_path = 'temp.png'
		)
  
	#def parse_mjcf_files(self):


	def printState(self): # For testing purposes
		print(self.obs_dict)

if __name__ == '__main__':

	env = VirtualWrapper(
		env = suite.make(
				"Lift",
				robots = "Sawyer",
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

	action = np.random.randn(8)
	obs, reward, done, info = env.step(action)

	#env.printState() # For testing purposes

	env.render(render_type = "png")

	env.close()

	print('Done.')

