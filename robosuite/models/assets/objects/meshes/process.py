import os
import numpy as np

xml_string = """
<mujoco model="bread">
  <asset>
    <mesh file="nontextured.stl" name="obj_mesh" scale="0.8 0.8 0.8"/>
    <texture file="../../../../../textures/bread.png" type="2d" name="tex-bread" />
    <material name="bread" reflectance="0.7" texrepeat="15 15" texture="tex-bread" texuniform="true"/>
  </asset>
  <worldbody>
    <body>
      <body name="collision">
        <geom pos=center mesh="obj_mesh" type="mesh" solimp="0.998 0.998 0.001" solref="0.001 1" density="50" friction="0.95 0.3 0.1"  material="bread" group="1" condim="4"/>
      </body>
      <body name="visual">
        <geom pos=center mesh="obj_mesh" type="mesh" material="bread"  conaffinity="0" contype="0"  group="0" mass="0.0001"/>
        <geom pos=center mesh="obj_mesh" type="mesh" material="bread"  conaffinity="0" contype="0"  group="1" mass="0.0001"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos=bottom name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos=top name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos=size name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""
for folder in os.listdir("ycb/"):
	print(folder)
	path = "ycb/"+folder+"/google_16k/nontextured.stl"
	ret = os.popen("surfaceCheck "+path+"| grep Bounding").read().split('\n')[0]
	ret = ret.split(":")[1].split(')')
	min_bound = np.asarray(list(map(float,ret[0][2:].split(" "))))
	max_bound = np.asarray(list(map(float,ret[1][2:].split(" "))))
	bounding_size = 0.5*(max_bound-min_bound)
	bounding_center = -0.5*(max_bound+min_bound)

	out = str(xml_string)
	out=out.replace("pos=center",'pos="'+str(bounding_center[0])+" "+str(bounding_center[1])+" "+str(bounding_center[2])+'"')
	out=out.replace("pos=size",'pos="'+str(bounding_size[0])+" "+str(bounding_size[1])+" "+str(bounding_size[2])+'"')
	out=out.replace("pos=bottom",'pos="0 0 -'+str(bounding_size[2])+'"')
	out=out.replace("pos=top",'pos="0 0 '+str(bounding_size[2])+'"')

	text_file = open("ycb/"+folder+"/google_16k/mesh.xml", "w")
	text_file.write(out)
	text_file.close()