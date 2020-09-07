# Sensors

The simulator generates virtual physical signals as response to a robot's interactions. Virtual signals include images, force-torque measurements (from a force-torque sensor like the one included by default in the wrist of all [Gripper models](../modeling/gripper_model)), pressure signals (e.g. from a sensor on the robot's finger or on the environment), etc. Sensors (except cameras and joint sensors) are accessed via the function `get_sensor_measurement` provided the name of the sensor.

Joint sensors provide information about the state of each robot's joint including position and velocity. In MuJoCo these are not measured by sensors, but resolved and set by the simulator as the result of the actuation forces. Therefore, they are not accessed through the common `get_sensor_measurement` function but as properties of the [Robot simulation API](../simulation/robot), i.e., `_joint_positions` and `_joint_velocities`.

Cameras bundle a name to a set of properties to render images of the environment such as the pose and pointing direction, field of view, and resolution. Inheriting from Mujoco, cameras are defined in the [robot](../modeling/robot_model) and [arena models](../modeling/arena) and can be attached to any body. Images, as they would be generated from the cameras, are not accessed through `get_sensor_measurement` but via the renderer (see below). In a common user pipeline, images are not queried directly; we specify one or several cameras we want to use images from when we create the environment, and the images are generated and appended automatically to the observation dictionary.

#### Renderers

Renderers compute virtual images of the environment based on the properties defined in the cameras. In **robosuite**, the user can select one of these two renderers:

##### MjViewer

This is the default renderer from [mujoco-py](https://openai.github.io/mujoco-py/build/html/reference.html#mjviewer-3d-rendering). Based on [OpenGL](https://www.opengl.org/), our assets and environment definitions have been tuned to look good with this renderer. 

<!-- ##### iGibson Renderer-->
<!--This renderer is included in the [iGibson simulator](http://svl.stanford.edu/igibson/). We include an initial (not optimized) set of alternative meshes (OBJ files instead of the STL used by MjViewer) to be used with this simulator. The iGibson simulator can be used in simple mode (renders only albedo) or in physics-based rendering (PBR) mode (renders additional properties such as metallic, or roughness).--> 

##### PyGame

[PyGame](https://www.pygame.org/news) is a simple renderer that serves also as an alternative to MjViewer. A limitation of PyGame is that it can only render on-screen, limiting its applicability to train on computing clusters. However, it is useful for visualizing the robots' behaviors in the system runtime where MjViewer is not supported.