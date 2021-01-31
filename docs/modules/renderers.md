# Renderers

Renderers are used to visualize the simulation, and can be used either in on-screen mode or headless (off-screen) mode. Renderers are also responsible for generating image-based observations that are returned from a given environment, and compute virtual images of the environment based on the properties defined in the cameras. In **robosuite**, the user can select one of these two renderers:

##### MjViewer

This is the default renderer from [mujoco-py](https://openai.github.io/mujoco-py/build/html/reference.html#mjviewer-3d-rendering). Based on [OpenGL](https://www.opengl.org/), our assets and environment definitions have been tuned to look good with this renderer. 

<!-- ##### iGibson Renderer-->
<!--This renderer is included in the [iGibson simulator](http://svl.stanford.edu/igibson/). We include an initial (not optimized) set of alternative meshes (OBJ files instead of the STL used by MjViewer) to be used with this simulator. The iGibson simulator can be used in simple mode (renders only albedo) or in physics-based rendering (PBR) mode (renders additional properties such as metallic, or roughness).--> 

##### PyGame

[PyGame](https://www.pygame.org/news) is a simple renderer that serves also as an alternative to MjViewer. A limitation of PyGame is that it can only render on-screen, limiting its applicability to train on computing clusters. However, it is useful for visualizing the robots' behaviors in the system runtime where MjViewer is not supported.
