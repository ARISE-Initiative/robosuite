# Renderers

[Renderers](../source/robosuite.renderers) are used to visualize the simulation and can be used either in on-screen mode or headless (off-screen) mode. Renderers are also responsible for generating image-based observations that are returned from a given environment, and compute virtual images of the environment based on the properties defined in the cameras.

Currently, the following ground-truth vision modalities are supported by the MuJoCo renderer:

- **RGB**: Standard 3-channel color frames with values in range `[0, 255]`. This is set during environment construction with the `use_camera_obs` argument.
- **Depth**: 1-channel frame with normalized values in range `[0, 1]`. This is set during environment construction with the `camera_depths` argument.
- **Segmentation**: 1-channel frames with pixel values corresponding to integer IDs for various objects. Segmentation can
    occur by class, instance, or geom, and is set during environment construction with the `camera_segmentations` argument.

Additional modalities are supported by a subset of the renderers. In **robosuite**, the user has the following rendering options:

![Comparison of renderer options](../images/renderers/renderers.png "Comparison of renderer options")

## MuJoCo

MuJoCo exposes users to an OpenGL context supported by [mujoco](https://mujoco.readthedocs.io/en/latest/python.html#rendering). Based on [OpenGL](https://www.opengl.org/), our assets and environment definitions have been tuned to look good with this renderer. The rendered frames can be displayed in a window with [OpenCV's imshow](https://pythonexamples.org/python-opencv-imshow/).
