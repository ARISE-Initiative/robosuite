# Renderers

[Renderers](../source/robosuite.renderers) are used to visualize the simulation, and can be used either in on-screen mode or headless (off-screen) mode. Renderers are also responsible for generating image-based observations that are returned from a given environment, and compute virtual images of the environment based on the properties defined in the cameras.

Currently, the following ground-truth vision modalities are supported across the three renderers, MuJoCo, NVISII, and iGibson:

- **RGB**: Standard 3-channel color frames with values in range `[0, 255]`. This is set during environment construction with the `use_camera_obs` argument.
- **Depth**: 1-channel frame with normalized values in range `[0, 1]`. This is set during environment construction with the `camera_depths` argument.
- **Segmentation**: 1-channel frames with pixel values corresponding to integer IDs for various objects. Segmentation can
    occur by class, instance, or geom, and is set during environment construction with the `camera_segmentations` argument.

Additional modalities are supported by a subset of the renderers. In **robosuite**, the user has the following rendering options:

![Comparison of renderer options](../images/renderers/renderers.png "Comparison of renderer options")

## MuJoCo

MuJoCo exposes users to an OpenGL context supported by [mujoco](https://mujoco.readthedocs.io/en/latest/python.html#rendering). Based on [OpenGL](https://www.opengl.org/), our assets and environment definitions have been tuned to look good with this renderer. The rendered frames can be displayed in a window with [OpenCV's imshow](https://pythonexamples.org/python-opencv-imshow/).

## NVISII
NVISIIRenderer is a ray tracing-based renderer. It is primarily used for training perception models and visualizing results in high quality. Through [NVISII](https://github.com/owl-project/NVISII), we can obtain different vision modalities, including depth, segmentations, surface normals, texture coordinates, and texture positioning.

![NVISII renderer vision modalities](../images/renderers/vision_modalities_nvisii.png "NVISII renderer vision modalities")

### Using the NVISII renderer
Installing NVISII can be done using the command `pip install nvisii`. Note that NVISII requires users' drivers to be up to date. Please refer [here](https://github.com/owl-project/NVISII) for more information. You can try the NVISII renderer with the `demo_renderers.py` [script](../demos.html#rendering-options) and learn about the APIs for obtaining vision modalities with `demo_nvisii_modalities.py`.

## iGibson
iGibsonRenderer uses a [physically based rendering](https://en.wikipedia.org/wiki/Physically_based_rendering) (PBR), a computer graphics technique that seeks to render images in a way that models the flow of light in the real world. The original [iGibson](http://svl.stanford.edu/igibson/) environment combines fast visual rendering with physics simulation based on Bullet. We have created a version of robosuite that uses only the renderer of iGibson. This renderer supports faster image generation with optimization, and the generation of a variety of vision modalities like depth, surface normal, and segmentation. It is capable of rendering and returning [PyTorch tensors](https://pytorch.org/docs/stable/tensors.html), allowing for tensor-to-tensor rendering that reduces the tensor copying time between CPU and GPU accelerating substantially the process of training models for exampe with reinforcement learning. 

![iGibson renderer vision modalities](../images/renderers/vision_modalities_igibson.png "iGibson renderer vision modalities")

### Using the iGibson Renderer
Installing iGibson can be done using the command `pip install igibson`. Please refer to the [iGibson installation guide](http://svl.stanford.edu/igibson/docs/installation.html) for a step by step guide. You can try the iGibson renderer with the `demo_renderers.py` [script](../demos.html#rendering-options) and learn about the APIs for obtaining vision modalities with `demo_igibson_modalities.py`.

### Requirements to use the iGibson Renderer

Using iGibson's PBR requires Linux or Windows machines; you can still use the iGibson renderer with macOS but it won't be PBR, it goes back to classic OpenGL rendering. Apart from that, the minimum system requirements are the following:

- Linux
  - Ubuntu 16.04
  - Nvidia GPU with VRAM > 6.0GB
  - Nvidia driver >= 384
  - CUDA >= 9.0, CuDNN >= v7
  - libegl-dev (Debian/Ubuntu: vendor neutral GL dispatch library – EGL support)
- Windows
  - Windows 10
  - Nvidia GPU with VRAM > 6.0GB
  - Nvidia driver >= 384
  - CUDA >= 9.0, CuDNN >= v7
- macOS
  - Tested on 10.15
  - PBR features not supported

**NOTE**: Since PBR is not supported on macOS, colors and textures will look flat. More realistic renderings can be obtained on Linux and Windows.

Other system configurations may work, but have not been extensively tested.

## Renderer Profiling
The following table shows the estimated frame rate of each renderer in frames per second (FPS). The profiling was conducted on a machine with Ubuntu 18.04, Intel Core i9-900K CPU@3.60GHz, and Nvidia RTX. The FPS numbers of each rendering option are reported below. These numbers are estimated on the Door environment with IIWA robot and Joint Velocity controller and 256x256 image size. In the table, R2T means render2tensor and R2N means render2numpy, which are two modes offered by the iGibson renderer.

|                   | mujoco | iGibson<br>(R2T optimized) | iGibson<br>(R2T) | iGibson<br>(R2N) | NVISII |
|-------------------|:---------:|:---------------------------------:|:-----------------------:|:----------------------:|:------:|
| Simulation + rendering | 62 | 64 | 58 | 45 | 0.5 |
| Rendering only         | 508 | 1392 | 285 | 271 | 0.5 |


For the same environment setup, we profiled the renderer on a machine with Ubuntu 18.04, Intel Core i7-8700K CPU@3.70GHz 
and Nvidia GTX 1080ti.

|                   | mujoco | iGibson<br>(R2T optimized) | iGibson<br>(R2T) | iGibson<br>(R2N) | NVISII |
|-------------------|:---------:|:---------------------------------:|:-----------------------:|:----------------------:|:------:|
| Simulation + rendering | 65 | 62 | 45 | 41 | 0.4 |
| Rendering only         | 1000 | 1500 | 250 | 205 | 0.4 |

In practice, both mujoco and iGibson renderers are well-suited for vision-based policy learning. In comparison, iGibson offers a faster rendering speed and additional functionalities for perception research. You might also find that iGibson has better cross-platform compatibility than the generic mujoco renderer, but it requires iGibson as a dependency. NVISII is best suited for photorealistic rendering; however, the ray-tracing computation substantially slows down its rendering speed compared to the other two renderers. It is mainly intended for perception tasks and qualitative visualizations, rather than online policy training.
