# Renderers

[Renderers](../source/robosuite.renderers) are used to visualize the simulation and can be used either in on-screen mode or headless (off-screen) mode. Renderers are also responsible for generating image-based observations that are returned from a given environment, and compute virtual images of the environment based on the properties defined in the cameras.

Currently, the following ground-truth vision modalities are supported by the MuJoCo renderer:

- **RGB**: Standard 3-channel color frames with values in range `[0, 255]`. This is set during environment construction with the `use_camera_obs` argument.
- **Depth**: 1-channel frame with normalized values in range `[0, 1]`. This is set during environment construction with the `camera_depths` argument.
- **Segmentation**: 1-channel frames with pixel values corresponding to integer IDs for various objects. Segmentation can
    occur by class, instance, or geom, and is set during environment construction with the `camera_segmentations` argument.

**robosuite** presents the following rendering options:

<!-- ![Comparison of renderer options](../images/renderers/renderers.png "Comparison of renderer options") -->

## MuJoCo Default Renderer

MuJoCo exposes users to an OpenGL context supported by [mujoco](https://mujoco.readthedocs.io/en/latest/python.html#rendering). Based on [OpenGL](https://www.opengl.org/), our assets and environment definitions have been tuned to look good with this renderer. The rendered frames can be displayed in a window with [OpenCV's imshow](https://pythonexamples.org/python-opencv-imshow/).

![MuJoCo rendering](../images/gr1_cereal_mujoco.png "MuJoCo Default Renderer")

## Isaac Rendering

Users are also able to render using photorealistic methods through Isaac Sim. Specifically, we users are able to choose between two rendering modes: ray tracing and path tracing. For more information about Isaac Sim rendering options, please visit [here](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html). Isaac renderers are only available to those who are running on a Linux or Windows machine.

To install Isaac on your local system, please follow the instructions listed [here](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html). Make sure to follow instructions to install both Isaac Sim and Isaac Lab. 

### Ray tracing
![Ray tracing](../images/gr1_cereal_ray_tracing.png "Ray tracing")

Ray tracing can be performed in real time. We are currently working on enhancing the rendering pipeline to support an online viewer with ray tracing capabilities.

### Path tracing
![Path tracing](../images/gr1_cereal_path_tracing.png "Path tracing")

Path tracing typically offers higher quality and is ideal for offline learning. If you have the time to collect data and plan to train algorithms using offline data, we recommend using path tracing for its photorealistic results.

### Basic usage

Once all dependecies for Isaac rendering have been installed, users can run the `robosuite/scripts/render_dataset_with_omniverse.py` to render previously collected demonstrations using either ray tracing or path tracining. Below we highlight the arguments that can be passed into the script.

- **dataset**: Path to hdf5 dataset with the demonstrations to render.
- **ds_format**: Dataset format (options include `robosuite` and `robomimic` depending on if the dataset was collected using robosuite or robomimic, respectively).
- **episode**: Episode/demonstration to render. If no episode is provided, all demonstrations will be rendered.
- **output_directory**: Directory to store outputs from Isaac rendering and USD generation.
- **cameras**: List of cameras to render images. Cameras must be defined in robosuite.
- **width**: Width of the rendered output.
- **height**: Height of the rendered output.
- **renderer**: Renderer mode to use (options include `RayTracedLighting` or `PathTracing`).
- **save_video**: Whether to save the outputs renderings as a video.
- **online**: Enables online rendering and will not save the USD for future rendering offline.
- **skip_frames**: Renders every nth frame.
- **hide_sites**: Hides all sites in the scene.
- **reload_model**: Reloads the model from the Mujoco XML file.
- **keep_models**: List of names of models to keep from the original Mujoco XML file.
- **rgb**: Render with the RGB modality. If no other modality is selected, we default to rendering with RGB.
- **normals**: Render with normals.
- **semantic_segmentation**: Render with semantic segmentation.

Here is an example command to render an video of a demonstration using ray tracing with the RGB and normal modality.

```bash
$ python robosuite/scripts/render_dataset_with_omniverse.py --dataset /home/abhishek/Documents/research/rpl/robosuite/robosuite/models/assets/demonstrations_private/1734107564_9898326/demo.hdf5 --ds_format robosuite --episode 1 --camera agentview frontview --width 1920 --height 1080 --renderer RayTracedLighting --save_video --hide_sites --rgb --normals
```

### Rendering Speed

Below, we present a table showing the estimated frames per second when using these renderers. Note that the exact speed of rendering might depend on your machine and scene size. Larger scenes may take longer to render. Additionally, changing renderer inputs such as samples per pixel (spp) or max bounces might affect rendering speeds. The values below are estimates using the `Lift` task with an NVIDIA GeForce RTX 4090. We use an spp of 64 when rendering with path tracing.

| Renderer       | Estimated FPS |
|----------------|---------------|
| MuJoCo         | 3500          |
| Ray Tracing    | 58            |
| Path Tracing   | 2.8           |
