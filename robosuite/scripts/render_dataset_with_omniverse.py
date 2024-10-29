import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Settings for SimulationApp")

# General arguments for robocasa script
parser.add_argument("--dataset", type=str, required=True, help="the hdf5 dataset")

parser.add_argument(
    "--ds_format", type=str, default="robomimic", help="the format of the dataset. Can be robosuite or robomimic"
)

# episode is a list
parser.add_argument(
    "--episode",
    type=int,
    nargs="+",
    default=[],
    help="(optional) episode number(s) to render. Default is all episodes in the dataset",
)

parser.add_argument(
    "--output_directory", type=str, default=None, help="(optional) directory to store outputs of USD rendering pipeline"
)

parser.add_argument(
    "--cameras",
    type=str,
    nargs="+",
    default=["agentview"],
    help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
    "None, which corresponds to a predefined camera for each env type",
)

# Arguments to set up simulation app
parser.add_argument(
    "--width", type=int, default=1280, help="(optional) width of the viewport and generated images. Defaults to 1280"
)

parser.add_argument(
    "--height", type=int, default=720, help="(optional) height of the viewport and generated images. Defaults to 720"
)

parser.add_argument(
    "--renderer",
    type=str,
    default="RayTracedLighting",
    help="(optional) rendering mode, can be RayTracedLighting or PathTracing. Defaults to RayTracedLighting",
)

parser.add_argument(
    "--save_video", action="store_true", default=False, help="(optional) save the rendered frames to a video file"
)

parser.add_argument(
    "--online",
    action="store_true",
    default=False,
    help="(optional) enable online rendering, will not save the usd file in this mode",
)

parser.add_argument(
    "--skip_frames", 
    type=int, 
    default=1, 
    help="(optional) render every nth frame. Defaults to 1"
)

parser.add_argument(
    "--hide_sites", 
    action="store_true", 
    default=False, 
    help="(optional) hide all sites in the scene"
)

parser.add_argument(
    "--reload_model",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--keep_models",
    type=str,
    nargs='+',
    default=[],
    help="(optional) keep the model from the Mujoco XML file"
)

# Add arguments for launch
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args = parser.parse_args()
# args.headless = True
args.enable_cameras = True
# Launch the Omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import json
import os
import re
import shutil
from enum import Enum

import carb.settings
import cv2
import h5py
import lxml.etree as ET
import omni
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
from termcolor import colored
from tqdm import tqdm

# Robocasa imports
try:
    import robocasa
except ImportError:
    print("Warning: Robocasa not found.")

try:
    import dexmimicgen_environments
    import mimicgen
except ImportError:
    print("Warning: MimicGen not found.")

# USD imports
import mujoco

import robosuite
import robosuite.utils.usd.exporter as exporter

scene_option = mujoco.MjvOption()
scene_option.geomgroup = [0, 1, 0, 0, 0, 0]


def make_sites_invisible(mujoco_xml):
    """
    Makes all site elements in a Mujoco XML string invisible by setting their rgba attribute to fully transparent.
    """
    # Parse the Mujoco XML string
    root = ET.fromstring(mujoco_xml)

    # Find all site elements
    site_elements = root.findall(".//site")

    # Make all site elements invisible by setting rgba attribute
    for site in site_elements:
        site.set("rgba", "0 0 0 0")  # Set rgba to fully transparent

    # Return the modified XML string
    return ET.tostring(root, encoding="unicode")


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if hasattr(env, "unset_ep_meta"):  # unset the ep meta after reset complete
            env.unset_ep_meta()
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    elif ds_format == "robosuite":
        env_name = f["data"].attrs["env"]
        env_info = json.loads(f["data"].attrs["env_info"])
        env_meta = dict(
            env_name=env_name,
            env_version=f["data"].attrs["repository_version"],
            env_kwargs=env_info,
        )
    else:
        raise ValueError
    f.close()
    return env_meta


class RobosuiteEnvInterface:
    """
    Env wrapper to load a robosuite demonstration
    """

    def __init__(self, dataset, episode, output_directory, cameras="agentview", reload_model=False, keep_models=[]) -> None:
        self.dataset = dataset
        self.episode = episode
        self.output_directory = output_directory
        self.cameras = cameras
        self.reload_model = reload_model
        self.keep_models = keep_models
        self.env = self.load_robosuite_environment()
        initial_state, self.states = self.get_states()
        reset_to(self.env, initial_state)
        self.exp = self.link_env_with_ov()
        self.traj_len = self.states.shape[0]

    def load_robosuite_environment(self):
        """
        Loads the specified robosuite demonstration and
        """
        env_meta = get_env_metadata_from_dataset(dataset_path=self.dataset, ds_format=args.ds_format)
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["has_offscreen_renderer"] = False
        env_kwargs["use_camera_obs"] = False
        # for robosuite compatibility, remove robocasa specific arguments
        if "env_lang" in env_kwargs:
            del env_kwargs["env_lang"]
        # temporary fix for backward compatibility here to load the correct config if version is less than v1.5.0
        # assume all version number is single digit
        if env_meta["env_version"] < "v1.5.0":
            from robosuite.controllers import load_composite_controller_config

            if "composite_controller_configs" in env_kwargs:
                del env_kwargs["composite_controller_configs"]
            robot = env_kwargs["robots"][0] if isinstance(env_kwargs["robots"], list) else env_kwargs["robots"]
            if robot == "PandaMobile":
                robot = "PandaOmron"
            composite_controller_config = load_composite_controller_config(controller=None, robot=robot)
            # breakpoint()
            env_kwargs["controller_configs"] = composite_controller_config
        print(colored("Initializing environment for {}...".format(env_kwargs["env_name"]), "yellow"))
        env = robosuite.make(**env_kwargs)

        return env

    def get_states(self):
        """
        Gets the initial state of the robosuite demonstration
        """
        f = h5py.File(self.dataset, "r+")
        ep = "demo_" + str(self.episode)
        states = f["data/{}/states".format(ep)][()][:: args.skip_frames]
        initial_state = dict(states=states[0])
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        if self.reload_model:
            env_xml = self.env.sim.model.get_xml()
            for name in self.keep_models:
                root_model = ET.fromstring(model_xml)
                root_env = ET.fromstring(env_xml)
                body_model = root_model.find(".//body[@name='{}']".format(name))
                body_env = root_env.find(".//body[@name='{}']".format(name))
                # change the properties of the body in env to match the model
                if body_model is not None and body_env is not None:
                    for attr_name, attr_value in body_model.attrib.items():
                        body_env.set(attr_name, attr_value)
                    env_xml = ET.tostring(root_env, encoding='unicode')
            model_xml = env_xml

        if args.hide_sites:
            model_xml = make_sites_invisible(model_xml)

        initial_state["model"] = model_xml
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        return initial_state, states

    def link_env_with_ov(self):
        """
        Loads the initial state of a robosuite scene into simulation
        """
        model = self.env.sim.model._model
        data = self.env.sim.data._data

        # Create a USD exporter instance with the current stage
        stage = None
        if args.online:
            stage = stage_utils.get_current_stage()

        exp = exporter.USDExporter(
            model=model,
            output_directory_name=self.output_directory,
            camera_names=args.cameras,
            online=args.online,
            shareable=not args.online,
            framerate=20,
            stage=stage,
        )
        exp.update_scene(data=data, scene_option=scene_option)
        exp.add_light(pos=[0, 0, 0], intensity=1500, light_type="dome", light_name="dome_1")
        return exp

    def update_simulation(self, index):
        """
        Renders a robosuite trajectory given the state at a given frame
        """
        state = self.states[index]
        reset_to(self.env, {"states": state})
        data = self.env.sim.data._data
        self.exp.update_scene(data=data, scene_option=scene_option)

    def close(self):
        if not args.online:
            self.exp.save_scene(filetype="usd")
        self.env.close()


__version__ = "0.0.2"


class RobosuiteWriter(rep.Writer):
    def __init__(
        self,
        output_dir: str = None,
        image_output_format: str = "png",
        rgb: bool = False,
        frame_padding: int = 4,
    ):
        self._output_dir = output_dir
        if output_dir:
            self._backend = rep.BackendDispatch(output_dir=output_dir)

        self._frame_id = 0
        self._frame_padding = frame_padding
        self._sequence_id = 0
        self._image_output_format = image_output_format
        self._output_data_format = {}
        self.annotators = []
        self.version = __version__
        self.data_structure = "annotator"
        self.write_ready = False

        # RGB
        if rgb:
            self.annotators.append(rep.AnnotatorRegistry.get_annotator("rgb"))

    def write(self, data: dict):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        Args:
            data: A dictionary containing the annotator data for the current frame.
        """
        if self.write_ready:
            for annotator_name, annotator_data in data["annotators"].items():
                for idx, (render_product_name, anno_rp_data) in enumerate(annotator_data.items()):
                    if annotator_name == "rgb":
                        filepath = os.path.join(args.cameras[idx], f"rgb_{self._frame_id}.{self._image_output_format}")
                        self._backend.write_image(filepath, anno_rp_data["data"])

            self._frame_id += 1


rep.WriterRegistry.register(RobosuiteWriter)


class RecorderState(Enum):
    READY = 0
    RUNNING = 1
    COMPLETED = 2


class DataGenerator:
    def __init__(self, robosuite_env) -> None:
        # State variables
        self.recorder_state = RecorderState.READY
        self.writer = None
        self.render_products = []
        self.current_frame = 0
        self.robosuite_env = robosuite_env

        # Define writer format
        self.writer_name = "RobosuiteWriter"

        self.num_frames = self.robosuite_env.traj_len
        self.rt_subframes = 1
        # skip 5 frames at the beginning to allow the scene to settle
        self.initial_skip = 5

        self.output_dir = os.path.abspath(self.robosuite_env.output_directory)

    def _check_if_valid_camera(self, path):
        context = omni.usd.get_context()
        stage = context.get_stage()
        prim = stage.GetPrimAtPath(path)

        if not prim.IsValid():
            print(f"{path} is not a valid prim path.")
            return False

        if prim.GetTypeName() == "Camera":
            return True
        else:
            print(f"{prim} is not a 'Camera' type.")
            return False

    def _check_if_valid_resolution(self, resolution):
        width, height = resolution[0], resolution[1]
        MAX_RESOLUTION = (16000, 8000)  # 16K
        if 0 <= width <= MAX_RESOLUTION[0] and 0 <= height <= MAX_RESOLUTION[1]:
            return True
        else:
            print(
                f"Invalid resolution: {width}x{height}. Must be between 1x1 and {MAX_RESOLUTION[0]}x{MAX_RESOLUTION[1]}."
            )
        return False

    def load_stage(self):
        usd_path = os.path.join(self.output_dir, f"frames/frame_{self.num_frames + 1}.usd")
        print(f"Opening stage {usd_path}")
        stage_utils.open_stage(usd_path)
        print("Stage loaded")

    def init_recorder(self):
        # Open USD stage
        if not args.online:
            self.load_stage()

        if carb.settings.get_settings().get("/omni/replicator/captureOnPlay"):
            carb.settings.get_settings().set_bool("/omni/replicator/captureOnPlay", False)

        carb.settings.get_settings().set_bool("/app/renderer/waitIdle", False)
        carb.settings.get_settings().set_bool("/app/hydraEngine/waitIdle", False)
        carb.settings.get_settings().set_bool("/app/asyncRendering", True)
        carb.settings.get_settings().set("/rtx/pathtracing/spp", 30)
        carb.settings.get_settings().set_bool("/exts/omni.replicator.core/Orchestrator/enabled", True)

        # Create writer for capturing generated data
        self.writer = rep.WriterRegistry.get(self.writer_name)
        self.writer.initialize(output_dir=self.output_dir, rgb=True)

        print("Writer Initiazed")

        # Create render products
        for camera_name in args.cameras:
            resolution = (args.width, args.height)
            camera_path = f"/World/Camera_Xform_{camera_name}/Camera_{camera_name}"
            if self._check_if_valid_camera(camera_path) and self._check_if_valid_resolution(resolution):
                rp = rep.create.render_product(camera_path, (resolution[0], resolution[1]), force_new=True)
                self.render_products.append(rp)
            else:
                print(f"Invalid render product entry: {camera_path}")

        print("Render products created")

        # Attach render products to writers
        if self.render_products:
            self.writer.attach(self.render_products)
        else:
            print("No valid render products found to initialize the writer.")
            return False

        print("Render products attached")
        return True

    def start_recorder(self):
        if self.recorder_state == RecorderState.READY and self.init_recorder():
            self.recorder_state = RecorderState.RUNNING
            self.run_recording_loop()
        else:
            self.finish_recording()

    def run_recording_loop(self):
        max_frames = self.num_frames if self.num_frames > 0 else 200  # Testing
        print(f"Recording {max_frames} frames to: {self.output_dir}")

        for _ in range(self.initial_skip):
            rep.orchestrator.step(rt_subframes=1, delta_time=None, pause_timeline=False)

        self.writer.write_ready = True
        if not args.online:
            timeline = omni.timeline.get_timeline_interface()
            timeline.set_end_time(max_frames)
        with tqdm(total=max_frames) as pbar:
            while self.current_frame < max_frames:
                if self.recorder_state != RecorderState.RUNNING:
                    break
                if args.online:
                    self.robosuite_env.update_simulation(self.current_frame)
                else:
                    timeline.forward_one_frame()
                rep.orchestrator.step(rt_subframes=self.rt_subframes, delta_time=None, pause_timeline=True)
                self.current_frame += 1
                pbar.update(1)

        if self.recorder_state == RecorderState.RUNNING:
            self.finish_recording()

    def finish_recording(self):
        if not args.online:
            timeline = omni.timeline.get_timeline_interface()
            timeline.stop()
        rep.orchestrator.wait_until_complete()
        print(f"[SDR] Finished;\tWrote {self.current_frame} frames to: {self.output_dir};")
        self.clear_recorder()
        self.recorder_state = RecorderState.COMPLETED

    def clear_recorder(self):
        if self.recorder_state != RecorderState.COMPLETED:
            self.recorder_state = RecorderState.COMPLETED
        if self.writer:
            self.writer.detach()
            self.writer = None
        for rp in self.render_products:
            rp.destroy()
        self.render_products.clear()
        stage_utils.clear_stage()
        if args.online:
            stage_utils.close_stage()
            stage_utils.create_new_stage()
        stage_utils.update_stage()

    def natural_sort_key(self, s):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]

    def create_video_from_frames(self, frame_folder, output_path, fps=30):
        frames = [f for f in os.listdir(frame_folder) if f.endswith(".png")]
        frames.sort(key=self.natural_sort_key)
        assert (
            len(frames) == self.current_frame
        ), f"Number of frames in folder ({len(frames)}) does not match number of frames rendered ({self.current_frame})"
        if not frames:
            print(f"No frames found in {frame_folder}")
            return

        first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
        height, width, layers = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            video.write(cv2.imread(os.path.join(frame_folder, frame)))

        video.release()
        print(f"Video saved: {output_path}")

    def process_folders(self):
        videos_folder = os.path.join(self.output_dir, "videos")
        os.makedirs(videos_folder, exist_ok=True)

        # Iterate over each camera folder in the output directory
        for camera in args.cameras:
            camera_folder_path = os.path.join(self.output_dir, camera)
            if not os.path.isdir(camera_folder_path):
                continue

            # Construct output filename and path
            output_filename = f"{camera}_rgb.mp4"
            output_path = os.path.join(videos_folder, output_filename)

            # Create the video from the frames in the camera folder
            self.create_video_from_frames(camera_folder_path, output_path)


def main():
    f = h5py.File(args.dataset, "r")
    total_episodes = args.episode
    if not total_episodes:
        # get all the int keys
        total_episodes = [int(k.split("_")[1]) for k in f["data"].keys()]
    f.close()
    print(f"Total episodes: {total_episodes}")
    dataset_name = "_".join(args.dataset.split("/")[-1].split(".")[:-1])
    if args.output_directory is None:
        # use the dataset directory
        args.output_directory = os.path.dirname(args.dataset)

    for episode in total_episodes:
        output_directory = os.path.join(args.output_directory, dataset_name, "demo_" + str(episode))
        robosuite_env = RobosuiteEnvInterface(
            dataset=args.dataset,
            episode=episode,
            output_directory=output_directory,
            cameras=args.cameras,
            reload_model=args.reload_model,
            keep_models=args.keep_models
        )
        if not args.online:
            # generate the usd first, and close the env to save the usd
            for i in tqdm(range(robosuite_env.traj_len)):
                robosuite_env.update_simulation(i)
            robosuite_env.close()

        data_generator = DataGenerator(robosuite_env)
        data_generator.start_recorder()

        while data_generator.recorder_state != RecorderState.COMPLETED:
            stage_utils.update_stage()

        if args.save_video:
            data_generator.process_folders()

        # remove all image frames
        for camera in args.cameras:
            camera_folder_path = os.path.join(output_directory, camera)
            if os.path.isdir(camera_folder_path):
                shutil.rmtree(camera_folder_path)

        print("Recording complete")


if __name__ == "__main__":
    main()
    print("Script execution completed.")
