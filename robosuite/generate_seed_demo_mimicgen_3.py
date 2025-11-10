import robosuite as suite
import numpy as np
import h5py
import os
import json
import time
import cv2
import datetime
import argparse
import warnings
from glob import glob
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# --- Import your new environment to register it ---
try:
    import robosuite.environments.manipulation.pick_place_clutter
except ImportError:
    print("=" * 80)
    print("ERROR: Could not import PickPlaceClutter environment.")
    print("Please make sure 'pick_place_clutter.py' is in 'robosuite/environments/manipulation/'")
    print("and you have added 'from . import pick_place_clutter' to that folder's __init__.py")
    print("=" * 80)
    exit()

# --- Import Wrappers ---
from robosuite.wrappers import DataCollectionWrapper

# ==============================================================================
#
# --- MIMICGEN HDF5 CONVERTER FUNCTIONS ---
# (From mimicgen_seed_hdf5_recorder_simple.py)
#
# ==============================================================================

def quat_to_omega(q_prev_xyzw, q_curr_xyzw, dt):
    """Calculates angular velocity (omega) from two quaternions."""
    def to_wxyz(q):
        x,y,z,w = q
        return np.array([w,x,y,z], dtype=np.float64)

    qp = to_wxyz(q_prev_xyzw)
    qc = to_wxyz(q_curr_xyzw)
    r = np.array([
        qc[0]*qp[0] + qc[1]*(-qp[1]) + qc[2]*(-qp[2]) + qc[3]*(-qp[3]),
        qc[0]*(-qp[1]) + qc[1]*qp[0] + qc[2]*(-qp[3]) + qc[3]*qp[2],
        qc[0]*(-qp[2]) + qc[1]*qp[3] + qc[2]*qp[0] + qc[3]*(-qp[1]),
        qc[0]*(-qp[3]) + qc[1]*(-qp[2]) + qc[2]*qp[1] + qc[3]*qp[0],
    ], dtype=np.float64)
    angle = 2.0 * np.arccos(np.clip(r[0], -1.0, 1.0))
    s = np.sqrt(max(1e-12, 1.0 - r[0]*r[0]))
    axis = r[1:] / s if s > 1e-6 else np.array([0.0,0.0,0.0], dtype=np.float64)
    return (axis * angle) / max(1e-12, dt)


def stack_key(obs_list, key):
    """Stacks observations for a specific key from a list of obs dicts."""
    return np.stack([o[key] for o in obs_list], axis=0)


def resize_if_needed(arr, H, W):
    """Resizes a batch of images if they don't match the target H, W."""
    if arr.shape[1] == H and arr.shape[2] == W:
        return arr
    print(f"    Resizing images from {arr.shape[1:3]} to {(H, W)}...")
    return np.stack([cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in arr], axis=0)


def load_episode_npzs(ep_dir):
    """
    Loads all NPZ files from a DataCollectionWrapper episode directory.
    
    *** MODIFIED ***
    This version now checks for the 'successful' flag, just like your
    original `gather_demonstrations_as_hdf5` function did.
    """
    files = sorted(glob.glob(str(Path(ep_dir) / "state_*.npz")))
    if not files:
        raise FileNotFoundError(f"No 'state_*.npz' files found in {ep_dir}")
        
    obs_list, actions_list, xml_model_str = [], [], ""
    success = False
    
    for fp in files:
        try:
            dic = np.load(fp, allow_pickle=True)
            act = dic.get("actions", dic.get("action"))
            if act is None:
                raise KeyError(f"No 'actions' key in {fp}. Keys: {list(dic.keys())}")
            
            actions_list.append(np.asarray(act, dtype=np.float32))
            
            if "obs" not in dic:
                raise KeyError(f"No 'obs' key in {fp}. Keys: {list(dic.keys())}")
                
            obs = dic["obs"].item() if hasattr(dic["obs"], "item") else dic["obs"]
            obs_list.append(obs)
            
            if "model_file" in dic:
                try:
                    xml_model_str = str(dic["model_file"].item())
                except Exception:
                    pass
            
            # Check for success flag
            success = success or dic.get("successful", False)

        except Exception as e:
            print(f"Warning: Could not load state file {fp}. Error: {e}")

    if not success:
        raise RuntimeError(f"Episode {ep_dir} was not marked successful. Skipping.")

    # Remove the last observation, as it's the one *after* the final action
    if obs_list:
        del obs_list[-1]
        
    if not actions_list or not obs_list:
        raise RuntimeError(f"Episode {ep_dir} had no actions or observations. Skipping.")
        
    if len(obs_list) != len(actions_list):
         raise RuntimeError(f"Episode {ep_dir} has mismatch: {len(obs_list)} obs, {len(actions_list)} actions.")

    return obs_list, np.stack(actions_list, axis=0), xml_model_str


def episode_to_mg_core(h5_group, obs_list, actions, env_name, cameras, H, W, control_freq, xml_model_str):
    """
    Converts a single episode's data into the MimicGen HDF5 format.
    """
    T = actions.shape[0]
    dt = 1.0 / float(control_freq)

    # --- Check for required keys ---
    required = [
        "object-state",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_joint_pos",
        "robot0_joint_pos_cos",
        "robot0_joint_pos_sin",
        "robot0_joint_vel",
        "agentview_image",
        "robot0_eye_in_hand_image", # We now require this
    ]
    for k in required:
        if k not in obs_list[0]:
            # Try to use robot0_eef_quat_site if eef_quat is missing (robosuite 1.5+)
            if k == "robot0_eef_quat" and "robot0_eef_quat_site" in obs_list[0]:
                print("    Info: 'robot0_eef_quat' not in obs, using 'robot0_eef_quat_site' instead.")
                obs_list = [{**o, "robot0_eef_quat": o["robot0_eef_quat_site"]} for o in obs_list]
            else:
                raise KeyError(f"Missing observation key '{k}' in episode. Present keys: {sorted(obs_list[0].keys())}")

    # --- Stack low-dimensional observations ---
    object_vec     = stack_key(obs_list, "object-state").astype(np.float32)
    eef_pos        = stack_key(obs_list, "robot0_eef_pos").astype(np.float32)
    eef_quat       = stack_key(obs_list, "robot0_eef_quat").astype(np.float32)
    grip_qpos      = stack_key(obs_list, "robot0_gripper_qpos").astype(np.float32)
    grip_qvel      = stack_key(obs_list, "robot0_gripper_qvel").astype(np.float32)
    joint_pos      = stack_key(obs_list, "robot0_joint_pos").astype(np.float32)
    joint_pos_cos  = stack_key(obs_list, "robot0_joint_pos_cos").astype(np.float32)
    joint_pos_sin  = stack_key(obs_list, "robot0_joint_pos_sin").astype(np.float32)
    joint_vel      = stack_key(obs_list, "robot0_joint_vel").astype(np.float32)

    # --- Stack and resize images ---
    agent_imgs = stack_key(obs_list, "agentview_image").astype(np.uint8)
    agent_imgs = resize_if_needed(agent_imgs, H, W)

    eye_imgs = stack_key(obs_list, "robot0_eye_in_hand_image").astype(np.uint8)
    eye_imgs = resize_if_needed(eye_imgs, H, W)

    # --- Calculate velocities from finite differences ---
    eef_vel_lin = np.zeros_like(eef_pos, dtype=np.float32)
    eef_vel_lin[1:] = (eef_pos[1:] - eef_pos[:-1]) / max(1e-12, dt)

    eef_vel_ang = np.zeros_like(eef_pos, dtype=np.float32)
    for t in range(1, T):
        eef_vel_ang[t] = quat_to_omega(eef_quat[t-1], eef_quat[t], dt).astype(np.float32)

    # --- Create flat 'states' dataset (for compatibility) ---
    # We use robot0_proprio-state if available, otherwise build it
    if "robot0_proprio-state" in obs_list[0]:
         proprio_vec = stack_key(obs_list, "robot0_proprio-state").astype(np.float32)
    else:
        # Fallback: construct proprio state manually
        print("    Info: 'robot0_proprio-state' not found. Building it manually.")
        proprio_vec = np.concatenate([
            eef_pos, eef_quat, grip_qpos, grip_qvel,
            joint_pos, joint_pos_sin, joint_pos_cos, joint_vel
        ], axis=1).astype(np.float32)

    flat_states = np.concatenate([proprio_vec, object_vec], axis=1).astype(np.float32)

    # --- Write to HDF5 ---
    h5_group.attrs["model_file"] = xml_model_str
    h5_group.attrs["num_samples"] = int(T)

    h5_group.create_dataset("actions", data=actions, compression="gzip")
    h5_group.create_dataset("states", data=flat_states, compression="gzip")
    h5_group.create_dataset("dones", data=np.array([False]*(T-1) + [True], dtype=np.bool_), compression="gzip")
    h5_group.create_dataset("rewards", data=np.zeros((T,), dtype=np.float32), compression="gzip")

    # --- Create hierarchical 'obs' group ---
    og = h5_group.create_group("obs")
    og.create_dataset("agentview_image", data=agent_imgs, compression="gzip")
    og.create_dataset("robot0_eye_in_hand_image", data=eye_imgs, compression="gzip")
    og.create_dataset("object", data=object_vec, compression="gzip") # Renamed from 'object-state'
    og.create_dataset("robot0_eef_pos", data=eef_pos, compression="gzip")
    og.create_dataset("robot0_eef_quat", data=eef_quat, compression="gzip")
    og.create_dataset("robot0_eef_vel_ang", data=eef_vel_ang, compression="gzip")
    og.create_dataset("robot0_eef_vel_lin", data=eef_vel_lin, compression="gzip")
    og.create_dataset("robot0_gripper_qpos", data=grip_qpos, compression="gzip")
    og.create_dataset("robot0_gripper_qvel", data=grip_qvel, compression="gzip")
    og.create_dataset("robot0_joint_pos", data=joint_pos, compression="gzip")
    og.create_dataset("robot0_joint_pos_cos", data=joint_pos_cos, compression="gzip")
    og.create_dataset("robot0_joint_pos_sin", data=joint_pos_sin, compression="gzip")
    og.create_dataset("robot0_joint_vel", data=joint_vel, compression="gzip")
    
    # Add any other obs keys that exist
    other_keys = obs_list[0].keys() - required
    for k in other_keys:
        if k.endswith("image") or k in og or k == "robot0_eef_quat_site": # Skip images, duplicates, or handled keys
             continue
        try:
             og.create_dataset(k, data=stack_key(obs_list, k), compression="gzip")
        except Exception:
             print(f"    Skipping non-stackable obs key: {k}")


def convert_episodes_root_to_hdf5(episodes_root, out_h5, env_name, control_freq, cameras, camera_height, camera_width):
    """
    Converts a root directory full of 'ep_*' subdirectories into a
    single MimicGen HDF5 file.
    """
    root = Path(episodes_root)
    ep_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("ep_")])
    if not ep_dirs:
        raise FileNotFoundError(f"No ep_* subdirs found under {episodes_root}")

    with h5py.File(out_h5, "w") as f:
        g = f.create_group("data")
        total = 0
        for ep_dir in ep_dirs:
            print(f"Processing episode: {ep_dir.name}")
            try:
                obs_list, actions, xml_model_str = load_episode_npzs(ep_dir)
                print(f"  ...Loaded {actions.shape[0]} successful steps.")
            except Exception as e:
                warnings.warn(f"Skipping {ep_dir.name}: {e}")
                continue
            
            if actions.shape[0] < 20: # Sanity check
                 warnings.warn(f"Skipping {ep_dir.name}: Too few steps ({actions.shape[0]}).")
                 continue

            demo = g.create_group(f"demo_{total}")
            episode_to_mg_core(demo, obs_list, actions, env_name, cameras, camera_height, camera_width, control_freq, xml_model_str)
            total += 1

        if total == 0:
            print("\n" + "="*80)
            print("FATAL ERROR: No successful episodes were found or converted.")
            print(f"Please check the temp directory: {episodes_root}")
            print("=" * 80)
            f.close()
            os.remove(out_h5) # Clean up empty HDF5
            return

        # Store env info in 'data' group attributes
        env_info = {
            "env_name": env_name,
            "camera_names": cameras,
            "camera_height": camera_height,
            "camera_width": camera_width,
            "control_freq": control_freq,
        }
        g.attrs["env_args"] = json.dumps(env_info)
        g.attrs["total"] = int(total)
        
        # Add legacy robosuite attrs for compatibility
        now = datetime.datetime.now()
        g.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        g.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        g.attrs["repository_version"] = suite.__version__
        g.attrs["env"] = env_name

    print(f"\nWrote {out_h5} with {total} demos.")


# ==============================================================================
#
# --- YOUR CEREAL SEED GENERATOR CLASS ---
# (FIXED)
#
# ==============================================================================

class CerealSeedGenerator:

    def __init__(self, tmp_directory):
        
        # 1. Load Controller Config
        controller_name = "basic_abs_pose.json"
        controller_path = os.path.join(os.path.dirname(__file__), 'controllers', controller_name)
        
        if not os.path.exists(controller_path):
            print("=" * 80)
            print(f"FATAL ERROR: Controller file not found.")
            print(f"Expected path: {controller_path}")
            print(f"Please create a folder named 'controllers' and place '{controller_name}' inside it.")
            print("=" * 80)
            exit()
            
        print(f"Loading controller config from: {controller_path}")
        controller_config = suite.load_composite_controller_config(controller=controller_path)
        
        # --- Store key info for the converter ---
        self.control_freq = 20
        self.camera_names = ["agentview", "robot0_eye_in_hand"]
        self.camera_height = 84
        self.camera_width = 84

        # 2. Create argument configuration
        # This config dict now holds ALL environment arguments
        self.config = {
            "env_name": "PickPlaceClutter",
            "robots": "Panda",
            "controller_configs": controller_config,
            "control_freq": self.control_freq,
            "camera_names": self.camera_names,
            "camera_heights": self.camera_height, # <-- Note: env expects 'camera_heights' (plural)
            "camera_widths": self.camera_width,   # <-- Note: env expects 'camera_widths' (plural)
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
        }
        
        # 3. Create environment
        print("Creating 'PickPlaceClutter' environment...")
        
        # --- THIS IS THE FIX ---
        # We now *only* unpack self.config.
        # All arguments (camera_names, control_freq, etc.) are inside it.
        self.env = suite.make(
            **self.config
        )
        # --- END OF FIX ---
        
        # 4. Wrap the environment with data collection
        self.tmp_directory = tmp_directory # Use the one from main
        print(f"Using temp directory for raw data: {self.tmp_directory}")
        #self.env = DataCollectionWrapper(self.env, self.tmp_directory, record_video=False)
        self.env = DataCollectionWrapper(self.env, self.tmp_directory)

        # 5. Reset and get initial robot pose
#        self.obs = self.env.reset()
#        self.robot_pos = self.obs["robot0_eef_pos"].copy()
#        self.robot_quat = self._get_current_quat() # Use helper
#        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
#        self._rec = {}
#        self.video_path = "" # Placeholder for video path
# 5. Initialize placeholders
# 5. Initialize placeholders
        self.obs = None
        self.robot_pos = None
        self.robot_quat = None
        self.robot_rotvec = None
        self._rec = {}
        self.video_path = "" # Placeholder for video path

    def _get_current_quat(self):
        """Helper to consistently get the correct quaternion from observations."""
        if "robot0_eef_quat_site" in self.obs:
            return self.obs["robot0_eef_quat_site"].copy()
        else:
            return self.obs["robot0_eef_quat"].copy()

    #================================================================
    # Robot Movement Helpers (Unchanged)
    #================================================================
    
    def move_to_pose(self, target_pos, target_quat, gripper, count, time_for_residual_movement=10):
        """
        Moves the robot to the target pose in a straight line using Slerp.
        This is the smooth interpolation method from your run.py.
        """
        
        # rotation interpolation
        rotations = R.from_quat([self.robot_quat, target_quat])
        key_times = [0, 1]
        slerp = Slerp(key_times, rotations)

        for i in range (1, count+1):
            next_target_pos = (target_pos - self.robot_pos) * i/count + self.robot_pos
            next_target_quat = slerp(float(i)/count).as_quat()
            
            action = np.concatenate([next_target_pos, R.from_quat(next_target_quat).as_rotvec(degrees=False), [gripper]])
            
            self.obs, _, _, _ = self.env.step(action)
            self._video_frame("Moving to pose (smooth)")
            
        # wait a bit for any potential residual movement to complete
        for i in range (time_for_residual_movement):
            action = np.concatenate([target_pos, R.from_quat(target_quat).as_rotvec(degrees=False), [gripper]])
            self.obs, _, _, _ = self.env.step(action)

        # Update robot state
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)

    def move_gripper(self, gripper_action, count=20):
        """
        Operates the robot gripper for 'count' steps.
        """
        action = np.concatenate([self.robot_pos, self.robot_rotvec, [gripper_action]])
        
        text = "Closing gripper" if gripper_action > 0 else "Opening gripper"
        
        for i in range(count):
            self.obs, _, _, _ = self.env.step(action)
            self._video_frame(text)
            
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)

    #=====================================================================
    # Video Saving Helpers (Unchanged, but now renders 84x84)
    #=====================================================================
    def _video_frame(self, text=None):
        """Records a single video frame."""
        if not getattr(self, "_rec", None) or not self._rec["on"]:
            return
        # Use the env's camera size
        H, W, cam = self.camera_height, self.camera_width, self._rec["camera"]
        rgb = self.env.sim.render(camera_name=cam, height=H, width=W, depth=False)
        frame = cv2.flip(rgb, 0)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        if text:
            # Scale font for 84x84 images
            font_scale = 0.3
            thickness = 1
            cv2.putText(frame, text, (4, 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness*2, cv2.LINE_AA)
            cv2.putText(frame, text, (4, 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)
            
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def _video_start(self, path="cereal_seed_generation.mp4", fps=30, camera_name="agentview"):
        """Initializes the video recorder."""
        H, W = self.camera_height, self.camera_width
        print(f"\n[VIDEO] Starting video recording ({H}x{W}), saving to: {path}")
        self._rec = {"on": False, "path": path, "fps": fps, "H": H, "W": W, "camera": camera_name, "frames": 0}
        for fourcc_str in ("mp4v", "avc1", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(path, fourcc, fps, (W, H), True)
            if writer.isOpened():
                self._rec.update({"writer": writer, "fourcc": fourcc_str, "on": True})
                print(f"[VIDEO] Using codec: {fourcc_str}")
                break
        if not self._rec["on"]:
            print("[VIDEO] ERROR: Failed to open VideoWriter. Video will not be saved.")
            return
        _ = self.env.sim.render(camera_name=camera_name, height=H, width=W, depth=False)
        for _ in range(3): self._video_frame("STARTING")

    def _video_stop(self):
        """Stops and releases the video recorder."""
        if getattr(self, "_rec", None) and self._rec.get("on", False):
            for _ in range(10): self._video_frame("FINISHED")
            self._rec["writer"].release()
            print(f"[VIDEO] Saved to {self._rec['path']} (frames={self._rec['frames']})")
            self._rec["on"] = False

    #================================================================
    # Main Trajectory (Policy)
    #================================================================
    def run_trajectory(self):
    
    # --- THIS IS THE FIX ---
        # Reset the environment HERE to create the new episode directory
        self.obs = self.env.reset()
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat() # Use helper
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        
        # Video path now includes episode name
        ep_name = self.env.ep_directory.split("/")[-1]
        self.video_path = os.path.join(self.tmp_directory, f"{ep_name}_video.mp4")
        self._video_start(path=self.video_path)
        
        print(f"\nExecuting 'Pick Cereal' trajectory for episode: {ep_name}")
        
        try:
            cereal_pos = self.env.sim.data.body_xpos[self.env.obj_body_id["Cereal"]]
            target_bin_pos = self.env.target_bin_placements[2]
        except Exception as e:
            print(f"FATAL ERROR: Could not find 'CCereal' object or target bin.")
            print(f"Error: {e}")
            self._video_stop()
            self.env.close()
            return

        neutral_pos = self.robot_pos.copy()
        grasp_rot = R.from_euler('xyz', [180, 0, 90], degrees=True)
        grasp_quat = grasp_rot.as_quat()

        # Heights defined for the trajectory
        hover_pos_cereal = cereal_pos + np.array([0, 0, 0.30])
        grasp_pos_cereal = cereal_pos + np.array([0, 0, 0.03])
        hover_pos_bin = target_bin_pos + np.array([0, 0, 0.30])
        place_pos_bin = target_bin_pos + np.array([0, 0, 0.10])
        
        try:
            self.move_gripper(-1, count=20) # Open gripper
            self.move_to_pose(hover_pos_cereal, grasp_quat, gripper=-1.0, count=60)
            self.move_to_pose(grasp_pos_cereal, grasp_quat, gripper=-1.0, count=50)
            self.move_gripper(1, count=20) # Close gripper
            self.move_to_pose(hover_pos_cereal, grasp_quat, gripper=1.0, count=50) # Lift
            
            self.move_to_pose(hover_pos_bin, grasp_quat, gripper=1.0, count=70)
            self.move_to_pose(place_pos_bin, grasp_quat, gripper=1.0, count=50)
            self.move_gripper(-1, count=20) # Open gripper
            self.move_to_pose(hover_pos_bin, grasp_quat, gripper=-1.0, count=50) # Retract
            self.move_to_pose(neutral_pos, grasp_quat, gripper=-1.0, count=60) # Go home
            
            print("\nTrajectory complete.")
            print("Manually marking demonstration as 'successful'...")
            self.env.mark_success() # <-- CRITICAL: This flags the ep for conversion
            
        except Exception as e:
            print(f"An error occurred during trajectory execution: {e}")
            # Success is NOT marked
        finally:
            # This flushes the data to the temp NPZ files
            self.env.close()
            self._video_stop()

    #================================================================
    # --- NEW FUNCTION ---
    # Helper to pass config values to the converter
    #================================================================
    def get_converter_config(self):
        """Returns the env config needed by the HDF5 converter."""
        return {
            "env_name": self.config["env_name"],
            "control_freq": self.config["control_freq"],
            "camera_names": self.config["camera_names"],
            "camera_height": self.config["camera_heights"],
            "camera_width": self.config["camera_widths"],
        }

# ==============================================================================
#
# --- MAIN EXECUTION BLOCK ---
# (Updated to use get_converter_config)
#
# ==============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.expanduser("~/my_cereal_demo/demo.hdf5"),
        help="Path to save the final MimicGen HDF5 file."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of seed demonstrations to generate."
    )
    args = parser.parse_args()

    # Create ONE temp directory for all episodes
    tmp_root = f"/tmp/cereal_seed_gen_{str(time.time()).replace('.', '_')}"
    os.makedirs(tmp_root, exist_ok=True)
    print(f"Using temp directory for raw data: {tmp_root}")

    # Set MUJOCO_GL for headless rendering
    if os.environ.get("MUJOCO_GL") is None:
        print("Setting MUJOCO_GL=egl for headless rendering.")
        os.environ["MUJOCO_GL"] = "egl"
    else:
        print(f"MUJOCO_GL is already set to: {os.environ.get('MUJOCO_GL')}")

    env_config_for_converter = None # To store env info
    for i in range(args.num_episodes):
        print("\n" + "="*80)
        print(f"Initializing Cereal Seed Generator for episode {i+1} / {args.num_episodes}")
        print("="*80)
        
        # Pass the *same* tmp_root to the generator
        generator = CerealSeedGenerator(tmp_root)
        
        # Store the config from the first generator instance
        if env_config_for_converter is None:
            env_config_for_converter = generator.get_converter_config()
        
        print("Running trajectory...")
        generator.run_trajectory()
        print(f"Episode {i+1} complete. Debug video in {generator.video_path}")

    # --- After ALL episodes are run ---
    print("\n" + "="*80)
    print("All episodes generated. Converting to MimicGen HDF5...")
    print("="*80)

    if env_config_for_converter is None:
        print("No generator was created. Exiting.")
        exit()

    # Ensure output directory exists
    output_file = Path(args.output).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get env info from the last generator instance for the converter
    try:
        convert_episodes_root_to_hdf5(
            tmp_root,
            str(output_file),
            **env_config_for_converter
        )
    except Exception as e:
        print(f"FATAL ERROR during HDF5 conversion: {e}")
        print(f"Raw data is still available in: {tmp_root}")
