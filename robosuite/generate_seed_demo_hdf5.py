import robosuite as suite
import numpy as np
import h5py
import os
import json
import time
import cv2
import datetime
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp  # <-- We are now using Slerp!

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

#
# --- This function is copied from collect_human_demonstrations.py ---
#
def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    """
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    grp = f.create_group("data")
    num_eps = 0
    env_name = None

    print(f"Searching for episode directories in: {directory}")
    for ep_directory in os.listdir(directory):
        ep_path = os.path.join(directory, ep_directory)
        if not os.path.isdir(ep_path):
            continue
            
        state_paths = os.path.join(ep_path, "state_*.npz")
        states = []
        actions = []
        success = False

        print(f"Processing episode: {ep_directory}")
        for state_file in sorted(glob(state_paths)):
            try:
                dic = np.load(state_file, allow_pickle=True)
                env_name = str(dic["env"])
                states.extend(dic["states"])
                for ai in dic["action_infos"]:
                    actions.append(ai["actions"])
                success = success or dic.get("successful", False)
            except Exception as e:
                print(f"Warning: Could not load state file {state_file}. Error: {e}")

        if len(states) == 0:
            print("  ...no states found, skipping.")
            continue

        if success:
            print(f"  ...Episode {ep_directory} is SUCCESSFUL and has been saved.")
            del states[-1] # Remove last state
            assert len(states) == len(actions)
            
            num_eps += 1
            ep_data_grp = grp.create_group(f"demo_{num_eps}")
            
            xml_path = os.path.join(ep_path, "model.xml")
            with open(xml_path, "r") as xml_f:
                xml_str = xml_f.read()
                
            ep_data_grp.attrs["model_file"] = xml_str
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print(f"  ...Episode {ep_directory} is UNSUCCESSFUL and has NOT been saved.")

    if num_eps == 0:
        print("\n" + "="*80)
        print("FATAL ERROR: No successful demonstrations were found to save.")
        print(f"Please check the temp directory: {directory}")
        print("=" * 80)

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    f.close()
# --- END OF COPIED FUNCTION ---


class CerealSeedGenerator:

    def __init__(self):
        
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

        # 2. Create argument configuration
        self.config = {
            "env_name": "PickPlaceClutter",
            "robots": "Panda",
            "controller_configs": controller_config,
        }
        
        # 3. Create environment
        print("Creating 'PickPlaceClutter' environment...")
        self.env = suite.make(
            **self.config,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            camera_names="agentview",
            camera_heights=512,
            camera_widths=512,
            control_freq=20,
        )
        
        # 4. Wrap the environment with data collection
        self.tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
        print(f"Using temp directory for HDF5 data: {self.tmp_directory}")
        self.env = DataCollectionWrapper(self.env, self.tmp_directory)

        # 5. Reset and get initial robot pose
        self.obs = self.env.reset()
        
        print("OBS KEYS AT RESET:", sorted(self.obs.keys()))
        for k in sorted(self.obs.keys()):
            try:
                print(k, np.array(self.obs[k]).shape)
            except Exception:
                print(k, type(self.obs[k]))
        
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat() # Use helper
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        self._rec = {}

    def _get_current_quat(self):
        """Helper to consistently get the correct quaternion from observations."""
        if "robot0_eef_quat_site" in self.obs:
            return self.obs["robot0_eef_quat_site"].copy()
        else:
            return self.obs["robot0_eef_quat"].copy()

    # --- NEW HELPER FUNCTION ---
    def _get_shortest_angle_diff(self, a_deg, b_deg):
        """
        Calculates the absolute shortest angle difference between two angles.
        Result will be in degrees, in range [0, 180].
        """
        a_rad = np.deg2rad(a_deg)
        b_rad = np.deg2rad(b_deg)
        diff_rad = np.arctan2(np.sin(a_rad - b_rad), np.cos(a_rad - b_rad))
        return np.abs(np.rad2deg(diff_rad))
    # --- END OF NEW HELPER ---
    #================================================================
    # Robot Movement Helpers
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
            
            # --- MODIFIED ACTION ---
            # Use the provided gripper state instead of hard-coding [0]
            action = np.concatenate([next_target_pos, R.from_quat(next_target_quat).as_rotvec(degrees=False), [gripper]])
            # --- END MODIFICATION ---
            
            self.obs, _, _, _ = self.env.step(action)
            print("OBS KEYS AFTER ONE STEP:", sorted(self.obs.keys()))
            self._video_frame("Moving to pose (smooth)")
            
        # wait a bit for any potential residual movement to complete
        for i in range (time_for_residual_movement):
            action = np.concatenate([target_pos, R.from_quat(target_quat).as_rotvec(degrees=False), [gripper]])
            self.obs, _, _, _ = self.env.step(action)
            print("OBS KEYS AFTER ONE STEP:", sorted(self.obs.keys()))

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
            print("OBS KEYS AFTER ONE STEP:", sorted(self.obs.keys()))
            self._video_frame(text)
            
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)

    #=====================================================================
    # Video Saving Helpers (Unchanged)
    #=====================================================================
    def _video_frame(self, text=None):
        """Records a single video frame."""
        if not getattr(self, "_rec", None) or not self._rec["on"]:
            return
        H, W, cam = self._rec["H"], self._rec["W"], self._rec["camera"]
        rgb = self.env.sim.render(camera_name=cam, height=H, width=W, depth=False)
        frame = cv2.flip(rgb, 0)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        if text:
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            
        # --- FIX FOR THE CV2 TYPO ---
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # --- END OF FIX ---
        
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def _video_start(self, path="cereal_seed_generation.mp4", fps=30, H=512, W=512, camera_name="agentview"):
        """Initializes the video recorder."""
        print(f"\n[VIDEO] Starting video recording, saving to: {path}")
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
    # Main Trajectory
    #================================================================
    def run_trajectory(self):
        
        self._video_start()
        
        print("\nExecuting hard-coded 'Pick Cereal' trajectory...")
        
        trajectory_successful = False
        
        try:
            cereal_body_id = self.env.obj_body_id["Cereal"]
            cereal_pos = self.env.sim.data.body_xpos[cereal_body_id]
            target_bin_pos = self.env.target_bin_placements[2]
            
            # --- UPDATED GRASPING STRATEGY ---
            print("  Analyzing object orientation...")
            
            cereal_mj_quat = self.env.sim.data.body_xquat[cereal_body_id]
            cereal_scipy_quat = np.array([
                cereal_mj_quat[1], cereal_mj_quat[2], cereal_mj_quat[3], cereal_mj_quat[0]
            ])
            cereal_yaw_deg = R.from_quat(cereal_scipy_quat).as_euler('xyz')[2]
            
            print(f"  Detected cereal yaw (long side): {cereal_yaw_deg:.2f} degrees.")
            
            # --- NEW: Find the most efficient 90-degree offset ---
            # Define the two valid yaw targets for grasping the short side
            grasp_yaw_1 = cereal_yaw_deg + 90.0
            grasp_yaw_2 = cereal_yaw_deg - 90.0
            
            # Define our "home" or "neutral" yaw
            neutral_yaw_deg = 90.0 # Based on the robot's default grasp
            
            # Find which grasp yaw is closer to our neutral yaw
            diff1 = self._get_shortest_angle_diff(grasp_yaw_1, neutral_yaw_deg)
            diff2 = self._get_shortest_angle_diff(grasp_yaw_2, neutral_yaw_deg)
            
            if diff1 < diff2:
                gripper_yaw_deg = grasp_yaw_1
                print(f"  Choosing +90 offset. Gripper yaw: {gripper_yaw_deg:.2f} (diff {diff1:.1f} deg)")
            else:
                gripper_yaw_deg = grasp_yaw_2
                print(f"  Choosing -90 offset. Gripper yaw: {gripper_yaw_deg:.2f} (diff {diff2:.1f} deg)")
            # --- END OF NEW LOGIC ---

            # Create the new grasp rotation
            grasp_rot = R.from_euler('xyz', [180, 0, gripper_yaw_deg], degrees=True)
            grasp_quat = grasp_rot.as_quat()

        except Exception as e:
            print(f"FATAL ERROR: Could not find 'Cereal' object or target bin.")
            print(f"Error: {e}")
            self._video_stop()
            self.env.close()
            return

        neutral_pos = self.robot_pos.copy()
        
        # --- UPDATED: Increased lift height to 35cm ---
        hover_pos_cereal = cereal_pos + np.array([0, 0, 0.35]) # Was 0.30
        grasp_pos_cereal = cereal_pos + np.array([0, 0, 0.03])
        hover_pos_bin = target_bin_pos + np.array([0, 0, 0.35]) # Was 0.30
        place_pos_bin = target_bin_pos + np.array([0, 0, 0.10])
        # --- END OF UPDATE ---
        
        try:
            self.move_gripper(-1, count=20)
            self.move_to_pose(hover_pos_cereal, grasp_quat, gripper=-1.0, count=60)
            self.move_to_pose(grasp_pos_cereal, grasp_quat, gripper=-1.0, count=50)
            self.move_gripper(1, count=20)
            self.move_to_pose(hover_pos_cereal, grasp_quat, gripper=1.0, count=50)
            self.move_to_pose(hover_pos_bin, grasp_quat, gripper=1.0, count=70)
            self.move_to_pose(place_pos_bin, grasp_quat, gripper=1.0, count=50)
            self.move_gripper(-1, count=20)
            self.move_to_pose(hover_pos_bin, grasp_quat, gripper=-1.0, count=50)
            self.move_to_pose(neutral_pos, grasp_quat, gripper=-1.0, count=60)
            
            print("\nTrajectory complete.")
            trajectory_successful = True
            
        except Exception as e:
            print(f"An error occurred during trajectory execution: {e}")
            trajectory_successful = False
        finally:
            self._video_stop()
            self.env.close()

        if trajectory_successful:
            print("Manually marking demonstration as 'successful'...")
            try:
                ep_directory_name = os.listdir(self.tmp_directory)[0]
                ep_directory_path = os.path.join(self.tmp_directory, ep_directory_name)
                
                state_files = sorted(list(filter(lambda x: "state" in x, os.listdir(ep_directory_path))))
                latest_state_file = os.path.join(ep_directory_path, state_files[-1])
                
                dic = np.load(latest_state_file, allow_pickle=True)
                dic_data = {k: dic[k] for k in dic.files}
                dic_data["successful"] = True
                np.savez(latest_state_file, **dic_data)
                print("  ...Success key injected.")
                
            except Exception as e:
                print(f"  ...ERROR: Failed to manually mark demo as successful. Error: {e}")
        else:
            print("Trajectory was not successful. HDF5 file will be empty.")

        print("\nSaving data to HDF5...")
        
        output_dir = os.path.expanduser("~/mimicgen_datasets/cereal_clutter")
        os.makedirs(output_dir, exist_ok=True)
        
        env_info = json.dumps(self.config)
        
        hdf5_path = os.path.join(output_dir, "demo.hdf5")
        gather_demonstrations_as_hdf5(self.tmp_directory, output_dir, env_info)
        
        if os.path.exists(hdf5_path):
            print(f"\nSuccess! Seed demo saved in: {hdf5_path}")
        else:
            print(f"\nFailed to create HDF5 file. Please check logs.")

if __name__ == "__main__":
    if os.environ.get("MUJOCO_GL") is None:
        print("Setting MUJOCO_GL=egl for headless rendering.")
        os.environ["MUJOCO_GL"] = "egl"
    else:
        print(f"MUJOCO_GL is already set to: {os.environ.get('MUJOCO_GL')}")
        
    print("\nInitializing Cereal Seed Generator...")
    generator = CerealSeedGenerator()
    
    print("Running trajectory...")
    generator.run_trajectory()
