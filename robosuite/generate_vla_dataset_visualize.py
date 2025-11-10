#!/usr/binbin/env python3

import robosuite as suite
import numpy as np
import os
import json
import time
import cv2
import datetime
import argparse
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# --- 1. IMPORT camera_utils ---
from robosuite.utils import camera_utils

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


class VLADataGenerator:

    def __init__(self, output_dir):
        
        # 1. Load Controller Config (Unchanged)
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

        # 2. Create argument configuration (Unchanged)
        self.config = {
            "env_name": "PickPlaceClutter",
            "robots": "Panda",
            "controller_configs": controller_config,
        }
        
        # 3. Create environment (MODIFIED)
        print("Creating 'PickPlaceClutter' environment...")
        self.env = suite.make(
            **self.config,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            camera_names="agentview",
            # --- 2. CHANGED TO 256x256 ---
            camera_heights=256,
            camera_widths=256,
            camera_depths=True, # (Unchanged)
            control_freq=20,
        )
        
        # 4. Set up output directory (Unchanged)
        self.output_dir = os.path.expanduser(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Saving dataset to: {self.output_dir}")

        # 5. Reset and get initial robot pose (Unchanged)
        self.obs = self.env.reset()
        self.robot_pos = self.obs["robot0_eef_pos"].copy()
        self.robot_quat = self._get_current_quat()
        self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
        self._rec = {} # For video saving

        # 6. Camera parameters (MODIFIED)
        # --- 2. UPDATED TO 256x256 ---
        self.cam_width = self.env.camera_widths[0]
        self.cam_height = self.env.camera_heights[0]

    def _get_current_quat(self):
#        """Helper to consistently get the correct quaternion from observations."""
#        if "robot0_eef_quat_site" in self.obs:
#            return self.obs["robot0_eef_quat_site"].copy()
#        else:
#            return self.obs["robot0_eef_quat"].copy()

        return self.obs["robot0_eef_quat_site"].copy()  # SciPy-friendly [x,y,z,w]

            
    def _get_shortest_angle_diff(self, a_deg, b_deg):
        """
        Calculates the absolute shortest angle difference between two angles.
        Result will be in degrees, in range [0, 180].
        """
        a_rad = np.deg2rad(a_deg)
        b_rad = np.deg2rad(b_deg)
        diff_rad = np.arctan2(np.sin(a_rad - b_rad), np.cos(a_rad - b_rad))
        return np.abs(np.rad2deg(diff_rad))

    #================================================================
    # Robot Movement Helpers (Unchanged)
    #================================================================
    
    def move_to_pose(self, target_pos, target_quat, gripper, count, time_for_residual_movement=10):
        """
        Moves the robot to the target pose in a straight line using Slerp.
        """
        rotations = R.from_quat([self.robot_quat, target_quat])
        key_times = [0, 1]
        slerp = Slerp(key_times, rotations)

        for i in range (1, count+1):
            next_target_pos = (target_pos - self.robot_pos) * i/count + self.robot_pos
            next_target_quat = slerp(float(i)/count).as_quat()
            action = np.concatenate([next_target_pos, R.from_quat(next_target_quat).as_rotvec(degrees=False), [gripper]])
            self.obs, _, _, _ = self.env.step(action)
            self._video_frame("Moving to pose (smooth)")
            
        for i in range (time_for_residual_movement):
            action = np.concatenate([target_pos, R.from_quat(target_quat).as_rotvec(degrees=False), [gripper]])
            self.obs, _, _, _ = self.env.step(action)

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
    # Video Saving Helpers (Unchanged)
    #=====================================================================
    def _video_frame(self, text=None):
        """Records a single video frame."""
        if not getattr(self, "_rec", None) or not self._rec["on"]:
            return
        H, W, cam = self._rec["H"], self._rec["W"], self._rec["camera"]
        rgb = self.env.sim.render(camera_name=cam, height=H, width=W, depth=False)
        frame = cv2.flip(rgb, 0) # <-- 3. FLIP is here for video
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        if text:
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._rec["frames"] += 1
        self._rec["writer"].write(bgr)
        
    def _video_start(self, path="cereal_seed_generation.mp4", fps=30, H=256, W=256, camera_name="agentview"):
        """Initializes the video recorder."""
        print(f"\n[VIDEO] Starting video recording, saving to: {path}")
        # --- 2. UPDATED TO 256x256 ---
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
    # NEW: Data Generation Functions (Unchanged, but cam_width/height are now 256)
    #================================================================

    def _get_7dof_pose(self, pos, quat, gripper_state):
        """
        Converts a position, quaternion, and gripper state into a 7-DOF pose.
        (x, y, z, roll, pitch, yaw, gripper)
        """
        # Scipy rotation: R.from_quat() expects [x, y, z, w]
        # robosuite/self.obs provides [w, x, y, z]... but our grasp_quat is from Scipy,
        # so it's already in [x, y, z, w] format. We're good.
        rpy = R.from_quat(quat).as_euler('xyz', degrees=False) # 'xyz' = roll, pitch, yaw
        return [
            pos[0], pos[1], pos[2],
            rpy[0], rpy[1], rpy[2],
            gripper_state
        ]

    def generate_instruction(self, target_name, receptacle_name):
        """
        Generates a language instruction from a set of templates.
        """
        templates = [
            f"Pick up the {target_name}.",
            f"Place the {target_name} in the {receptacle_name}.",
            f"Grasp the {target_name} and drop it in the {receptacle_name}.",
            f"Move the {target_name} to the {receptacle_name}.",
            f"Put the {target_name} in the {receptacle_name}.",
            f"Grab the {target_name}.",
            f"Take the {target_name} and put it in the {receptacle_name}."
        ]
        return np.random.choice(templates)
        
    def save_sample(self, trial_idx, rgb_img, depth_img, instruction, waypoint_labels, K, T_wc):
        
        # --- 1. Create the episode directory ---
        episode_dir = os.path.join(self.output_dir, f"episode_{trial_idx:05d}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # --- 2. Save Images (The Missing Part) ---
        
        # Save RGB image
        # Note: cv2 saves in BGR, so we must convert from RGB
        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(episode_dir, "image_rgb.png"), rgb_bgr)
        
        # Save Depth image
        # .npy is perfect for saving a numpy array
        np.save(os.path.join(episode_dir, "image_depth.npy"), depth_img)

        # --- 3. Save Metadata (Your existing code) ---
        metadata = {
            "instruction": instruction,
            "waypoints": waypoint_labels,  # This now holds A1, A2, A3, A4
            # --- NEW: Save camera parameters ---
            "camera_intrinsics": K.tolist(),
            "camera_extrinsics_wc": T_wc.tolist()
            # --- END OF ADDITION ---
        }
        
        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
    def check_success(self, target_obj_name="Cereal"):
        """
        Checks if the target object is inside the target bin, using the
        environment's own ground-truth logic.
        """
        try:
            # 1. Get final object position
            obj_id_in_sim = self.env.obj_body_id[target_obj_name]
            final_obj_pos = self.env.sim.data.body_xpos[obj_id_in_sim]
            
            # 2. Get the object's target bin_id (Cereal = 2)
            # This is based on the list order in pick_place_clutter.py
            target_bin_id = self.env.object_to_id[target_obj_name.lower()]
            
            # 3. Get the environment's bin parameters
            bin2_pos = self.env.bin2_pos
            bin_size = self.env.bin_size # This is the full table/bin-area size
            
            # 4. Re-implement the 'not_in_bin' logic from the env
            
            bin_x_low = bin2_pos[0]
            bin_y_low = bin2_pos[1]
            
            # --- This logic is directly from pick_place_clutter.py ---
            if target_bin_id == 0 or target_bin_id == 2:
                bin_x_low -= bin_size[0] / 2.0
            if target_bin_id < 2:
                bin_y_low -= bin_size[1] / 2.0
            # --- End of copied logic ---

            bin_x_high = bin_x_low + bin_size[0] / 2.0
            bin_y_high = bin_y_low + bin_size[1] / 2.0
            
            in_x = (bin_x_low < final_obj_pos[0] < bin_x_high)
            in_y = (bin_y_low < final_obj_pos[1] < bin_y_high)
            # Z check is slightly different
            in_z = (bin2_pos[2] < final_obj_pos[2] < bin2_pos[2] + 0.1)

            is_success = in_x and in_y and in_z

            print(f"Success Check: Object Pos={final_obj_pos}")
            print(f"Success Check: Bin (ID {target_bin_id}) X=[{bin_x_low:.2f}, {bin_x_high:.2f}], Y=[{bin_y_low:.2f}, {bin_y_high:.2f}]")
            print(f"Success Check: In X? {in_x}, In Y? {in_y}, In Z? {in_z}")
            
            return is_success

        except Exception as e:
            print(f"Error during success check: {e}")
            return False

    #================================================================
    # REFACTORED: Main Trajectory and Generation Loop
    #================================================================

    def execute_trajectory(self, trajectory_goals_3d):
        """
        Runs the hard-coded trajectory.
        Returns True on success, False on failure.
        """
        try:
            # 1. Get poses from the labels
            # 1. Get poses from the labels
            hover_pos_cereal = np.array(trajectory_goals_3d["A1_pregrasp"][0])
            grasp_pos_cereal = np.array(trajectory_goals_3d["A2_grasp"][0])
            place_pos_bin = np.array(trajectory_goals_3d["A3_release"][0])

            # --- Get BOTH quaternions ---
            grasp_quat = np.array(trajectory_goals_3d["A1_pregrasp"][1])
            place_quat = np.array(trajectory_goals_3d["A3_release"][1])
            # --- END MODIFICATION ---
            
            # 2. Get the "hover_pos_bin"
            target_bin_pos = self.env.target_bin_placements[2]
            # --- Use the same hover height as the pre-grasp ---
            hover_height = hover_pos_cereal[2] - grasp_pos_cereal[2]
            hover_pos_bin = target_bin_pos + np.array([0, 0, hover_height]) # e.g., 0.35
            
            # 3. Get current "home" pose
            neutral_pos = self.robot_pos.copy()
            neutral_quat = self.robot_quat.copy()

            # --- NEW 3-STEP GRASP TRAJECTORY ---
            
            # Open gripper at home
            self.move_gripper(-1, count=20)
            
            # Step 1: Move to Pre-Grasp POSITION (but keep home orientation)
            print("  Moving to pre-grasp position...")
            self.move_to_pose(hover_pos_cereal, neutral_quat, gripper=-1.0, count=60)
            
            # Step 2: Rotate to Grasp ORIENTATION (in the air)
            print("  Rotating to grasp orientation...")
            self.move_to_pose(hover_pos_cereal, grasp_quat, gripper=-1.0, count=40)
            
            # Step 3: Descend to Grasp POSITION
            print("  Descending to grasp...")
            self.move_to_pose(grasp_pos_cereal, grasp_quat, gripper=-1.0, count=50)

            # --- (Rest of trajectory is the same) ---
            
            print("  Closing gripper...")
            self.move_gripper(1, count=20)
            
            print("  Lifting object...")
            self.move_to_pose(hover_pos_cereal, grasp_quat, gripper=1.0, count=50)

            # --- NEW: ROTATE IN AIR TO STANDARD ORIENTATION ---
            print("  Rotating to standard place orientation...")
            self.move_to_pose(hover_pos_cereal, place_quat, gripper=1.0, count=40)
            # --- END OF ADDITION ---

            print("  Moving to bin...")
            self.move_to_pose(hover_pos_bin, place_quat, gripper=1.0, count=70)

            print("  Dropping object...")
            self.move_to_pose(place_pos_bin, place_quat, gripper=1.0, count=50)
            self.move_gripper(-1, count=20)

            print("  Retracting...")
            self.move_to_pose(hover_pos_bin, place_quat, gripper=-1.0, count=50)
            
            print("  Returning home...")
            self.move_to_pose(neutral_pos, neutral_quat, gripper=-1.0, count=60)
            
            print("Trajectory execution completed.")
            
            # Call the check *after* the trajectory is done
            return self.check_success(target_obj_name="Cereal")
        except Exception as e:
            print(f"An error occurred during trajectory execution: {e}")
            # Add a frame with the error text
            self._video_frame(f"CRITICAL ERROR: {e}")
            self._video_stop() # Stop video on error
            return False

    def run_generation_loop(self, num_trials, num_videos):
        """
        Main loop to generate and save data samples.
        """
        print(f"Starting data generation for {num_trials} trials...")
        success_count = 0
        
        for i in range(num_trials):
            print(f"\n--- Trial {i+1} / {num_trials} ---")
            
            # --- ADD THIS BLOCK TO START VIDEO ---
            save_video_this_trial = (i < num_videos)
            if save_video_this_trial:
                # Save video to the root output dir, named by trial number
                video_path = os.path.join(self.output_dir, f"debug_video_trial_{i:05d}.mp4")
                self._video_start(path=video_path)
            # --- END OF ADDITION ---
            
            # Reset env and robot state
            self.obs = self.env.reset()
            self.robot_pos = self.obs["robot0_eef_pos"].copy()
            self.robot_quat = self._get_current_quat()
            self.robot_rotvec = R.from_quat(self.robot_quat).as_rotvec(degrees=False)
            
            # --- CAPTURE "HOME" POSE (A4) ---
            # This is the pose we'll return to.
            neutral_pos = self.robot_pos.copy()
            neutral_quat = self.robot_quat.copy()

            # --- ADD THIS BLOCK ---
            # Get Camera Intrinsics and Extrinsics
            cam_name = "agentview"
            K = camera_utils.get_camera_intrinsic_matrix(
                self.env.sim, cam_name, self.cam_height, self.cam_width
            )
            T_wc = camera_utils.get_camera_extrinsic_matrix(
                self.env.sim, cam_name
            )
            # --- END OF ADDITION ---
            
            # --- 3. MODIFIED IMAGE CAPTURE ---
            # 1. Get initial images (AT THE START)
            rgb_image_raw = self.obs["agentview_image"]
            
            # --- 3. MODIFIED IMAGE CAPTURE ---
            # 1. Get initial images (AT THE START)
            rgb_image_raw = self.obs["agentview_image"]
            depth_image_raw = self.obs["agentview_depth"]
            
            # 2. Process RGB
            # Flip vertically (MuJoCo origin is bottom-left)
            rgb_image = cv2.flip(rgb_image_raw, 0)
            
            # 3. Process Depth
            # Convert to meters
            depth_real = camera_utils.get_real_depth_map(self.env.sim, depth_image_raw)
            # Flip vertically to match RGB
            depth_image = cv2.flip(depth_real, 0)
            # --- END OF MODIFICATIONS ---
            
        # 4. Get 3D poses and instruction
            try:
                cereal_body_id = self.env.obj_body_id["Cereal"]
                cereal_pos = self.env.sim.data.body_xpos[cereal_body_id]
                cereal_mj_quat = self.env.sim.data.body_xquat[cereal_body_id] # <-- ADD THIS

                target_bin_pos = self.env.target_bin_placements[2]
                target_obj_name = "cereal box"
                receptacle_name = "target bin"
            except Exception as e:
                print(f"ERROR: Could not find objects in scene. Skipping trial. {e}")
                continue
                
            instruction = self.generate_instruction(target_obj_name, receptacle_name)
            print(f"Generated Instruction: {instruction}")
            
            # 5. Calculate 3D and 2D Labels
            
            # 5. Calculate 7-DOF Waypoint Labels
            # --- NEW DYNAMIC GRASP LOGIC (MODIFIED as per ground-truth side vector) ---
            print("  Analyzing object orientation using ground-truth rotation matrix...")
            
            # 1. Convert MuJoCo quat [w, x, y, z] to SciPy quat [x, y, z, w]
            cereal_scipy_quat = np.array([
                cereal_mj_quat[1], cereal_mj_quat[2], cereal_mj_quat[3], cereal_mj_quat[0]
            ])
            
            # 2. Get the cereal box's rotation from its quaternion
            cereal_rotation = R.from_quat(cereal_scipy_quat)
            
            # 3. Get the "long side" (local X-axis) as a world-frame vector
            #    This is the first column of the rotation matrix.
            box_long_side_vector = cereal_rotation.as_matrix()[:, 0]
            print(f"  Detected box long side vector (world): {np.round(box_long_side_vector, 2)}")

            # 4. Calculate the perpendicular grasp vector (90-deg rotation in XY plane)
            #    This vector is perpendicular to the *long side*, so it's aligned with the *short side*.
            #    (dx, dy) -> (-dy, dx)
            x_axis_gripper = np.array([-box_long_side_vector[1], box_long_side_vector[0], 0])

            # 5. Normalize the vector
            norm = np.linalg.norm(x_axis_gripper)
            if norm < 1e-5:
                # Failsafe: happens if the box's long side is pointing straight up/down
                print("  Warning: Box long side is Z-aligned. Defaulting to world X-axis grasp.")
                x_axis_gripper = np.array([1., 0., 0.])
            else:
                x_axis_gripper /= norm
            print(f"  Calculated grasp X-axis (world): {np.round(x_axis_gripper, 2)}")

            # 6. Define Z-axis as "straight down"
            z_axis_gripper = np.array([0., 0., -1.])
            
            # 7. Find Y-axis via cross product
            y_axis_gripper = np.cross(z_axis_gripper, x_axis_gripper)
            
            # 8. Build rotation matrix and get quaternion
            grasp_rotation_matrix = np.array([x_axis_gripper, y_axis_gripper, z_axis_gripper]).T
            grasp_rot = R.from_matrix(grasp_rotation_matrix)
            grasp_quat = grasp_rot.as_quat()

            # 9. Define the standard "placing" orientation (unchanged)
            standard_place_rot = R.from_euler('xyz', [180, 0, 90], degrees=True)
            standard_place_quat = standard_place_rot.as_quat()
            # --- END OF MODIFIED LOGIC ---

            # --- NEW DYNAMIC GRASP LOGIC ---
#            print("  Analyzing object orientation...")
#            cereal_scipy_quat = np.array([
#                cereal_mj_quat[1], cereal_mj_quat[2], cereal_mj_quat[3], cereal_mj_quat[0]
#            ])
#            cereal_yaw_deg = R.from_quat(cereal_scipy_quat).as_euler('xyz')[2]
#            print(f"  Detected cereal yaw (long side): {cereal_yaw_deg:.2f} degrees.")
#
#            # Define the grasp yaw: just add 90 degrees to the box's long-side yaw
#            gripper_yaw_deg = cereal_yaw_deg + 90.0
#            print(f"  Calculating grasp yaw (long side + 90): {gripper_yaw_deg:.2f}")
#            
#            # Create the new grasp rotation
#            grasp_rot = R.from_euler('xyz', [180, 0, gripper_yaw_deg], degrees=True)
#            grasp_quat = grasp_rot.as_quat()
#            
#            standard_place_rot = R.from_euler('xyz', [180, 0, 90], degrees=True)
#            standard_place_quat = standard_place_rot.as_quat()
#            # --- END OF NEW LOGIC ---
#            grasp_rot = R.from_euler('xyz', [180, 0, 90], degrees=True)
#            grasp_quat = grasp_rot.as_quat()

            # A1: Pre-Grasp
            a1_pos = cereal_pos + np.array([0, 0, 0.30])
            # A2: Grasp
            a2_pos = cereal_pos + np.array([0, 0, 0.03])
            # A3: Release
            a3_pos = target_bin_pos + np.array([0, 0, 0.10])
            a4_pos = neutral_pos # Use the "home" pos we just saved
            
            # Use helper to create 7-DOF poses
            # Use helper to create 7-DOF poses
            waypoint_labels = {
                "A1_pregrasp": self._get_7dof_pose(a1_pos, grasp_quat, -1.0), # Gripper Open
                "A2_grasp":    self._get_7dof_pose(a2_pos, grasp_quat,  1.0), # Gripper Closed
                # --- MODIFY THIS LINE ---
                "A3_release":  self._get_7dof_pose(a3_pos, standard_place_quat, -1.0), # Gripper Open
                # --- END MODIFICATION ---
                "A4_home":     self._get_7dof_pose(a4_pos, neutral_quat, -1.0)  # Gripper Open
            }
            
            trajectory_goals_3d = {
                 "A1_pregrasp": (a1_pos.tolist(), grasp_quat.tolist()),
                 "A2_grasp": (a2_pos.tolist(), grasp_quat.tolist()),
                 # --- MODIFY THIS LINE ---
                 "A3_release": (a3_pos.tolist(), standard_place_quat.tolist()),
                 # --- END MODIFICATION ---
            }
            is_success = self.execute_trajectory(trajectory_goals_3d)
            
            # 7. Save data IF successful
            if is_success:
                success_count += 1
                print(f"Trial {i+1} SUCCESSFUL. Saving sample.")
                # --- Pass processed images to save_sample ---
                #self.save_sample(i, rgb_image, depth_image, instruction, waypoint_labels)
                self.save_sample(
                    trial_idx=i,
                    rgb_img=rgb_image,
                    depth_img=depth_image,
                    instruction=instruction,
                    waypoint_labels=waypoint_labels,
                    K=K,           # <-- ADD THIS
                    T_wc=T_wc      # <-- ADD THIS
                )
            else:
                print(f"Trial {i+1} FAILED. Discarding sample.")
                
            # --- ADD THIS BLOCK TO STOP VIDEO ---
            if save_video_this_trial:
                if is_success:
                    self._video_frame("TRIAL SUCCESSFUL")
                else:
                    self._video_frame("TRIAL FAILED")
                self._video_stop()
            # --- END OF ADDITION ---
                
        print(f"\nGeneration complete. {success_count} / {num_trials} successful samples saved.")
        self.env.close()


if __name__ == "__main__":
    if os.environ.get("MUJOCO_GL") is None:
        print("Setting MUJOCO_GL=egl for headless rendering.")
        os.environ["MUJOCO_GL"] = "egl"
    else:
        print(f"MUJOCO_GL is already set to: {os.environ.get('MUJOCO_GL')}")
        
    # --- NEW: Argparse ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/my_openvla_dataset",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    
    # --- ADD THIS ARGUMENT ---
    parser.add_argument(
        "--num_videos",
        type=int,
        default=5,
        help="Number of initial videos to save for debugging (e.g., 5). Set to 0 to disable."
    )
    # --- END OF ADDITION ---
    
    args = parser.parse_args()
        
    print("\nInitializing VLA Data Generator...")
    generator = VLADataGenerator(output_dir=args.output_dir)
    
    print("Running generation loop...")
    generator.run_generation_loop(num_trials=args.num_trials, num_videos=args.num_videos)
