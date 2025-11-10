#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
import argparse
from glob import glob
from scipy.spatial.transform import Rotation as R # <-- ADD THIS IMPORT

def print_directory_structure(root_dir, num_episodes=3):
    """Prints a simple tree view of the dataset directory."""
    print(f"Inspecting directory structure for: {root_dir}")
    print(f"{os.path.basename(root_dir)}/")
    
    episode_paths = sorted(glob(os.path.join(root_dir, "episode_*")))
    if not episode_paths:
        print("  (No 'episode_*' folders found.)")
        return

    for i, episode_path in enumerate(episode_paths):
        if i >= num_episodes:
            print(f"  ... (and {len(episode_paths) - num_episodes} more episodes)")
            break
            
        print(f"  ├── {os.path.basename(episode_path)}/")
        
        files = sorted(os.listdir(episode_path))
        for j, file_name in enumerate(files):
            if j == len(files) - 1:
                print(f"  │   └── {file_name}")
            else:
                print(f"  │   ├── {file_name}")
    print("-" * 40) # Separator

def project_world_to_pixel(P_world, T_wc, K, W, H):
    """
    Projects a 3D world point into 2D pixel coordinates.
    (FIXED: Removed the incorrect double-flip)
    """
    
    # 1. Explicitly check for invalid input
    if not np.isfinite(P_world).all():
        print(f"  Warning: Skipping invalid waypoint (NaN/inf): {P_world}")
        return None
    
    # 2. Get T_cw (Camera-from-World) by inverting T_wc
    try:
        if not np.isfinite(T_wc).all():
            print("  Warning: Skipping due to invalid T_wc matrix (NaN/inf).")
            return None
        T_cw = np.linalg.inv(T_wc)
    except np.linalg.LinAlgError:
        print("Error: Could not invert extrinsic matrix T_wc.")
        return None

    # 3. Convert world point to homogeneous coordinates
    P_world_h = np.array([P_world[0], P_world[1], P_world[2], 1.0])
    
    # 4. Transform to camera frame
    P_cam_h = np.dot(T_cw, P_world_h)
    P_cam = P_cam_h[:3]
    
    # 5. Explicitly check for invalid transform result
    if not np.isfinite(P_cam).all():
        print(f"  Warning: Skipping waypoint, invalid camera coordinates: {P_cam}")
        return None
    
    z_c = P_cam[2]
    
    # 6. Check if point is behind camera (use a small epsilon)
    if z_c <= 1e-6:
        return None
        
    # 7. Project to image plane (homogeneous)
    P_image_h = np.dot(K, P_cam)
    
    # 8. Normalize (perspective divide)
    u = P_image_h[0] / P_image_h[2]
    v = P_image_h[1] / P_image_h[2]
    
    # --- 9. FLIP LOGIC REMOVED ---
    # The image is already flipped, and the projection (K @ P_cam)
    # also assumes a top-left origin. No flip is needed.
    
    # 10. Check if pixel is within image bounds
    if not (0 <= u < W and 0 <= v < H): # <-- MODIFIED
        return None
        
    return int(round(u)), int(round(v)) # <-- MODIFIED


def visualize_dataset(dataset_dir, output_dir, num_samples=5):
    """
    Loads and visualizes the first N samples from a VLA dataset directory.
    (MODIFIED: Draws 3D coordinate frames for affordances)
    """
    
    # --- Call the directory structure printer ---
    print_directory_structure(dataset_dir, num_episodes=3)
    
    # 1. Find all episode directories, sorted by name
    episode_paths = sorted(glob(os.path.join(dataset_dir, "episode_*")))
    
    if not episode_paths:
        print(f"Error: No episodes found in '{dataset_dir}'.")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")

    # 2. Limit to the number of samples we want to see
    num_to_show = min(num_samples, len(episode_paths))
    print(f"Found {len(episode_paths)} episodes. Visualizing the first {num_to_show}...")

    # --- 3. Colors for 3D Axes (R-G-B for X-Y-Z) ---
    AXIS_COLORS = {
        "x": (0, 0, 255),  # Red
        "y": (0, 255, 0),  # Green
        "z": (255, 0, 0),  # Blue
    }
    AXIS_LENGTH = 0.05 # 5cm long axes

    for i, episode_path in enumerate(episode_paths[:num_to_show]):
        print(f"\n{'='*30} VISUALIZING EPISODE {i} {'='*30}")
        print(f"Path: {episode_path}")

        # 4. Load metadata.json
        metadata_path = os.path.join(episode_path, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"  Error: metadata.json not found in {episode_path}")
            continue
        
        # 5. Load image_rgb.png
        image_path = os.path.join(episode_path, "image_rgb.png")
        try:
            vis_image = cv2.imread(image_path)
            if vis_image is None:
                raise FileNotFoundError(f"Image file not found or is empty: {image_path}")
            H, W = vis_image.shape[:2]
        except Exception as e:
            print(f"  Error: Could not load image_rgb.png: {e}")
            continue
            
        # 6. Print Instruction
        instruction = metadata.get("instruction", "No instruction found.")
        print(f"\n  INSTRUCTION:\n  \"{instruction}\"")

        # 7. Load camera params and waypoints
        waypoints = metadata.get("waypoints", {})
        try:
            K = np.array(metadata["camera_intrinsics"])
            T_wc = np.array(metadata["camera_extrinsics_wc"])
        except KeyError:
            print("  Error: Camera matrices not found in metadata.json.")
            print("  Please re-generate your dataset with the modified script.")
            continue

        print(f"\n  SAVED 7-DOF WAYPOINTS (x, y, z, roll, pitch, yaw, gripper):")
        
        # --- 8. (MODIFIED) Project, Draw 3D Axes, and Print Waypoints ---
        for name, pose in waypoints.items():
            pose_str = [f"{p: .3f}" for p in pose]
            print(f"  - {name+':':<12} [{', '.join(pose_str)}]")
            
            # We only draw A1, A2, A3
            if name in ["A1_pregrasp", "A2_grasp", "A3_release"]:
                
                # Get pose components
                P_world_origin = np.array(pose[:3])
                rpy = pose[3:6] # (roll, pitch, yaw)
                
                # Get the rotation object
                R_obj = R.from_euler('xyz', rpy, degrees=False)
                
                # Define local axis endpoints
                P_local_x = np.array([AXIS_LENGTH, 0, 0])
                P_local_y = np.array([0, AXIS_LENGTH, 0])
                P_local_z = np.array([0, 0, AXIS_LENGTH])
                
                # Transform axis endpoints to world frame
                P_world_x = R_obj.apply(P_local_x) + P_world_origin
                P_world_y = R_obj.apply(P_local_y) + P_world_origin
                P_world_z = R_obj.apply(P_local_z) + P_world_origin
                
                # Project all 4 points to pixel coordinates
                pix_origin = project_world_to_pixel(P_world_origin, T_wc, K, W, H)
                pix_x = project_world_to_pixel(P_world_x, T_wc, K, W, H)
                pix_y = project_world_to_pixel(P_world_y, T_wc, K, W, H)
                pix_z = project_world_to_pixel(P_world_z, T_wc, K, W, H)

                # If all points are valid, draw the lines
                if all(p is not None for p in [pix_origin, pix_x, pix_y, pix_z]):
                    # Draw X-axis (Red)
                    cv2.line(vis_image, pix_origin, pix_x, AXIS_COLORS["x"], 2)
                    # Draw Y-axis (Green)
                    cv2.line(vis_image, pix_origin, pix_y, AXIS_COLORS["y"], 2)
                    # Draw Z-axis (Blue)
                    cv2.line(vis_image, pix_origin, pix_z, AXIS_COLORS["z"], 2)
                    
                    # Draw the label (A1, A2, A3)
                    cv2.putText(vis_image, name[:2], (pix_origin[0] + 5, pix_origin[1] + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                else:
                    print(f"    > Note: Waypoint {name} is partially outside camera view.")

        # 9. Save the visualization
        output_filename = os.path.join(output_dir, f"vis_episode_{i:05d}.png")
        
        # Add instruction text to the image
        (text_w, text_h), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis_image, (0, H - text_h - 10), (W, H), (0,0,0), -1)
        cv2.putText(vis_image, instruction, (5, H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    
        cv2.imwrite(output_filename, vis_image)
        print(f"  > Saved visualization to {output_filename}")

    
    print(f"\n{'='*74}")
    print("Visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize samples from the generated OpenVLA dataset."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="~/my_openvla_dataset",
        help="Directory where the dataset was saved (same as --output_dir for generator)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/my_openvla_dataset_VISUALS",
        help="Directory to save the visualization images"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize from the beginning of the dataset"
    )
    args = parser.parse_args()
    
    # Expand user paths
    dataset_dir_expanded = os.path.expanduser(args.dataset_dir)
    output_dir_expanded = os.path.expanduser(args.output_dir)
    
    visualize_dataset(dataset_dir_expanded, output_dir_expanded, args.num_samples)
