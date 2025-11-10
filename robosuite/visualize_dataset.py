#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob

def visualize_dataset(dataset_dir, num_samples=5):
    """
    Loads and visualizes the first N samples from a VLA dataset directory.
    
    This will:
    1. Print the instruction and 7-DOF waypoints to the console.
    2. Show the saved RGB image in a Matplotlib window.
    """
    
    # 1. Find all episode directories, sorted by name
    episode_paths = sorted(glob(os.path.join(dataset_dir, "episode_*")))
    
    if not episode_paths:
        print(f"Error: No episodes found in '{dataset_dir}'.")
        print("Please check the --dataset_dir path and ensure you have")
        print("run the 'generate_vla_dataset.py' script successfully.")
        return

    # 2. Limit to the number of samples we want to see
    num_to_show = min(num_samples, len(episode_paths))
    print(f"Found {len(episode_paths)} episodes. Visualizing the first {num_to_show}...")

    for i, episode_path in enumerate(episode_paths[:num_to_show]):
        print(f"\n{'='*30} VISUALIZING EPISODE {i} {'='*30}")
        print(f"Path: {episode_path}")

        # 3. Load metadata.json
        metadata_path = os.path.join(episode_path, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"  Error: metadata.json not found in {episode_path}")
            continue
        
        # 4. Load image_rgb.png
        image_path = os.path.join(episode_path, "image_rgb.png")
        try:
            # We use cv2.imread and convert to match the generator's color space
            bgr_image = cv2.imread(image_path)
            if bgr_image is None:
                raise FileNotFoundError(f"Image file not found or is empty: {image_path}")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"  Error: Could not load image_rgb.png: {e}")
            continue
            
        # 5. Print Instruction
        instruction = metadata.get("instruction", "No instruction found.")
        print(f"\n  INSTRUCTION:\n  \"{instruction}\"")

        # 6. Print Waypoints (the 7-DOF poses)
        waypoints = metadata.get("waypoints", {})
        print(f"\n  SAVED 7-DOF WAYPOINTS (x, y, z, roll, pitch, yaw, gripper):")
        if not waypoints:
            print("  No waypoints found in metadata.")
            continue

        for name, pose in waypoints.items():
            # Format the floats for clean printing (pose[3:6] are radians)
            pose_str = [f"{p: .3f}" for p in pose]
            print(f"  - {name+':':<12} [{', '.join(pose_str)}]")
            
        # 7. Show the image in a Matplotlib window
        plt.figure(figsize=(8, 8))
        # Wrap the title text for readability if it's long
        wrapped_title = "\n".join(instruction[j:j+60] for j in range(0, len(instruction), 60))
        plt.title(f"Episode {i}: {wrapped_title}", fontsize=10)
        plt.imshow(rgb_image)
        plt.axis('off')
    
    print(f"\n{'='*74}")
    print("Displaying images. Close the Matplotlib windows to exit the script.")
    plt.show() # This will show all figures at once

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
    
    # parser.add_group  <--- THIS WAS THE BAD LINE, REMOVE IT
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize from the beginning of the dataset"
    )
    args = parser.parse_args()
    
    # Expand the user path (e.g., convert ~ to /home/user)
    dataset_dir_expanded = os.path.expanduser(args.dataset_dir)
    
    visualize_dataset(dataset_dir_expanded, args.num_samples)
