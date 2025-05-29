#!/usr/bin/env python3
"""
Test script for VLM-integrated MultiHoverEnv.
This script demonstrates how to use the modified environment with VLM waypoint planning.
"""

import os
import numpy as np
from multi_hover_env_with_vlm import MultiHoverEnv

def main():
    # Define target trajectory (list of 3D coordinates)
    target_trajectory = [np.random.rand(3) for _ in range(5)]
    
    # Create directories for saving outputs
    base_dir = "vlm_test_outputs"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "pictures"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "states"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "videos"), exist_ok=True)
    
    # Initialize environment with VLM integration
    env = MultiHoverEnv(
        target_trajectory=target_trajectory,
        n_drones=3,  # Same number as targets
        show_viewer=True,
        snapshots=True,
        picture_save_path=os.path.join(base_dir, "pictures"),
        video_save_path=os.path.join(base_dir, "videos", "vlm_flight.mp4"),
        state_params_save_path=os.path.join(base_dir, "states"),
        rl_checkpoint_path="path/to/your/rl_checkpoint",  # Update this path
        ckpt=100,  # Update this checkpoint number
        use_vlm=True,
        vlm_replan_interval=5,  # Call VLM every 5 steps
        min_distance=0.3,  # Minimum safe distance between drones
        target_threshold=0.1,  # Distance threshold to consider target reached
    )
    
    print("Starting VLM-guided drone simulation...")
    print(f"Number of drones: {env.n_drones}")
    print(f"Number of targets: {len(target_trajectory)}")
    print(f"VLM replanning interval: {env.vlm_replan_interval} steps")
    
    # Run the simulation
    env.run()
    
    print("Simulation completed!")
    print(f"Results saved in: {base_dir}")

if __name__ == "__main__":
    main() 