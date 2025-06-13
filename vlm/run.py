# System imports
import os
import numpy as np
import argparse

# Local imports
from multi_hover_env_with_vlm import MultiHoverEnv

def generate_targets_in_range(n_targets, x_range, y_range, z_range):
    """Generate n_targets within specified XYZ ranges."""
    return [
        [
            round(np.random.uniform(*x_range), 3),
            round(np.random.uniform(*y_range), 3),
            round(np.random.uniform(*z_range), 3),
        ]
        for _ in range(n_targets)
    ]
    
def main(args):
    
    # Create directories for saving outputs
    base_dir = args.save_path
    picture_dir = os.path.join(base_dir, "pictures")
    state_dir = os.path.join(base_dir, "states")
    video_dir = os.path.join(base_dir, "videos")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(picture_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)


    if args.env_name == "plain":
        # Define target trajectory (list of 3D coordinates)
        target_trajectory = generate_targets_in_range(
            args.n_drones,
            x_range=args.x_range,
            y_range=args.y_range,
            z_range=args.z_range
        )
    elif args.env_name == "warehouse":
        # TODO: Specify target range based on args.target_range
        # target_trajectory = [np.random.rand(3) for _ in range(args.n_drones)]
        target_trajectory = generate_targets_in_range(
            args.n_drones,
            x_range=args.x_range,
            y_range=args.y_range,
            z_range=args.z_range
        )


    else:
        raise ValueError(f"Invalid environment name: {args.env_name}")
    
    
    # Initialize environment with VLM integration
    env = MultiHoverEnv(
        picture_save_path=picture_dir,
        video_save_path=video_dir,
        state_params_save_path=state_dir,
        target_trajectory=target_trajectory,
        env_name=args.env_name,
        save_path=args.save_path,
        ckpt=args.ckpt,
        n_drones=args.n_drones,
        llm_replan_interval=args.llm_replan_interval,
        use_vision=args.use_vision,
        model_name=args.model_name,
        min_distance=args.min_safe_distance,
        target_threshold=args.target_threshold,
        collision_llm_interval=args.collision_llm_interval,
        enable_ray_tracing=args.enable_ray_tracing,
        show_viewer=args.show_viewer
    )
    
    # Run the simulation
    env.run()
    


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--env_name", type=str, default="plain", choices=["plain", "warehouse"])
    parser.add_argument("--save_path", type=str, default="./outputs")
    # Target generation ranges (default for warehouse mode)
    parser.add_argument("--x_range", type=float, nargs=2, default=[-2.5, 0.0], help="Range for x-axis target positions")
    parser.add_argument("--y_range", type=float, nargs=2, default=[2.3, 3.5], help="Range for y-axis target positions")
    parser.add_argument("--z_range", type=float, nargs=2, default=[1.0, 2.2], help="Range for z-axis target positions")

    # Drone
    parser.add_argument("--ckpt", type=str, default="./checkpoints/rl_model")
    parser.add_argument("--n_drones", type=int, default=3)
    # LLM
    parser.add_argument("--llm_replan_interval", type=int, default=5)
    parser.add_argument("--use_vision", action="store_true", help="Use vision for LLM input (default: False)")
    parser.add_argument("--model_name", type=str, default="gpt-4o", choices=["o4-mini", "gpt-4o", "gemini-2.5-pro"])    
    # Safety
    parser.add_argument("--min_safe_distance", type=float, default=0.3)
    parser.add_argument("--target_threshold", type=float, default=0.1)
    parser.add_argument("--collision_llm_interval", type=int, default=20)
    # Viewer
    parser.add_argument("--show_viewer", action="store_true", help="Enable the viewer (default: True)")
    parser.add_argument("--enable_ray_tracing", action="store_true", help="Enable ray tracing (default: False)")
    parser.set_defaults(use_vision=False, show_viewer=True, enable_ray_tracing=False)

    args = parser.parse_args()
    
    # print args
    print("Initializing VLM-integrated drone simulation...")
    
    print("--------------Environment -----------------------")
    print(f"Environment name: {args.env_name}")
    print(f"Save path: {args.save_path}")
    
    print("--------------Drone -----------------------")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Number of drones: {args.n_drones}")
    
    print("--------------LLM -----------------------")
    print(f"LLM replanning interval: {args.llm_replan_interval} steps")
    print(f"Model name: {args.model_name}")
    print(f"Use vision: {args.use_vision}")
    
    print("--------------Safety -----------------------")
    print(f"Minimum safe distance: {args.min_safe_distance} meters")
    print(f"Target threshold: {args.target_threshold} meters")
    print(f"Collision LLM interval: {args.collision_llm_interval} steps")
    
    print("--------------Viewer -----------------------")
    print(f"Show viewer: {args.show_viewer}")
    print(f"Enable ray tracing: {args.enable_ray_tracing}")
    
    print("------------------------------------------------------")

    main(args) 