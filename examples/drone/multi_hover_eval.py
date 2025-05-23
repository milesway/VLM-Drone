import argparse
import os
import json
import pickle
import torch
import random
import genesis as gs
from rsl_rl.runners import OnPolicyRunner
from multi_hover_env import MultiHoverEnv

def generate_random_targets(n, x_range, y_range, z_range):
    """ Generate n random 3D target points."""
    return [
        (
            round(random.uniform(*x_range), 2),
            round(random.uniform(*y_range), 2),
            round(random.uniform(*z_range), 2),
        )
        for _ in range(n)
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="drone-demo")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--n_drones", type=int, default=3)
    parser.add_argument("--x_range", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--y_range", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--z_range", type=float, nargs=2, default=[0.5, 1.5])
    parser.add_argument("--video_save_path", type=str, default="./videos/rl_flight.mp4")
    parser.add_argument("--picture_save_path", type=str, default="./pictures/")
    parser.add_argument("--state_params_save_path", type=str, default="./logs/rl_multi_drone")
    parser.add_argument("--snapshots", type=bool, default=True)
    parser.add_argument("--snap_interval", type=int, default=100)
    # parser.add_argument("--rl_checkpoint_path", type=str, default=f"./logs/{args.exp_name}")
    args = parser.parse_args()

    # Generate targets and initialize env
    targets = generate_random_targets(args.n_drones, args.x_range, args.y_range, args.z_range)
    os.makedirs(args.state_params_save_path, exist_ok=True)
    os.makedirs(args.picture_save_path, exist_ok=True)

    env = MultiHoverEnv(
        target_trajectory=targets,
        n_drones=args.n_drones,
        snapshots=args.snapshots,
        snap_interval=args.snap_interval,
        picture_save_path=args.picture_save_path,
        video_save_path=args.video_save_path,
        state_params_save_path=args.state_params_save_path,
        rl_checkpoint_path=f"./logs/{args.exp_name}", # Load RL config file
        ckpt=args.ckpt, # Load trained RL model
        # policy=policy,
        # device=gs.device,
        show_viewer=True,
    )

    env.run() # Run the simulation, state parameters saved in .json files for each time step

if __name__ == "__main__":
    main()
