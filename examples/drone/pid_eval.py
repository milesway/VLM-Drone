import argparse
import json
import os
import random
from pid_env import PidEnv

def generate_random_targets(n, x_range, y_range, z_range):
    " Generate n random targets "
    return [
        (
            round(random.uniform(*x_range), 2),
            round(random.uniform(*y_range), 2),
            round(random.uniform(*z_range), 2)
        )
        for _ in range(n)
    ]

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--record", action="store_true", help="Enable video recording.")
    parser.add_argument("--video_save_path", type=str, default="./videos/pid_flight.mp4")
    parser.add_argument("--state_params_save_path", type=str, default="./logs/multi_drone_flight")
    parser.add_argument(
        "--pid_params",
        type=str,
        default=None,
        help="JSON string for PID params, 9x3 format. E.g. '[[2,0,0],[2,0,0],...]'",
    )
    parser.add_argument("--n_drones", type=int, default=3)
    parser.add_argument("--x_range", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--y_range", type=float, nargs=2, default=[-1.0, 1.0])
    parser.add_argument("--z_range", type=float, nargs=2, default=[0, 1.5])
    parser.add_argument("--snapshots", type=bool, default=True)
    parser.add_argument("--snap_interval", type=int, default=100)
    args = parser.parse_args()

    # Randomly generate sequence of target points (visualized as red balls) 
    targets = generate_random_targets(args.n_drones, args.x_range, args.y_range, args.z_range)
    pid_params = json.loads(args.pid_params) if args.pid_params else None 
    
    os.makedirs(args.state_params_save_path, exist_ok=True)

    # Save the parameters in a .json file for each time step
    env = PidEnv(
        target_trajectory=targets,
        pid_params=pid_params,
        n_drones=args.n_drones,
        show_viewer=True,
        snapshots=args.snapshots,
        snap_interval=args.snap_interval,
    )

    env.run(
        save_video_path=args.video_save_path,
        state_params_save_path=args.state_params_save_path
    )


if __name__ == "__main__":
    main()

