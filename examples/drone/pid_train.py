import os
import time
from pid_env import PidEnv
import numpy as np


def evaluate_pid(pid_params, trajectory, record=False, save_video=False, save_index=0):
    # Initialize environment with specified PID gains
    env = PidEnv(
        target_trajectory=trajectory,
        pid_params=pid_params,
        show_viewer=True,
        record=record,
    )

    if save_video:
        save_path = f"../videos/pid_tune_{save_index}.mp4"
    else:
        save_path = None

    # Run and time the flight
    start_time = time.time()
    env.run(save_video_path=save_path)
    elapsed = time.time() - start_time

    # Compute final error to target
    final_pos = env.drone.get_pos().cpu().numpy()
    final_error = np.linalg.norm(np.array(trajectory[-1]) - final_pos)

    return elapsed, final_error


def main():
    # Define a trajectory
    trajectory = [(0.3, 0.2, 0.5)]

    # Candidate PID configs to try (e.g., change z controller)
    candidate_pid_sets = [
        # Format: pos_xyz, vel_xyz, att_rpy
        [[2,0,0], [2,0,0], [2,0,0],
         [15,0,15], [15,0,15], [20,0,20],
         [5,0,1], [5,0,1], [1,0,0.1]],

        [[2,0,0], [2,0,0], [2.5,0,0],
         [15,0,15], [15,0,15], [25,0,25],
         [5,0,1], [5,0,1], [1,0,0.1]],

        [[2,0,0], [2,0,0], [3.0,0,0],
         [15,0,15], [15,0,15], [30,0,30],
         [5,0,1], [5,0,1], [1,0,0.1]],
    ]

    print(f"Running {len(candidate_pid_sets)} PID configs...\n")
    os.makedirs("../videos", exist_ok=True)

    for i, pid_params in enumerate(candidate_pid_sets):
        print(f"→ [Trial {i}] Testing PID Z gain: {pid_params[2][0]} | Vel_Z: {pid_params[5][0]}")
        elapsed, final_error = evaluate_pid(pid_params, trajectory, record=False, save_video=False, save_index=i)
        print(f"    Time: {elapsed:.2f}s, Final error: {final_error:.4f}\n")


if __name__ == "__main__":
    main()
