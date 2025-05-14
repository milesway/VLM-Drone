import argparse
import json
from pid_env import PidEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Enable video recording.")
    parser.add_argument("--save_path", type=str, default="../../../videos/pid_flight.mp4")
    parser.add_argument(
        "--pid_params",
        type=str,
        default=None,
        help="JSON string for PID params, 9x3 format. E.g. '[[2,0,0],[2,0,0],...]'",
    )
    args = parser.parse_args()

    trajectory = [(1.0, 1.0, 2.0)]

    if args.pid_params:
        pid_params = json.loads(args.pid_params)
    else:
        pid_params = None

    env = PidEnv(
        target_trajectory=trajectory,
        pid_params=pid_params,
        show_viewer=True,
        record=args.record,
    )

    env.run(save_video_path=args.save_path)


if __name__ == "__main__":
    main()

