import argparse
import os
import pickle
import shutil
from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from terrain_env import TerrainEnv

def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
    }
    reward_cfg = {"reward_scales": {}}  # Add actual rewards as needed
    command_cfg = {"num_commands": 3}
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def get_train_cfg(exp_name, max_iterations):
    return {
        "algorithm": {"class_name": "PPO", "learning_rate": 0.001},
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [128, 128], "critic_hidden_dims": [128, 128]},
        "runner": {"experiment_name": exp_name, "max_iterations": max_iterations},
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "seed": 1,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-terrain")
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init()
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    os.makedirs(f"logs/{args.exp_name}", exist_ok=True)
    with open(f"logs/{args.exp_name}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    env = TerrainEnv(num_envs=4096, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)
    runner = OnPolicyRunner(env, train_cfg, f"logs/{args.exp_name}", device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()
