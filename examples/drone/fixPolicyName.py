import pickle

# Load the config
with open("logs/drone-demo/cfgs.pkl", "rb") as f:
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

# # Add the missing key
# train_cfg["algorithm"]["class_name"] = "PPO"  # <-- Required

# Patch missing key
train_cfg["num_steps_per_env"] = 100  # ← Required by OnPolicyRunner
# Optionally, double-check and print to confirm
print("num_steps_per_env:", train_cfg.get("num_steps_per_env"))

train_cfg["save_interval"] = 100
train_cfg["empirical_normalization"] = None
train_cfg["seed"] = 1
        
# Save it back
with open("logs/drone-demo/cfgs.pkl", "wb") as f:
    pickle.dump((env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg), f)
