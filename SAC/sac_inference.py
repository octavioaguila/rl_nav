#!/usr/bin/env python3
# -------------------------------------------------------------
#  Run inference with a SAC policy trained in train_bunker.py
# -------------------------------------------------------------
import os
import sys
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_bunker_env import BunkerEnv
from gymnasium.wrappers import TimeLimit
from feature_extractor import FeatureExtractor

# Configuration
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Variables
run_name = "run_1"
max_goal_sampling_distance = 8.0
env_name = "val/world_99_easy"

# Paths
xml  = os.path.join(root, "assets", "worlds", f"{env_name}.xml")
ckpt = os.path.join(root, "SAC", "log", run_name, "best_model", "best_model.zip")

# Environment Setup
env = DummyVecEnv([lambda: TimeLimit(BunkerEnv(xml_path=xml, render_mode="human", max_goal_sampling_distance=max_goal_sampling_distance), 
                                                max_episode_steps=600)])

model: SAC = SAC.load(ckpt, env, device="auto")
n_episodes      = 200
deterministic   = True
max_ep_len      = 600

# Trackers
successful_episodes = 0
crashed_episodes = 0
truncated_episodes = 0
total_episodes = 0

episode_lengths = []

print(f"Starting Inference: {n_episodes} episodes...")

for ep in range(n_episodes):
    obs = env.reset()
    
    # Run episode
    for step in range(max_ep_len):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, info = env.step(action)

        # Check for done conditions
        # terminated[0] corresponds to 'dones' in SB3 VecEnv (True if Terminated or Truncated)
        if terminated[0] or info[0].get("TimeLimit.truncated", False):
            break

    # Outcome Logic
    steps_taken = step + 1
    episode_lengths.append(steps_taken)

    # Determine outcome priority: Success > Crash > Timeout
    # info[0] contains the info dict for the last step (or the transition that ended the episode)
    if info[0].get("is_success", False):
        successful_episodes += 1
        outcome = "SUCCESS"
    elif info[0].get("collision", False):
        crashed_episodes += 1
        outcome = "CRASH"
    else:
        # If not success and not collision, it's a timeout
        truncated_episodes += 1
        outcome = "TIMEOUT"
    
    total_episodes += 1
    print(f"[EP {ep+1}/{n_episodes}] {outcome} | Steps: {steps_taken}")

# Final Stats Calculations
stats_output = []
stats_output.append(f"\n{'='*60}")
stats_output.append(f"INFERENCE RESULTS: {run_name} on {env_name}")
stats_output.append(f"Training parameters: Max goal sampling distance: {max_goal_sampling_distance}")
stats_output.append(f"{'='*60}")
stats_output.append(f"Success Rate:      {successful_episodes/total_episodes*100:.1f}% ({successful_episodes}/{total_episodes})")
stats_output.append(f"Collision Rate:    {crashed_episodes/total_episodes*100:.1f}% ({crashed_episodes}/{total_episodes})")
stats_output.append(f"Timeout Rate:      {truncated_episodes/total_episodes*100:.1f}% ({truncated_episodes}/{total_episodes})")
stats_output.append(f"Mean Episode Len:  {np.mean(episode_lengths):.1f} steps")
stats_output.append(f"{'='*60}")

stats_str = "\n".join(stats_output)
print(stats_str)

# Save Results
results_dir = os.path.join(root, "SAC", "inference", "results")
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f"{run_name}_{env_name}_metrics_analysis.txt")
with open(results_file, "w") as f:
    f.write(stats_str)

env.close()