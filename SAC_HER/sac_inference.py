#!/usr/bin/env python3
# -------------------------------------------------------------
#  Run inference with a SAC_HER policy trained in train_bunker.py
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
env_name = "test/world_17_easy"

# Paths
xml  = os.path.join(root, "assets", "worlds", f"{env_name}.xml")
ckpt = os.path.join(root, "SAC_HER", "log", run_name, "best_model", "best_model.zip")

# Environment Setup
env = DummyVecEnv([lambda: TimeLimit(BunkerEnv(xml_path=xml, render_mode="human", max_goal_sampling_distance=max_goal_sampling_distance), 
                                                max_episode_steps=600)])

# Get env constants for observation decoding
raw_env = env.envs[0].unwrapped
n_lidar = raw_env.n_lidar
lidar_max_range = raw_env.lidar_max_range
dt = raw_env.model.opt.timestep * raw_env.frame_skip

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
angular_path_smoothness_list = []
linear_path_smoothness_list = []
spl_list = []
avg_clearance_list = []

print(f"Starting Inference: {n_episodes} episodes...")

for ep in range(n_episodes):
    obs = env.reset()
    
    # Episode-specific trackers
    ep_path_length = 0
    ep_angular_smoothness = 0
    ep_linear_smoothness = 0
    ep_clearance = []
    last_pos = None
    last_ang_vel = 0
    last_linear_vel = 0
    
    # Calculate initial distance for SPL
    # In SAC_HER, we get 'achieved_goal' and 'desired_goal' in obs
    # obs is a dict-like from VecEnv (stacked arrays)
    ag = obs['achieved_goal'][0]
    dg = obs['desired_goal'][0]
    initial_dist = np.linalg.norm(ag[:2] - dg[:2])

    # Run episode
    for step in range(max_ep_len):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, info = env.step(action)
        
        # Metric Extraction
        inf = info[0]
        
        # SAC_HER: Dict observation. LiDAR is first part of 'observation' key.
        # obs['observation'] has shape (1, dim).
        lidar_colum = obs['observation'][0, :n_lidar*3]
        lidar_data = lidar_colum.reshape(-1, 3)
        d_norm = lidar_data[:, 2] # 3rd column is normalized distance
        # d_norm is in [-1, 1], map back to [0, lidar_max_range]
        d_real = (d_norm + 1.0) / 2.0 * lidar_max_range
        min_scan = np.min(d_real)

        # Get velocities from environment
        curr_v, curr_ang_vel = raw_env.get_robot_velocities()
        curr_v = curr_v[0]
        curr_ang_vel = curr_ang_vel[2]

        # Approximate distance traveled in this step
        step_dist = abs(curr_v) * dt
        ep_path_length += step_dist

        # Smoothness
        ep_angular_smoothness += ((curr_ang_vel - last_ang_vel)/ dt)**2
        ep_linear_smoothness += ((curr_v - last_linear_vel) / dt)**2
        last_ang_vel = curr_ang_vel
        last_linear_vel = curr_v

        # Clearance
        ep_clearance.append(min_scan)

        if terminated[0] or inf.get("TimeLimit.truncated", False):
            break

    # Outcome Logic
    steps_taken = step + 1
    episode_lengths.append(steps_taken)
    
    # Calculate SPL for this episode
    is_success = inf.get("is_success", False)
    if is_success:
        successful_episodes += 1
        outcome = "SUCCESS"
        # SPL formula: S * (d_init / max(d_init, path_length))
        spl = initial_dist / max(initial_dist, ep_path_length)
    elif inf.get("collision", False):
        crashed_episodes += 1
        outcome = "CRASH"
        spl = 0
    else:
        truncated_episodes += 1
        outcome = "TIMEOUT"
        spl = 0

    spl_list.append(spl)
    angular_smoothness = ep_angular_smoothness / steps_taken
    linear_smoothness = ep_linear_smoothness / steps_taken
    angular_path_smoothness_list.append(angular_smoothness)
    linear_path_smoothness_list.append(linear_smoothness)
    avg_clearance_list.append(np.mean(ep_clearance))
    
    total_episodes += 1
    print(f"[EP {ep+1}/{n_episodes}] {outcome} | Steps: {steps_taken} | Ang. Smoothness: {angular_smoothness:.4f} Lin. Smoothness: {linear_smoothness}")

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
stats_output.append(f"-"*30)
stats_output.append(f"SPL:               {np.mean(spl_list):.4f}")
stats_output.append(f"Mean Ang. Smooth:  {np.mean(angular_path_smoothness_list):.6f}")
stats_output.append(f"Mean Lin. Smooth:  {np.mean(linear_path_smoothness_list):.6f}")
stats_output.append(f"Mean Clearance:    {np.mean(avg_clearance_list):.4f} m")
stats_output.append(f"{'='*60}")

stats_str = "\n".join(stats_output)
print(stats_str)

# Save Results
results_dir = os.path.join(root, "SAC_HER", "inference_results")
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f"{run_name}_{env_name.replace('/', '_')}_metrics_analysis.txt")
with open(results_file, "w") as f:
    f.write(stats_str)

env.close()