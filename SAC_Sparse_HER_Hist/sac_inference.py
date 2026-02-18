#!/usr/bin/env python3
# -------------------------------------------------------------
#  Run inference with a SAC-Sparse (HER) + Hist policy trained in train_bunker.py
#  Method of Batch Means: 20 batches × 10 episodes = 200 total
# -------------------------------------------------------------
import os
import sys
import time
import numpy as np
from scipy import stats
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
k_history_window = 10

# Paths
xml  = os.path.join(root, "assets", "worlds", f"{env_name}.xml")
ckpt = os.path.join(root, "SAC_Sparse_HER_Hist", "log", run_name, "best_model", "best_model.zip")

# Environment Setup
env = DummyVecEnv([lambda: TimeLimit(BunkerEnv(xml_path=xml, render_mode="human", max_goal_sampling_distance=max_goal_sampling_distance, k_history_window=k_history_window), 
                                                max_episode_steps=600)])

# Get env constants for observation decoding
raw_env = env.envs[0].unwrapped
n_lidar = raw_env.n_lidar
lidar_max_range = raw_env.lidar_max_range
dt = raw_env.model.opt.timestep * raw_env.frame_skip

model: SAC = SAC.load(ckpt, env, device="auto")
deterministic   = True
max_ep_len      = 600

# ── Batch Means Configuration ────────────────────────────────
n_batches       = 20
eps_per_batch   = 10
n_episodes      = n_batches * eps_per_batch  # 200 total

# ── CI helper (applied to the 20 batch means) ────────────────
def batch_ci95(batch_means):
    """Return (grand_mean, ci_half_width) from a list of batch means."""
    arr = np.array(batch_means, dtype=float)
    n = len(arr)
    mean = np.mean(arr)
    if n < 2:
        return mean, 0.0
    sd = np.std(arr, ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean, h

# ── Batch-level accumulators ─────────────────────────────────
batch_success_rates = []
batch_collision_rates = []
batch_timeout_rates = []
batch_ep_lengths = []
batch_spl = []
batch_ang_smooth = []
batch_lin_smooth = []
batch_clearance = []

# Global counters (for raw totals in the report)
total_successes  = 0
total_collisions = 0
total_timeouts   = 0
total_episodes   = 0

print(f"Starting Inference: {n_batches} batches × {eps_per_batch} episodes = {n_episodes} total")

for batch in range(n_batches):
    b_spl = []
    b_ang = []
    b_lin = []
    b_clr = []
    b_len = []
    b_successes = 0
    b_collisions = 0
    b_timeouts = 0

    for ep_in_batch in range(eps_per_batch):
        global_ep = batch * eps_per_batch + ep_in_batch
        obs = env.reset()

        ep_path_length = 0
        ep_angular_smoothness = 0
        ep_linear_smoothness = 0
        ep_clearance = []
        last_ang_vel = 0
        last_linear_vel = 0

        # Calculate initial distance for SPL
        # In SAC-Sparse (HER) + Hist, we get 'achieved_goal' and 'desired_goal' in obs
        ag = obs['achieved_goal'][0]
        dg = obs['desired_goal'][0]
        initial_dist = np.linalg.norm(ag[:2] - dg[:2])

        for step in range(max_ep_len):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, info = env.step(action)

            inf = info[0]

            # SAC-Sparse (HER) + Hist: Dict observation. LiDAR is first part of 'observation' key.
            lidar_colum = obs['observation'][0, :n_lidar*3]
            lidar_data = lidar_colum.reshape(-1, 3)
            d_norm = lidar_data[:, 2]
            d_real = (d_norm + 1.0) / 2.0 * lidar_max_range
            min_scan = np.min(d_real)

            curr_v, curr_ang_vel = raw_env.get_robot_velocities()
            curr_v = curr_v[0]
            curr_ang_vel = curr_ang_vel[2]

            step_dist = abs(curr_v) * dt
            ep_path_length += step_dist

            ep_angular_smoothness += ((curr_ang_vel - last_ang_vel) / dt)**2
            ep_linear_smoothness += ((curr_v - last_linear_vel) / dt)**2
            last_ang_vel = curr_ang_vel
            last_linear_vel = curr_v

            ep_clearance.append(min_scan)

            if terminated[0] or inf.get("TimeLimit.truncated", False):
                break

        steps_taken = step + 1
        is_success = inf.get("is_success", False)
        if is_success:
            b_successes += 1
            outcome = "SUCCESS"
            spl = initial_dist / max(initial_dist, ep_path_length)
        elif inf.get("collision", False):
            b_collisions += 1
            outcome = "CRASH"
            spl = 0
        else:
            b_timeouts += 1
            outcome = "TIMEOUT"
            spl = 0

        ang_smooth = ep_angular_smoothness / steps_taken
        lin_smooth = ep_linear_smoothness / steps_taken

        b_spl.append(spl)
        b_ang.append(ang_smooth)
        b_lin.append(lin_smooth)
        b_clr.append(np.mean(ep_clearance))
        b_len.append(steps_taken)

        print(f"[Batch {batch+1}/{n_batches} | EP {ep_in_batch+1}/{eps_per_batch}] {outcome} "
              f"| Steps: {steps_taken} | Ang: {ang_smooth:.4f} | Lin: {lin_smooth:.4f}")

    # ── Store batch means ─────────────────────────────────────
    batch_success_rates.append(b_successes / eps_per_batch)
    batch_collision_rates.append(b_collisions / eps_per_batch)
    batch_timeout_rates.append(b_timeouts / eps_per_batch)
    batch_ep_lengths.append(np.mean(b_len))
    batch_spl.append(np.mean(b_spl))
    batch_ang_smooth.append(np.mean(b_ang))
    batch_lin_smooth.append(np.mean(b_lin))
    batch_clearance.append(np.mean(b_clr))

    total_successes  += b_successes
    total_collisions += b_collisions
    total_timeouts   += b_timeouts
    total_episodes   += eps_per_batch

    print(f"  >> Batch {batch+1} done — SR: {b_successes/eps_per_batch*100:.0f}%, "
          f"SPL: {np.mean(b_spl):.4f}, Clr: {np.mean(b_clr):.4f}")

# ── Final Stats (Method of Batch Means) ──────────────────────
sr_mean,  sr_ci  = batch_ci95(batch_success_rates)
cr_mean,  cr_ci  = batch_ci95(batch_collision_rates)
tr_mean,  tr_ci  = batch_ci95(batch_timeout_rates)
el_mean,  el_ci  = batch_ci95(batch_ep_lengths)
spl_mean, spl_ci = batch_ci95(batch_spl)
ang_mean, ang_ci = batch_ci95(batch_ang_smooth)
lin_mean, lin_ci = batch_ci95(batch_lin_smooth)
clr_mean, clr_ci = batch_ci95(batch_clearance)

stats_output = []
stats_output.append(f"\n{'='*60}")
stats_output.append(f"INFERENCE RESULTS: {run_name} on {env_name}")
stats_output.append(f"Training parameters:\nMax goal sampling distance: {max_goal_sampling_distance}")
stats_output.append(f"k_history_window: {k_history_window}")
stats_output.append(f"Method of Batch Means: {n_batches} batches × {eps_per_batch} episodes = {n_episodes} total")
stats_output.append(f"95% CI computed over {n_batches} batch means (t-distribution, df={n_batches-1})")
stats_output.append(f"{'='*60}")
stats_output.append(f"Success Rate:      {sr_mean*100:.1f}% ± {sr_ci*100:.1f}% ({total_successes}/{total_episodes})")
stats_output.append(f"Collision Rate:    {cr_mean*100:.1f}% ± {cr_ci*100:.1f}% ({total_collisions}/{total_episodes})")
stats_output.append(f"Timeout Rate:      {tr_mean*100:.1f}% ± {tr_ci*100:.1f}% ({total_timeouts}/{total_episodes})")
stats_output.append(f"Mean Episode Len:  {el_mean:.1f} ± {el_ci:.1f} steps")
stats_output.append(f"-"*30)
stats_output.append(f"SPL:               {spl_mean:.4f} ± {spl_ci:.4f}")
stats_output.append(f"Mean Ang. Smooth:  {ang_mean:.6f} ± {ang_ci:.6f}")
stats_output.append(f"Mean Lin. Smooth:  {lin_mean:.6f} ± {lin_ci:.6f}")
stats_output.append(f"Mean Clearance:    {clr_mean:.4f} ± {clr_ci:.4f} m")
stats_output.append(f"{'='*60}")

stats_str = "\n".join(stats_output)
print(stats_str)

# Save Results
results_dir = os.path.join(root, "SAC_Sparse_HER_Hist", "inference_results")
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f"{run_name}_{env_name.replace('/', '_')}_metrics_analysis.txt")
with open(results_file, "w") as f:
    f.write(stats_str)

env.close()