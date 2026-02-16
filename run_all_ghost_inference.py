#!/usr/bin/env python3
"""
Unified Ghost Trajectory Inference
====================================
Runs all 5 SAC variants with the same initial/goal pose and rendering settings.
Each variant is launched as a separate subprocess to avoid MuJoCo context conflicts.
Saves all ghost trajectory images into a single output folder.

Usage:
    cd /path/to/rl_nav
    python3 run_all_ghost_inference.py
"""
import os
import sys
import json
import subprocess

# ========================== Shared Configuration ==========================
ROOT = os.path.dirname(os.path.abspath(__file__))

run_name = "run_1"
max_goal_sampling_distance = 8.0
env_name = "test/world_16_hard"

# Initial and goal poses: [x, y, theta]
initial_pose = [-0.5, 3.0, 0.0]
goal_pose    = [-2.5, -3.0, 0.0]

# Ghost rendering settings
ghost_subsample     = 15
ghost_alpha         = 1.0
ghost_width         = 1920
ghost_height        = 1080
ghost_dpi           = 300
ghost_temporal_fade = False

# Camera configuration (top-down bird's eye)
ghost_cam_lookat    = [0.0, 0.0, 0.0]
ghost_cam_distance  = 13.0
ghost_cam_azimuth   = 90.0
ghost_cam_elevation = -90.0

deterministic = True
max_ep_len    = 600

# Output folder for all variants
output_dir = os.path.join(ROOT, "ghost_trajectory_results")
os.makedirs(output_dir, exist_ok=True)

# ========================== Variant Definitions ==========================
VARIANTS = [
    {"name": "SAC",          "obs_type": "flat", "env_kwargs": {}},
    {"name": "SAC_HER",      "obs_type": "dict", "env_kwargs": {}},
    {"name": "SAC_SMOOTH",   "obs_type": "flat", "env_kwargs": {}},
    {"name": "SAC_HIST",     "obs_type": "flat", "env_kwargs": {"k_history_window": 10}},
    {"name": "SAC_HER_HIST", "obs_type": "dict", "env_kwargs": {"k_history_window": 10}},
]

# ========================== Worker Script (runs in subprocess) ==========================
WORKER_SCRIPT = r'''
import os, sys, json, csv, numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

cfg = json.loads(sys.argv[1])

root         = cfg["root"]
variant_name = cfg["name"]
obs_type     = cfg["obs_type"]
extra_kwargs = cfg["env_kwargs"]

sys.path.insert(0, os.path.join(root, variant_name))
sys.path.insert(0, root)

from gym_bunker_env import BunkerEnv
from feature_extractor import FeatureExtractor
from lib.ghost_trajectory_renderer import render_ghost_trajectory

xml_path = os.path.join(root, "assets", "worlds", f"{cfg['env_name']}.xml")
ckpt     = os.path.join(root, variant_name, "log", cfg["run_name"], "best_model", "best_model.zip")

if not os.path.exists(ckpt):
    print(f"  [SKIP] Checkpoint not found: {ckpt}")
    sys.exit(0)

env_kwargs = dict(xml_path=xml_path, render_mode=None,
                  max_goal_sampling_distance=cfg["max_goal_sampling_distance"],
                  **extra_kwargs)
env = DummyVecEnv([lambda: TimeLimit(BunkerEnv(**env_kwargs), max_episode_steps=cfg["max_ep_len"])])
raw_env = env.envs[0].unwrapped

model = SAC.load(ckpt, env, device="auto")

initial_pose = cfg["initial_pose"]
goal_pose    = cfg["goal_pose"]

# Velocity limits (same for all variants)
V_MAX = 0.5   # m/s
W_MAX = 0.5   # rad/s

print(f"  Setting manual pose: start={initial_pose}, goal={goal_pose}")
obs = env.reset()
manual_obs = raw_env.set_manual_pose(initial_pose, goal_pose)

if obs_type == "dict":
    obs = {k: np.expand_dims(v, axis=0) for k, v in manual_obs.items()}
else:
    obs = np.expand_dims(manual_obs, axis=0)

episode_qpos = [raw_env.data.qpos.copy()]

# ----- Metric collection -----
# Uniform formulas (independent of variant reward function):
#   v_cmd       = action[0] * V_MAX          [m/s]
#   w_cmd       = action[1] * W_MAX          [rad/s]
#   d_tg        = ||goal_xy - robot_xy||     [m]
#   dtheta_tg   = atan2(sin(yaw-goal_yaw), cos(yaw-goal_yaw))  [rad]
goal_xy  = np.array(goal_pose[:2], dtype=np.float64)
goal_yaw = float(goal_pose[2])

def compute_metrics(qpos, action_raw):
    """Compute metrics from raw qpos and action. Returns dict."""
    robot_xy = qpos[:2].copy()
    qw, qx, qy, qz = qpos[3:7].copy()
    yaw = 2.0 * np.arctan2(qz, qw)

    v_cmd = float(action_raw[0]) * V_MAX
    w_cmd = float(action_raw[1]) * W_MAX
    d_tg  = float(np.linalg.norm(goal_xy - robot_xy))
    angle_diff = yaw - goal_yaw
    dtheta_tg = float(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))

    return {"v_cmd": v_cmd, "w_cmd": w_cmd, "d_tg": d_tg, "dtheta_tg": dtheta_tg}

metrics_log = []  # list of dicts

print(f"  Running episode...")
outcome = "TIMEOUT"
for step in range(cfg["max_ep_len"]):
    action, _ = model.predict(obs, deterministic=cfg["deterministic"])

    # action from DummyVecEnv is (1, 2) — squeeze for metric computation
    action_flat = action.flatten()

    obs, reward, terminated, info = env.step(action)
    episode_qpos.append(raw_env.data.qpos.copy())

    # Compute metrics AFTER step (state reflects result of action)
    m = compute_metrics(raw_env.data.qpos, action_flat)
    m["timestep"] = step + 1
    metrics_log.append(m)

    inf = info[0]
    if terminated[0] or inf.get("TimeLimit.truncated", False):
        is_success = inf.get("is_success", False)
        collision  = inf.get("collision", False)
        outcome = "SUCCESS" if is_success else ("CRASH" if collision else "TIMEOUT")
        print(f"  Episode ended at step {step+1}: {outcome}")
        break

print(f"  Collected {len(episode_qpos)} qpos states, {len(metrics_log)} metric rows")

# ----- Output directory: ghost_trajectory_results/<VARIANT>/ -----
variant_dir = os.path.join(cfg["output_dir"], variant_name)
os.makedirs(variant_dir, exist_ok=True)

# Save ghost trajectory image
ghost_path = os.path.join(variant_dir, f"{cfg['run_name']}_ghost.png")
render_ghost_trajectory(
    model=raw_env.model,
    episode_qpos=episode_qpos,
    output_path=ghost_path,
    width=cfg["ghost_width"],     height=cfg["ghost_height"],
    subsample=cfg["ghost_subsample"], alpha=cfg["ghost_alpha"],
    temporal_fade=cfg["ghost_temporal_fade"],
    cam_lookat=cfg["ghost_cam_lookat"], cam_distance=cfg["ghost_cam_distance"],
    cam_azimuth=cfg["ghost_cam_azimuth"], cam_elevation=cfg["ghost_cam_elevation"],
    dpi=cfg["ghost_dpi"],
)

# Save metrics CSV
csv_path = os.path.join(variant_dir, f"{cfg['run_name']}_metrics.csv")
fieldnames = ["timestep", "v_cmd", "w_cmd", "d_tg", "dtheta_tg"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metrics_log)

print(f"  Saved metrics to: {csv_path}")
print(f"  Episode outcome: {outcome}")

env.close()
'''


def run_variant(variant: dict) -> bool:
    """Run a single variant in a subprocess. Returns True on success."""
    name = variant["name"]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Build config dict to pass to the subprocess
    cfg = {
        "root": ROOT,
        "name": name,
        "obs_type": variant["obs_type"],
        "env_kwargs": variant["env_kwargs"],
        "run_name": run_name,
        "env_name": env_name,
        "max_goal_sampling_distance": max_goal_sampling_distance,
        "initial_pose": initial_pose,
        "goal_pose": goal_pose,
        "max_ep_len": max_ep_len,
        "deterministic": deterministic,
        "output_dir": output_dir,
        "ghost_subsample": ghost_subsample,
        "ghost_alpha": ghost_alpha,
        "ghost_width": ghost_width,
        "ghost_height": ghost_height,
        "ghost_dpi": ghost_dpi,
        "ghost_temporal_fade": ghost_temporal_fade,
        "ghost_cam_lookat": ghost_cam_lookat,
        "ghost_cam_distance": ghost_cam_distance,
        "ghost_cam_azimuth": ghost_cam_azimuth,
        "ghost_cam_elevation": ghost_cam_elevation,
    }

    cfg_json = json.dumps(cfg)

    result = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, cfg_json],
        cwd=ROOT,
    )

    if result.returncode != 0:
        print(f"  [ERROR] {name} exited with code {result.returncode}")
        return False
    return True


# ========================== Main ==========================
if __name__ == "__main__":
    print(f"Ghost Trajectory Inference — All Variants")
    print(f"World: {env_name}")
    print(f"Start: {initial_pose}  →  Goal: {goal_pose}")
    print(f"Output: {output_dir}")

    results = {}
    for variant in VARIANTS:
        success = run_variant(variant)
        results[variant["name"]] = "✓" if success else "✗"

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {status}  {name}")
    print(f"\n  Results saved to: {output_dir}")
    print(f"{'='*60}")
