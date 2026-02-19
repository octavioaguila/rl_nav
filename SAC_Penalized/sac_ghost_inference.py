#!/usr/bin/env python3
# -------------------------------------------------------------
#  Ghost Trajectory Inference for SAC-Penalized
#  Runs a single episode with a fixed initial/goal pose,
#  stores qpos at every step, and renders a ghosted trajectory.
# -------------------------------------------------------------
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_bunker_env import BunkerEnv
from gymnasium.wrappers import TimeLimit
from feature_extractor import FeatureExtractor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from media.ghost_trajectory_renderer import render_ghost_trajectory

# ========================== Configuration ==========================
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

run_name = "run_1"
max_goal_sampling_distance = 8.0
env_name = "test/world_16_hard"

# Initial and goal poses: [x, y, theta]
initial_pose = [0.0, 0.0, 0.0]
goal_pose    = [5.0, 5.0, 0.0]

# Ghost rendering settings
ENABLE_GHOST_RENDERING = True
ghost_subsample        = 5
ghost_alpha            = 0.4
ghost_width            = 1920
ghost_height           = 1080
ghost_dpi              = 300
ghost_temporal_fade    = True

# Camera configuration (top-down bird's eye)
ghost_cam_lookat   = [3.0, 3.0, 0.0]
ghost_cam_distance = 18.0
ghost_cam_azimuth  = 90.0
ghost_cam_elevation = -90.0

# ========================== Environment Setup ==========================
xml  = os.path.join(root, "assets", "worlds", f"{env_name}.xml")
ckpt = os.path.join(root, "SAC_Penalized", "log", run_name, "best_model", "best_model.zip")

env = DummyVecEnv([lambda: TimeLimit(BunkerEnv(xml_path=xml, render_mode=None, max_goal_sampling_distance=max_goal_sampling_distance), 
                                                max_episode_steps=600)])

raw_env = env.envs[0].unwrapped

model: SAC = SAC.load(ckpt, env, device="auto")
deterministic = True
max_ep_len    = 600

# ========================== Episode Execution ==========================
print(f"[Ghost Inference] Setting manual pose: start={initial_pose}, goal={goal_pose}")
obs = env.reset()

# Set manual pose
manual_obs = raw_env.set_manual_pose(initial_pose, goal_pose)
obs = np.expand_dims(manual_obs, axis=0)

# Store qpos trajectory
episode_qpos = []
episode_qpos.append(raw_env.data.qpos.copy())

print(f"[Ghost Inference] Running episode...")
for step in range(max_ep_len):
    action, _ = model.predict(obs, deterministic=deterministic)
    obs, reward, terminated, info = env.step(action)
    
    episode_qpos.append(raw_env.data.qpos.copy())
    
    inf = info[0]
    if terminated[0] or inf.get("TimeLimit.truncated", False):
        is_success = inf.get("is_success", False)
        collision = inf.get("collision", False)
        outcome = "SUCCESS" if is_success else ("CRASH" if collision else "TIMEOUT")
        print(f"[Ghost Inference] Episode ended at step {step+1}: {outcome}")
        break

print(f"[Ghost Inference] Collected {len(episode_qpos)} qpos states")

# ========================== Ghost Trajectory Rendering ==========================
if ENABLE_GHOST_RENDERING:
    results_dir = os.path.join(root, "SAC_Penalized", "inference_results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{run_name}_ghost_trajectory.png")
    
    mj_model = raw_env.model
    
    render_ghost_trajectory(
        model=mj_model,
        episode_qpos=episode_qpos,
        output_path=output_path,
        width=ghost_width,
        height=ghost_height,
        subsample=ghost_subsample,
        alpha=ghost_alpha,
        temporal_fade=ghost_temporal_fade,
        cam_lookat=ghost_cam_lookat,
        cam_distance=ghost_cam_distance,
        cam_azimuth=ghost_cam_azimuth,
        cam_elevation=ghost_cam_elevation,
        dpi=ghost_dpi,
    )

env.close()
print("[Ghost Inference] Done.")
