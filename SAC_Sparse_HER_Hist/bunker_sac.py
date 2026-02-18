#!/usr/bin/env python3
import os, sys

import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
from lib.bunker_callback import BunkerCallback
from gym_bunker_env import BunkerEnv
from feature_extractor import FeatureExtractor

def make_single_env(xml_paths: list[str], max_ep_steps: int, env_i: int):
    def _init():
        xml_path = xml_paths[env_i % len(xml_paths)] # If I have 5 envs and 2 xmls, env 0 will use xml 0, env 1 will use xml 1, env 2 will use xml 0, etc.
        # env = BunkerEnv(xml_path=xml_path, render_mode="human")
        env = BunkerEnv(xml_path=xml_path, render_mode=None, n_lidar=449)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_ep_steps)
        return env
    return _init


def train(train_xml_paths, val_xml_paths, total_steps, n_envs, max_ep_steps, log_dir):

    os.makedirs(log_dir, exist_ok=True)

    # Vectorised env wrapped with VecMonitor so every worker writes a Monitor file
    env = SubprocVecEnv([make_single_env(train_xml_paths, max_ep_steps, env_i) for env_i in range(n_envs)])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor"))

    # Evaluation env
    eval_env = SubprocVecEnv([make_single_env(val_xml_paths, max_ep_steps, i) for i in range(len(val_xml_paths))])
    eval_env = VecMonitor(eval_env)
    n_eval_episodes = len(val_xml_paths) * 10

    # callbacks ────────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(eval_env, best_model_save_path=os.path.join(log_dir, "best_model"), log_path=log_dir, eval_freq=5_000,
                     n_eval_episodes=n_eval_episodes, deterministic=True, render=False, verbose=1)
    
    callbacks = CallbackList([
        eval_callback,
        BunkerCallback()
    ])

    # policy/network definition
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256, n_lidar=449, max_distance_diagonal=env.get_attr("max_distance_diagonal")[0]),
        net_arch=[512, 512],
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            copy_info_dict=True,
        ),
        batch_size=512,
        learning_rate=3e-4,
        learning_starts=n_envs * max_ep_steps,
        gamma=0.99,
        tau=0.005,
        buffer_size=1_000_000,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=2,
        tensorboard_log=os.path.join(log_dir, "tb"),
        verbose=1,
        device="cuda",
    )

    print(f"Starting training on {n_envs} environments...")
    print(f"Logging to {log_dir}")
    print(f"Train XML pool: {len(train_xml_paths)} worlds")
    print(f"Val XML pool: {len(val_xml_paths)} worlds")

    model.learn(total_timesteps=total_steps, callback=callbacks, tb_log_name="run", progress_bar=True)


def get_next_run_name(log_root):
    if not os.path.exists(log_root):
        return "run_1"
    
    existing_runs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d)) and d.startswith("run_")]
    if not existing_runs:
        return "run_1"
    
    run_nums = []
    for run in existing_runs:
        try:
            run_nums.append(int(run.split("_")[1]))
        except (IndexError, ValueError):
            continue
            
    if not run_nums:
        return "run_1"
        
    return f"run_{max(run_nums) + 1}"

if __name__ == "__main__":

    root = os.path.dirname(os.path.abspath(__file__))
    worlds_path_train  = os.path.join(root, "..", "assets", "worlds", "train")
    worlds_path_val  = os.path.join(root, "..", "assets", "worlds", "val")
    
    train_xml_pool = [os.path.join(worlds_path_train, f) for f in os.listdir(worlds_path_train) if f.endswith(".xml")]
    train_xml_pool.sort()

    val_xml_pool = [os.path.join(worlds_path_val, f) for f in os.listdir(worlds_path_val) if f.endswith(".xml")]
    val_xml_pool.sort()

    # Parameters
    total_steps = 2_000_000
    n_envs = 8
    
    base_log_dir = os.path.join(root, "log")

    run_name = get_next_run_name(base_log_dir)

    log_dir = os.path.join(base_log_dir, run_name)
    
    max_ep_steps = 600 # This is 6000 Mujoco steps, because I have a n_frames = 10

    train(train_xml_paths=train_xml_pool, val_xml_paths=val_xml_pool, total_steps=total_steps, n_envs=n_envs, max_ep_steps=max_ep_steps, log_dir=log_dir)
