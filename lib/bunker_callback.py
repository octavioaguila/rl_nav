from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np

class BunkerCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.collision_buffer = deque(maxlen=100)
        self.max_goal_sampling_buffer = deque(maxlen=100)
        self.k_history_window_buffer = deque(maxlen=100)

    def _on_step(self) -> bool:
        # Iterate over all environments
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                
                # Log Collision
                if "collision" in info:
                    self.collision_buffer.append(float(info["collision"]))

                if "max_goal_sampling_distance" in info:
                    self.max_goal_sampling_buffer.append(float(info["max_goal_sampling_distance"]))

                if "k_history_window" in info:
                    self.k_history_window_buffer.append(float(info["k_history_window"]))
   
        return True

    def _on_rollout_end(self) -> None:
        # Log to TensorBoard at the end of each rollout
        if len(self.collision_buffer) > 0:
            self.logger.record("rollout/collision_rate", np.mean(self.collision_buffer))

        if len(self.max_goal_sampling_buffer) > 0:
            self.logger.record("vars/max_goal_sampling", np.mean(self.max_goal_sampling_buffer))

        if len(self.k_history_window_buffer) > 0:
            self.logger.record("vars/k_history_window", np.mean(self.k_history_window_buffer))


class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.current_level = 0
        
        # Define your curriculum levels
        self.levels = [
            {"max_goal_sampling_distance": 4.0, "threshold": 0.7}, # In the first level the only parameter that matters is threshold, the rest is just for compatibility
            {"max_goal_sampling_distance": 8.0, "threshold": 0.7},
            {"max_goal_sampling_distance": 12.0, "threshold": 0.7},
            {"max_goal_sampling_distance": 16.0, "threshold": 0.7},
            {"max_goal_sampling_distance": 20.0, "threshold": 0.7},
        ]

    def _on_step(self) -> bool:
        eval_callback = self.parent
        evaluation_successes = eval_callback.evaluations_successes
        success_rate = np.mean(evaluation_successes[-1])

        print(f"\n[Curriculum] Last success rate: {success_rate}")

        if self.current_level < len(self.levels) - 1:
            target_threshold = self.levels[self.current_level]["threshold"]

            print(f"\n[Curriculum] Current level: {self.current_level}, Target threshold: {target_threshold}\n")
            
            if success_rate >= target_threshold:
                self.current_level += 1
                new_cfg = self.levels[self.current_level]
                    
                print(f"\n[Curriculum] LEVEL UP! Now at Level {self.current_level}\n")
                print(f"Updating: max_goal_sampling_distance={new_cfg['max_goal_sampling_distance']}\n")
                
                # Update all vectorized environments (Training & Eval)
                self._update_envs(new_cfg)
        
        return True

    def _update_envs(self, cfg):
        # We must use env_method or set_attr for SubprocVecEnv
        # Update Training Envs
        self.training_env.env_method("set_curriculum", cfg["max_goal_sampling_distance"])
        # Update Eval Envs
        self.eval_env.env_method("set_curriculum", cfg["max_goal_sampling_distance"])