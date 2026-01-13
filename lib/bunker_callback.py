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
        self.success_buffer = deque(maxlen=100)
        self.energy_buffer = deque(maxlen=100)
        self.last_params = {}

    def _on_step(self) -> bool:
        # Iterate over all environments
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                
                # Log Collision
                if "collision" in info:
                    self.collision_buffer.append(float(info["collision"]))

                # Log Energy
                if "energy_joules" in info:
                    self.energy_buffer.append(float(info["energy_joules"]))
                
        return True

    def _on_rollout_end(self) -> None:
        # Log to TensorBoard at the end of each rollout
        if len(self.collision_buffer) > 0:
            self.logger.record("rollout/collision_rate", np.mean(self.collision_buffer))
        if len(self.energy_buffer) > 0:
            self.logger.record("rollout/energy_joules", np.mean(self.energy_buffer))

        # Log environment parameters if they exist
        for param in ["max_goal_sampling_distance", "energy_weight"]:
            try:
                # Use get_attr to support vectorized environments
                # We assume all environments have the same parameter value
                value = self.training_env.get_attr(param)[0]
                self.logger.record(f"env/{param}", value)
                
                # Print only if the value has changed or it's the first time
                if self.verbose > 0 and self.last_params.get(param) != value:
                    print(f"[BunkerCallback] {param} updated to: {value}")
                    self.last_params[param] = value
            except (AttributeError, Exception):
                # Parameter not present in the environment
                pass

class CurriculumCallback(BaseCallback):
    def __init__(self, eval_env, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.current_level = 0
        
        # Define your curriculum levels
        self.levels = [
            {"max_goal_sampling_distance": 4.0, "energy_weight": 0.0,  "threshold": 0.8}, # In the first level the only parameter that matters is threshold, the rest is just for compatibility
            {"max_goal_sampling_distance": 8.0, "energy_weight": 0.0, "threshold": 0.8},
            {"max_goal_sampling_distance": 12.0, "energy_weight": 0.0, "threshold": 0.8},
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
                print(f"Updating: max_goal_sampling_distance={new_cfg['max_goal_sampling_distance']}, energy_weight={new_cfg['energy_weight']}\n")
                
                # Update all vectorized environments (Training & Eval)
                self._update_envs(new_cfg)
        
        # Log curriculum metrics
        current_cfg = self.levels[self.current_level]
        self.logger.record("curriculum/level", self.current_level)
        self.logger.record("curriculum/max_goal_sampling_distance", current_cfg["max_goal_sampling_distance"])
        self.logger.record("curriculum/energy_weight", current_cfg["energy_weight"])
        
        return True

    def _update_envs(self, cfg):
        # We must use env_method or set_attr for SubprocVecEnv
        # Update Training Envs
        self.training_env.env_method("set_curriculum", cfg["max_goal_sampling_distance"], cfg["energy_weight"])
        # Update Eval Envs
        self.eval_env.env_method("set_curriculum", cfg["max_goal_sampling_distance"], cfg["energy_weight"])