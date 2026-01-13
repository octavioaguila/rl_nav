import sys
import os
import time
import numpy as np

try:
    from pynput import keyboard
except ImportError:
    print("Please install pynput: pip install pynput")
    sys.exit(1)

# Ensure envs is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gym_bunker_env import BunkerEnv

# Path to the bunker.xml file
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(ROOT, "assets", "worlds", "cylinders.xml")

print("""
Manual Bunker Control (WASD)
--------------------------
Controls:
  i: Increase forward velocity (+0.1)
  k: Decrease forward velocity (-0.1)
  j: Increase angular velocity (+0.1)
  l: Decrease angular velocity (-0.1)
  Space: Stop (zero both)
  q: Quit
--------------------------
Focus the terminal window and use the keys above.
""")

class KeyController:
    def __init__(self):
        self.action = np.array([0.0, 0.0], dtype=np.float32)  # [linear, angular]
        self.quit = False
        self.step = 0.1
        self.min_val = -1.0
        self.max_val = 1.0

    def on_press(self, key):
        updated = False
        try:
            k = key.char.lower()
            if k == 'i':
                self.action[0] = min(self.action[0] + self.step, self.max_val)
                updated = True
            elif k == 'k':
                self.action[0] = max(self.action[0] - self.step, self.min_val)
                updated = True
            elif k == 'j':
                self.action[1] = min(self.action[1] + self.step, self.max_val)
                updated = True
            elif k == 'l':
                self.action[1] = max(self.action[1] - self.step, self.min_val)
                updated = True
            elif k == 'Q':
                self.quit = True
            # No else: ignore other keys
        except AttributeError:
            if key == keyboard.Key.space:
                self.action[:] = 0.0
                updated = True
        if updated:
            print(f"[CMD] Linear: {self.action[0]:.2f}, Angular: {self.action[1]:.2f}")

    def on_release(self, key):
        pass  # No action on release


def main():
    env = BunkerEnv(XML_PATH, render_mode="human", n_lidar=449)
    obs, info = env.reset()
    controller = KeyController()
    listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    listener.start()
    steps = 0
    reward_cum = 0
    max_step_reward = np.inf

    try:
        while not controller.quit:
            steps += 1
            action = controller.action
            # print(f"\n[CMD] action: {action}")
            normalized_action =  np.clip(np.array([action[0], action[1]], np.float32) / env.vel_scale, -1.0, 1.0)
            # print(f"[CMD] normalized_action: {normalized_action}\n")
            obs, reward, terminated, truncated, info = env.step(normalized_action)
            print(f"[ENV] reward: {reward}")

            reward_cum += reward
            max_step_reward = min(max_step_reward, reward)

            env.render()
            if terminated or truncated or steps == 600:
                print("\nIs success?: ", info['is_success'])
                print("Is collision?: ", info['collision'])
                print("Final reward: ", reward_cum)
                print("Steps: ", steps)
                print("Mean reward: ", reward_cum / steps)
                print("Max step reward: ", max_step_reward)
                print("Episode finished. Resetting...")
                obs, info = env.reset()
                steps = 0
                reward_cum = 0
                max_step_reward = -np.inf

            time.sleep(0.05)  # ~20 FPS
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        env.close()
        listener.stop()

if __name__ == "__main__":
    main()
