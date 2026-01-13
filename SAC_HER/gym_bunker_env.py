# bunker_pid_env.py
from __future__ import annotations

import os
import numpy as np
import mujoco
import torch
import sys
from collections import deque
from numpy.typing import NDArray
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3.common.env_checker import check_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.bunker_velocity_controller import BunkerVelocityController


class BunkerEnv(MujocoEnv):
    """
    ## Parameters
    - n_lidar (int): Number of LiDAR rays (default: 449)

    ## Action Space
    | Num | Action                                                                          | Control Min       | Control Max       | Controller -> Joint                     | Joint      | Type (Unit)      |
    |-----|---------------------------------------------------------------------------------|-------------------|-------------------|-----------------------------------------|------------|------------------|
    | 0   | Linear velocity in robot's reference frame                                      | -1 (-0.5 m/s)     | 1 (0.5 m/s)       | BunkerVelocityController -> w_*_joint   | w_*_joint  | velocity (rad/s) |
    | 1   | Angular velocity in robot's reference frame                                     | -1 (-0.5 rad/s)   | 1 (0.5 rad/s)     | BunkerVelocityController -> w_*_joint   | w_*_joint  | velocity (rad/s) |

    ## Observation Space
    The observation space is a Dict with:
    - `observation`: `Box(-1, 1, (n_lidar*3 + 2), float32)` -> LiDAR + (v, w)
    - `achieved_goal`: `Box(-inf, inf, (3,), float32)` -> [x, y, yaw]
    - `desired_goal`: `Box(-inf, inf, (3,), float32)` -> [x, y, yaw]

    ## Reward Function
    r = (0 if success; -1000 if crash; else -1)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, xml_path: str, frame_skip: int = 10, render_mode: str | None = None, n_lidar: int = 449, inference_mode: bool = False,
                max_goal_sampling_distance: float = 8.0):

        print("[Gym Bunker Env] Max goal sampling distance:", max_goal_sampling_distance)
        
        # LiDAR parameters
        self.n_lidar   = n_lidar
        self.lidar_angles = np.linspace(-np.pi, np.pi, self.n_lidar, endpoint=False).astype(np.float32)
        self.lidar_max_range = 20.0 # This has to be the same as the cutoff in the XML file and point_net_extractor.py.

        # World bounds for random reset and goal sampling
        self.xy_min = np.array([-4., -4.], np.float32)
        self.xy_max = np.array([ 10., 10.], np.float32)

        # Robot's yaw bounds
        self.yaw_min = -np.pi
        self.yaw_max = np.pi

        # Goal success and reward thresholds
        self.goal_xy_distance_threshold = 0.2
        self.min_goal_sampling_distance = 1.0
        self.max_goal_sampling_distance = max_goal_sampling_distance

        # Calculate diagonal distance as maximum possible distance, this is used for normalization of the relative goal vector
        self.max_distance_diagonal = np.linalg.norm(self.xy_max - self.xy_min)
        
        # Velocity: linear and angular velocity bounds (high-level limits)
        self.v_max, self.w_max = 0.5, 0.5
        self.vel_scale = np.array([self.v_max, self.w_max], np.float32)

        # Observation space: LiDAR + (v, w)
        obs_dim = self.n_lidar * 3 + 2
        
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.max_geom = 1_800

        # Call parent constructor (creates model/data/renderer)
        super().__init__(model_path=xml_path, frame_skip=frame_skip, observation_space=self.observation_space, render_mode=render_mode, max_geom=self.max_geom)

        self._initialize_velocity_controller()

        # Get the goal marker body ID and mocap ID
        self.mid_goal_bid = self.model.body('mid_goal_marker').id
        self.mid_goal_mid = int(self.model.body_mocapid[self.mid_goal_bid])
        self.final_goal_bid = self.model.body('final_goal_marker').id
        self.final_goal_mid = int(self.model.body_mocapid[self.final_goal_bid])

        # internal book-keeping
        self._goal: NDArray[np.float32]
        self._is_final_goal: bool = True  # Flag to distinguish between intermediate and final goals

        self.inference_mode = inference_mode

    def get_robot_velocities(self):
        """Get robot velocities body frame"""
        linear_vel_world_qvel = self.data.qvel[0:3].copy()    # [x, y, z] in world frame (m/s)
        angular_vel_world_qvel = self.data.qvel[3:6].copy()    # roll, pitch, yaw] in world frame (rad/s)
        
        # Transform world frame to body frame
        body_id = self.model.body('mobile_base').id
        rotation_matrix = self.data.xmat[body_id].reshape(3, 3)
        
        # Transform linear velocity to body frame
        linear_vel_body = rotation_matrix.T @ linear_vel_world_qvel
        angular_vel_body = rotation_matrix.T @ angular_vel_world_qvel
        
        return linear_vel_body, angular_vel_body

    def reset_model(self) -> NDArray[np.float64]:

        self.velocity_controller.reset()

        rng = self.np_random

        # Sample collision-free initial robot pose
        while True:
            initial_qpos_x, initial_qpos_y = rng.uniform(self.xy_min, self.xy_max)
            initial_theta = rng.uniform(self.yaw_min, self.yaw_max)

            self.data.qpos[:] = 0.
            self.data.qpos[:2] = initial_qpos_x, initial_qpos_y
            self.data.qpos[2] = 0.25 # height above ground, defined in XML
            c, s = np.cos(initial_theta / 2), np.sin(initial_theta / 2)
            self.data.qpos[3:7] = (c, 0., 0., s)
            
            # Check if initial pose is collision-free
            mujoco.mj_forward(self.model, self.data)
            if not self._is_collision():
                # print(f"Resetting initial pose at {initial_qpos_x, initial_qpos_y}, theta={initial_theta:.2f} rad")
                break

        # sample goal between self.min_distance and self.max_distance_diagonal and collision-free
        while True:
            goal_qpos_x, goal_qpos_y = rng.uniform(self.xy_min, self.xy_max)
            goal_theta = rng.uniform(self.yaw_min, self.yaw_max)

            self._goal = np.array([goal_qpos_x, goal_qpos_y, goal_theta])

            distance = np.linalg.norm(self._goal[:2] - (initial_qpos_x, initial_qpos_y))

            if self.min_goal_sampling_distance < distance <= self.max_goal_sampling_distance:
                # Check if goal position is collision-free
                self.data.qpos[:2] = self._goal[:2]
                c, s = np.cos(goal_theta / 2), np.sin(goal_theta / 2)
                self.data.qpos[3:7] = (c, 0., 0., s)
                mujoco.mj_forward(self.model, self.data)
                has_collision = self._is_collision()

                # Return robot to original position
                self.data.qpos[:2] = initial_qpos_x, initial_qpos_y
                c, s = np.cos(initial_theta / 2), np.sin(initial_theta / 2)
                self.data.qpos[3:7] = (c, 0., 0., s)
                mujoco.mj_forward(self.model, self.data)
                
                if not has_collision:
                    # print(f"Resetting goal at {goal_qpos_x, goal_qpos_y, goal_theta} ")
                    # print(f"Goal distance: {distance} m")
                    break

        self.data.qvel[:] = 0.
        self._is_final_goal = True  # When reset, this is the final goal

        self._set_goal_marker_position(self._goal, is_final_goal=True)

        return self._get_obs()

    def set_manual_pose(self, initial_pose: NDArray[np.float32] | list[float], goal_pose: NDArray[np.float32] | list[float]) -> dict[str, np.ndarray]:
        """
        Manually set the initial robot pose and the goal pose.
        Follows the same logic as reset_model, including collision checks, bounds checks,
        distance checks, and controller reset.
        """
        self.velocity_controller.reset()

        initial_pose = np.array(initial_pose, dtype=np.float32)
        goal_pose = np.array(goal_pose, dtype=np.float32)

        # Bounds check
        if not (self.xy_min[0] <= initial_pose[0] <= self.xy_max[0] and 
                self.xy_min[1] <= initial_pose[1] <= self.xy_max[1]):
            raise ValueError(f"Initial pose {initial_pose[:2]} is out of world bounds {self.xy_min} to {self.xy_max}")
        
        if not (self.xy_min[0] <= goal_pose[0] <= self.xy_max[0] and 
                self.xy_min[1] <= goal_pose[1] <= self.xy_max[1]):
            raise ValueError(f"Goal pose {goal_pose[:2]} is out of world bounds {self.xy_min} to {self.xy_max}")

        # Initial robot pose collision check
        self.data.qpos[:] = 0.
        self.data.qpos[:2] = initial_pose[:2]
        self.data.qpos[2] = 0.25  # height above ground, defined in XML/reset_model
        c, s = np.cos(initial_pose[2] / 2), np.sin(initial_pose[2] / 2)
        self.data.qpos[3:7] = (c, 0., 0., s)

        mujoco.mj_forward(self.model, self.data)
        if self._is_collision():
            raise ValueError(f"Initial pose {initial_pose} results in a collision.")

        # Goal distance check
        distance = np.linalg.norm(goal_pose[:2] - initial_pose[:2])
        if not (self.min_goal_sampling_distance < distance <= self.max_goal_sampling_distance):
            raise ValueError(f"Distance between start and goal ({distance:.2f}m) is outside the valid range "
                             f"[{self.min_goal_sampling_distance}, {self.max_goal_sampling_distance}]")

        # Goal pose collision check
        self._goal = goal_pose.copy()
        
        # Temporarily set robot to goal position to check for collision
        self.data.qpos[:2] = self._goal[:2]
        c, s = np.cos(self._goal[2] / 2), np.sin(self._goal[2] / 2)
        self.data.qpos[3:7] = (c, 0., 0., s)
        mujoco.mj_forward(self.model, self.data)
        has_collision = self._is_collision()

        # Return robot to original (initial) position
        self.data.qpos[:2] = initial_pose[:2]
        c, s = np.cos(initial_pose[2] / 2), np.sin(initial_pose[2] / 2)
        self.data.qpos[3:7] = (c, 0., 0., s)
        mujoco.mj_forward(self.model, self.data)

        if has_collision:
            raise ValueError(f"Goal pose {goal_pose} results in a collision.")

        # Finalize state (reset velocities, markers, etc.)
        self.data.qvel[:] = 0.
        self._is_final_goal = True
        self._set_goal_marker_position(self._goal, is_final_goal=True)

        return self._get_obs()

    def reset(self, *, seed: int | None = None, options: dict[str, any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, any]]:
        ob, _ = super().reset(seed=seed, options=options)
        info = self._get_reset_info()
        return ob, info

    def set_inference_goal(self, goal: NDArray[np.float32]) -> None:
        """
        Set the goal for inference mode. We just set an intermediate goal from the global planner. We suppose that the goal is collision-free.
        """
        self._goal = np.array(goal, dtype=np.float32)
        self._set_goal_marker_position(self._goal, is_final_goal=False)
        self._is_final_goal = False  # This is an intermediate goal, not the final goal

    def set_curriculum(self, max_goal_sampling_distance: float) -> None:
        self.max_goal_sampling_distance = max_goal_sampling_distance
        
    def step(self, action: NDArray[np.float32]) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        # Denormalize action from [-1, 1] to actual velocity limits
        v_cmd = action[0] * self.v_max
        w_cmd = action[1] * self.w_max

        # print(f"\n[GYM ENV] v_cmd: {v_cmd}, w_cmd: {w_cmd}")
        self.velocity_controller.set_cmd(float(v_cmd), float(w_cmd))

        # print(f" Position: {self.data.qpos[:2]}")

        # integrate physics
        self.do_simulation(ctrl=np.zeros(self.model.nu), n_frames=self.frame_skip)

        obs        = self._get_obs()

        robot_pos  = obs["achieved_goal"]
        goal_pos   = obs["desired_goal"]

        is_success = self._is_success(goal_pos[:2], robot_pos[:2])
        collision  = self._is_collision()

        reward     = float(self.compute_reward(achieved_goal=robot_pos, desired_goal=goal_pos, info={"collision": collision}))

        info = {
            "is_success": is_success,
            "collision": collision,
        }
        
        if self.inference_mode:
            terminated = collision or (is_success and self._is_final_goal)
        else:
            terminated = is_success or collision

        truncated = False # remember that this is handled by the wrapper

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _initialize_velocity_controller(self):
        self.velocity_controller = BunkerVelocityController()

        id_rr = self.model.body("w_rr").id
        id_lr = self.model.body("w_lr").id
        width_track = abs(self.model.body_pos[id_rr][1] - self.model.body_pos[id_lr][1])

        self.velocity_controller.w_track = width_track

        id_geom = self.model.geom("w_rr_geom").id
        self.velocity_controller.r_wheel = self.model.geom_size[id_geom][0]

        self.velocity_controller.act_r = [self.model.actuator(n).id for n in ("w_rr","w_rc","w_rf")]
        self.velocity_controller.act_l = [self.model.actuator(n).id for n in ("w_lr","w_lc","w_lf")]

        self.sensor_frc_r = [self.model.sensor(n).id for n in ("sf_rr","sf_rc","sf_rf")]
        self.sensor_frc_l = [self.model.sensor(n).id for n in ("sf_lr","sf_lc","sf_lf")]

        mujoco.set_mjcb_control(self.velocity_controller)

    def _set_action_space(self):
        """
        The official environments let the base class infer the action space from the MuJoCo XML actuator ranges.
        However, in this case, a custom velocity controller is used, so the action space is manually set.
        """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        return self.action_space

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        Constructs the full observation dictionary.
        """
        pts = self._lidar_points().astype(np.float32)  # shape (n_lidar_points, 3)
        pts_flat = pts.flatten()  # shape (n_lidar_points*3,)
        v_raw, w_raw = self.get_robot_velocities()
        v_raw, w_raw = v_raw[0], w_raw[2]
        vel = np.clip(np.array([v_raw, w_raw], np.float32) / self.vel_scale, -1.0, 1.0)  # shape (2,)
        obs = np.concatenate([pts_flat, vel]).astype(np.float32)  # shape (n_lidar_points*3+2,)

        # Robot Pose (Achieved Goal)
        world_robot_xy_pos = self.data.qpos[:2].copy()
        qw, qx, qy, qz = self.data.qpos[3:7].copy()
        world_robot_yaw = 2.0 * np.arctan2(qz, qw)
        achieved_goal = np.array([world_robot_xy_pos[0], world_robot_xy_pos[1], world_robot_yaw], dtype=np.float32)

        return {
            "observation": obs,
            "achieved_goal": achieved_goal,
            "desired_goal": self._goal.astype(np.float32)
        }

    def _get_reset_info(self): 
        return {"goal": self._goal.copy()}


    def _goal_pose_error(self) -> float:
        qw, qx, qy, qz = self.data.qpos[3:7].copy()
        yaw = 2.0 * np.arctan2(qz, qw) # robot yaw in world frame
        goal_yaw = self._goal[2]

        # Normalize angle difference to [-π, π] range
        angle_diff = yaw - goal_yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        return float(np.abs(angle_diff))

    def _is_collision(self) -> bool:
        for i in range(self.data.ncon):
            g1 = self.model.geom(self.data.contact[i].geom1).name
            g2 = self.model.geom(self.data.contact[i].geom2).name

            # Ignore collisions with the floor
            if g1 == "floor" or g2 == "floor":
                continue
            return True

        return False

    def _is_success(self, robot_pos: np.ndarray, goal_pos: np.ndarray) -> bool:
        if float(np.linalg.norm(robot_pos - goal_pos)) < self.goal_xy_distance_threshold:
            return True
        else:
            return False

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, any] | list[dict[str, any]]) -> np.ndarray:
        """
        Calculates the reward.
        Vectorized to handle both single steps (scalars) and HER replay buffer (batches).
        """

        # 1. Calculate distance (X, Y only)
        # axis=-1 ensures this works for shapes (3,) and (N, 3)
        d = np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)
        
        # 2. Check success condition
        is_success = d < self.goal_xy_distance_threshold
        
        # 3. Extract collision info safely (Handles Dict vs List of Dicts)
        if isinstance(info, dict):
            # Case A: Called from step() (Single dict)
            collision = info["collision"]                   # I DON'T USE .get() on porpouse to avoid missing keys in the dict
            is_collision = np.array(collision, dtype=bool)
        else:
            # Case B: Called from HER Replay Buffer (List of dicts)
            is_collision = np.array([x["collision"] for x in info], dtype=bool)
        
        reward = np.full_like(d, -1.0, dtype=np.float32) # Base step penalty
        reward = np.where(is_success, 0.0, reward)       # Success reward
        reward = np.where(is_collision, -1000.0, reward)  # Collision penalty (highest priority)

        return reward

    # LiDAR
    def _lidar_points(self) -> np.ndarray:
        """
        Returns (n_lidar, 3) array with columns:
        [sin alpha, cos alpha, d_norm]
        where d_norm ∈ [-1, 1] (0 m → -1, 6 m → +1, >6 m → +1)

        ROS2 standard: [-π, π) Left: +π/2 Right: -π/2
        """
        # raw distances from MuJoCo rangefinder
        d_raw = self.data.sensordata.astype(np.float32) # shape (n_lidar,)
        d_raw = d_raw[6:] # NOTE: the first 6 values are the torque sensor readings

        # Transform -1 values (no reading) to self.lidar_max_range
        d_raw = np.where(d_raw == -1.0, self.lidar_max_range, d_raw)

        # Clip & scale to [-1, 1]
        d_clipped = np.clip(d_raw, 0.0, self.lidar_max_range)               # 0 … 6
        d_norm    = (d_clipped / self.lidar_max_range) * 2.0 - 1.0          # 0 → –1, 6 → +1

        # Stack into feature matrix
        pts = np.stack([
            np.sin(self.lidar_angles),                 # sin α  ∈ [-1, 1]
            np.cos(self.lidar_angles),                 # cos α  ∈ [-1, 1]
            d_norm                                  # normalised distance
        ], axis=-1)

        # Print 8 LiDAR points at 45-degree intervals
        # angles_deg = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
        # print(f"\n=== LiDAR Points at 45° intervals (n_lidar={self.n_lidar}) ===")
        # for angle_deg in angles_deg:
        #     # Convert angle to index
        #     angle_rad = np.radians(angle_deg)
        #     # Find closest index in lidar_ang array
        #     idx = np.argmin(np.abs(self.lidar_angles - angle_rad))
        #     actual_angle_deg = np.degrees(self.lidar_angles[idx])
        #     print(f"Angle {angle_deg:3d}° (actual: {actual_angle_deg:6.2f}°): "
        #           f"sin={pts[idx, 0]:6.3f}, cos={pts[idx, 1]:6.3f}, d_norm={pts[idx, 2]:6.3f}, "
        #           f"d_raw={d_raw[idx]:6.3f}m idx={idx}")

        return pts
    
    def _set_goal_marker_position(self, pos: np.ndarray, is_final_goal: bool = False):
        """Set the goal marker position in MuJoCo simulation, just for visualization purposes."""
        pos = np.array([pos[0], pos[1], 0.1])
        if is_final_goal:
            self.data.mocap_pos[self.final_goal_mid] = pos
        else:
            self.data.mocap_pos[self.mid_goal_mid] = pos

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml  = os.path.join(root, "assets", "worlds", "empty.xml")
    env  = BunkerEnv(xml, render_mode="human")
    check_env(env, warn=True, skip_render_check=False)