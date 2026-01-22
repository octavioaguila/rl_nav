# bunker_pid_env.py
from __future__ import annotations

import os
import numpy as np
import mujoco
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
    The observation space is a `Box(-1, 1, (n_lidar*3 + 3 + k_history_window*2), float32)` where the elements are as follows:
    | Index Range                                  | Observation Description                                                                        | Min   | Max   |
    |----------------------------------------------|------------------------------------------------------------------------------------------------|-------|-------|
    |           0 to n_lidar*3-1                   | LiDAR features (sin(alpha), cos(alpha), d_norm)                                                | -1    | 1     |
    |   n_lidar*3 to n_lidar*3+2                   | Relative goal vector (distance_norm, angle_to_xy_goal_norm, angle_to_orientation_goal_norm)    | -1    | 1     |
    | n_lidar*3+3 to n_lidar*3+3+k_history_window*2| k_history_window * vel (v, w)                                                                  | -1    | 1     |

    ## Reward Function
    The reward function is dense, defined as:
    | Component         |       Value      |
    |-------------------|-------------------|
    | Collision         |       -10.0       |
    | Success           |        10.0       |
    | Any other         | d_{t-1}^g - d_t^g |
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, xml_path: str, frame_skip: int = 10, render_mode: str | None = None, n_lidar: int = 449, inference_mode: bool = False,
                 max_goal_sampling_distance: float = 8.0, k_history_window: int = 10):

        print("[Gym Bunker Env] Max goal sampling distance:", max_goal_sampling_distance)
        print("[Gym Bunker Env] K history window:", k_history_window)

        # LiDAR parameters
        self.n_lidar   = n_lidar
        self.lidar_angles = np.linspace(-np.pi, np.pi, self.n_lidar, endpoint=False).astype(np.float32)
        self.lidar_max_range = 20.0 # This has to be the same as the cutoff in the XML file and point_net_extractor.py.

        # History parameters
        self.k_history = k_history_window
        self.history = deque(maxlen=self.k_history)

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
        # and for sampling the goal position
        self.max_distance_diagonal = np.linalg.norm(self.xy_max - self.xy_min)
        
        # Velocity: linear and angular velocity bounds
        # high-level limits
        self.v_max, self.w_max = 0.5, 0.5
        self.vel_scale = np.array([self.v_max, self.w_max], np.float32)
        
        # Observation space: all lidar features, relative goal, velocity history
        obs_dim = self.n_lidar * 3 + 3 + self.k_history * 2
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.max_geom = 1_800

        # Call parent constructor (creates model/data/renderer)
        super().__init__(model_path=xml_path, frame_skip=frame_skip, observation_space=observation_space, render_mode=render_mode, max_geom=self.max_geom)

        self._initialize_velocity_controller()

        # Get the goal marker body ID and mocap ID
        self.mid_goal_bid = self.model.body('mid_goal_marker').id
        self.mid_goal_mid = int(self.model.body_mocapid[self.mid_goal_bid])
        self.final_goal_bid = self.model.body('final_goal_marker').id
        self.final_goal_mid = int(self.model.body_mocapid[self.final_goal_bid])

        # internal book-keeping
        self._prev_dist: float = 0.0
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

    def reset_model(self) -> NDArray[np.float32]:

        self.velocity_controller.reset()

        rng = self.np_random

        # Sample collision-free initial robot pose
        num_attempts = 0
        while True:
            num_attempts += 1
            if num_attempts > 10_000:
                raise RuntimeError(f"Could not sample collision-free initial robot pose after {num_attempts} attempts. Check your map size and obstacle density.")

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

        # sample goal between self.min_goal_sampling_distance and self.max_goal_sampling_distance and collision-free
        num_attempts = 0
        while True:
            num_attempts += 1
            if num_attempts > 10_000:
                raise RuntimeError(f"Could not sample collision-free goal pose after {num_attempts} attempts. Check your map size and obstacle density.")
                
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
        self._prev_dist = self._goal_xy_distance() 
        self._is_final_goal = True  # When reset, this is the final goal

        self._set_goal_marker_position(self._goal, is_final_goal=True)

        # Clear history and pre-fill with the initial state to avoid cold-start issues
        self.history.clear()
        initial_z_t = self._get_single_step_features()
        for _ in range(self.k_history):
            self.history.append(initial_z_t)

        return self._get_obs()

    def set_inference_goal(self, goal: NDArray[np.float32]) -> None:
        """
        Set the goal for inference mode. We just set an intermediate goal from the global planner. We suppose that the goal is collision-free.
        """
        self._goal = np.array(goal, dtype=np.float32)
        self._set_goal_marker_position(self._goal, is_final_goal=False)
        self._prev_dist = self._goal_xy_distance()
        self._is_final_goal = False  # This is an intermediate goal, not the final goal
        
    def step(self, action: NDArray[np.float32]) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        # Denormalize action from [-1, 1] to actual velocity limits
        v_cmd = action[0] * self.v_max
        w_cmd = action[1] * self.w_max
        self.velocity_controller.set_cmd(float(v_cmd), float(w_cmd))

        # integrate physics
        self.do_simulation(ctrl=np.zeros(self.model.nu), n_frames=self.frame_skip)

        obs        = self._get_obs()
        success    = self._is_success()
        collision  = self._is_collision()

        if self.inference_mode:
            # In inference mode, only terminate on collision or reaching final goal
            terminated = collision or (success and self._is_final_goal)
        else:
            terminated = success or collision

        truncated = False # remember that this is handled by the wrapper
        reward = self._reward(terminated or truncated)

        info = {
            "is_success": success,
            "collision": collision,
            "goal": self._goal,
            "max_goal_sampling_distance": self.max_goal_sampling_distance,
            "k_history_window": self.k_history
        }

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

        mujoco.set_mjcb_control(self.velocity_controller)

    def _set_action_space(self):
        """
        The official environments let the base class infer the action space from the MuJoCo XML actuator ranges.
        However, in this case, a custom velocity controller is used, so the action space is manually set.
        """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        return self.action_space

    def _get_single_step_features(self) -> np.ndarray:
        """
        Gathers the features for a single time step z_t (Goal Independent).
        z_t = [v, w]
        All values are normalized to roughly [-1, 1].
        """
        v_raw, w_raw = self.get_robot_velocities()
        v_raw, w_raw = v_raw[0], w_raw[2]
        vel = np.clip(np.array([v_raw, w_raw], np.float32) / self.vel_scale, -1.0, 1.0) # (v, w)

        return vel # (2,)

    def _get_obs(self) -> np.ndarray:
        """
        We get the lidar points, the relative goal vector, and the velocity history.
        """
        pts = self._lidar_points().astype(np.float32)  # shape (n_lidar_points, 3)
        pts_flat = pts.flatten()  # shape (n_lidar_points*3,)
        
        relative_goal = self._relative_goal_vec().astype(np.float32)  # shape (3,)
        
        # Get current features and update history
        z_t = self._get_single_step_features()
        self.history.append(z_t)

        # Flatten history for the network
        # history is a deque of K arrays of shape (2,). Stack -> (K, 2) -> Flatten -> (K*2,)
        history_flat = np.array(self.history).flatten().astype(np.float32) # (K*2,)
        
        obs = np.concatenate([pts_flat, relative_goal, history_flat]).astype(np.float32)  # shape (n_lidar_points*3 + 3 + K*2,)
        return obs
    
    def _relative_goal_vec(self) -> np.ndarray:
        """
        Returns [distance_to_goal, angle_to_goal] in robot frame.
        Distance is normalized to [-1, 1] and angle is in radians.
        """
        # World frame position delta
        dx_w, dy_w = (self._goal[:2] - self.data.qpos[:2].copy())
        distance_to_goal = np.linalg.norm([dx_w, dy_w])

        # Body frame rotation
        qw, qx, qy, qz = self.data.qpos[3:7].copy()
        yaw = 2.0 * np.arctan2(qz, qw) # yaw in world frame

        # Rotation using matrix multiplication to get the goal vector in robot frame
        c, s = np.cos(yaw), np.sin(yaw)
        dx_b =  c * dx_w + s * dy_w
        dy_b = -s * dx_w + c * dy_w

        # Calculate angle to goal in robot frame
        angle_to_goal = np.arctan2(dy_b, dx_b) # angle in robot frame

        # Normalize distance to [-1, 1] using diagonal as maximum
        # 0m means -1, max_distance means +1
        distance_norm = np.clip((distance_to_goal / self.max_distance_diagonal) * 2.0 - 1.0, -1.0, 1.0)
        
        # Normalize angle to [-1, 1]: -π → -1, 0 → 0, π → +1
        angle_to_xy_goal_norm = angle_to_goal / np.pi

        # Normalize angle difference to [-π, π] range before normalizing to [-1, 1]
        angle_diff = yaw - self._goal[2]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize to [-π, π]
        angle_to_orientation_goal_norm = angle_diff / np.pi

        return np.array([distance_norm, angle_to_xy_goal_norm, angle_to_orientation_goal_norm], np.float32)


    def _get_reset_info(self):   # optional detailed info
        return {"goal": self._goal.copy()}

    # Utils
    def _goal_xy_distance(self) -> float:
        return float(np.linalg.norm(self._goal[:2] - self.data.qpos[:2].copy()))

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

    def _is_success(self) -> bool:
        # if self._goal_xy_distance() < self.goal_xy_distance_threshold and self._goal_pose_error() < self.goal_pose_error_threshold:
        if self._goal_xy_distance() < self.goal_xy_distance_threshold:
            return True
        else:
            return False

    def _reward(self, terminated) -> float:
        if terminated and self._is_success():
            return 10.0
        if terminated and self._is_collision():
            return -10.0
        d_now = self._goal_xy_distance()           # raw metres
        delta_d = (self._prev_dist - d_now)

        reward = delta_d

        self._prev_dist = d_now

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
