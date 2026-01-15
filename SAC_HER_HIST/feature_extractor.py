import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FeatureExtractor(BaseFeaturesExtractor):
    """
    Hybrid Architecture for HER adapted for BunkerEnv: 
    1. Spatial Stream: PointNet for current LiDAR (N x 3)
    2. Temporal Stream: 1D-CNN for Velocity History (K x 2)
    3. Goal Stream: Calculated Relative Goal (3) -> Injected at Fusion
    
    Fusion: Concatenate [Spatial, Temporal, Goal] -> MLP
    """

    def __init__(self, observation_space, *, features_dim: int = 256, n_lidar: int = 449, max_range: float = 20.0, 
                 beta: float = 0.25, history_k: int = 10, history_features: int = 2, max_distance_diagonal: float = 20.0):
        super().__init__(observation_space, features_dim)

        self.n_lidar    = int(n_lidar)
        self.lidar_feat = 3                # (sin, cos, d_norm)
        self.max_range  = float(max_range)
        self.beta       = float(beta)
        self.k          = int(history_k)
        self.h_feat     = int(history_features)

        self.max_distance_diagonal = max_distance_diagonal

        # Shared point-wise MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 32),   nn.LeakyReLU(),
            nn.Linear(32, 64),  nn.LeakyReLU(),
            nn.Linear(64, 128), nn.LeakyReLU(),
        )
        self.ln_spatial = nn.LayerNorm(128)

        # Temporal Stream (1D-CNN)
        # Input: (B, 2, K) -> Output: (B, 64 * (K - 2))
        self.history_encoder = nn.Sequential(
            nn.Conv1d(in_channels=self.h_feat, out_channels=32, kernel_size=2, stride=1), nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        self.history_out_dim = 64 * (self.k - 2) 
        self.ln_temporal = nn.LayerNorm(self.history_out_dim)

        # Fusion Stream
        # Input: 128 (Spatial) + History_Out (Temporal) + 3 (Relative Goal)
        fusion_input_dim = 128 + self.history_out_dim + 3
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 256),              nn.LeakyReLU(),
            nn.Linear(256, features_dim),     nn.LeakyReLU(),
        )

    def forward(self, obs):
        """
        obs: Dict with keys 'observation', 'achieved_goal', 'desired_goal'
        'observation' layout: [current_lidar_flat, velocity_history_flat]
        """
        flat_obs = obs['observation']
        ag = obs['achieved_goal']
        dg = obs['desired_goal']

        B = flat_obs.size(0)

        # Split observation into LiDAR and Velocity History
        split_idx = self.n_lidar * self.lidar_feat
        lidar_flat = flat_obs[:, :split_idx]  # (B, N*3)
        hist_flat  = flat_obs[:, split_idx:]  # (B, K*2)

        # LiDAR Feature Extraction (PointNet)
        pts = lidar_flat.view(B, self.n_lidar, self.lidar_feat)
        sincos = pts[..., :2]
        d_norm = pts[..., 2]

        # Pre-process distance: convert d_norm [-1, 1] to 1/(d - beta)
        d_m      = self.max_range * (d_norm + 1.0) * 0.5
        inv_d    = 1.0 / torch.clamp(d_m - self.beta, min=1e-4)
        pts_proc = torch.cat([sincos, inv_d.unsqueeze(-1)], dim=-1)

        pts_feat = self.point_mlp(pts_proc.view(-1, 3))
        pts_feat = pts_feat.view(B, self.n_lidar, 128)
        global_feat, _ = torch.max(pts_feat, dim=1) # Symmetric function (Max Pool)
        global_feat = self.ln_spatial(global_feat)

        # History Feature Extraction (1D-CNN)
        h_time = hist_flat.view(B, self.k, self.h_feat) # (B, K, 2)
        h_conv_in = h_time.permute(0, 2, 1)             # (B, 2, K)
        hist_feat = self.history_encoder(h_conv_in)
        hist_feat = self.ln_temporal(hist_feat)

        # Relative Goal Extraction (Body Frame)
        diff = dg - ag
        dx, dy = diff[:, 0], diff[:, 1]

        # Rotate into Robot Frame
        robot_yaw = ag[:, 2]
        c = torch.cos(robot_yaw)
        s = torch.sin(robot_yaw)

        dx_b =  c * dx + s * dy
        dy_b = -s * dx + c * dy

        # Distance Normalization
        dist = torch.sqrt(dx_b**2 + dy_b**2 + 1e-6)
        dist_norm = torch.clamp((dist / self.max_distance_diagonal) * 2.0 - 1.0, -1.0, 1.0)

        # Angle to goal
        angle_to_goal = torch.atan2(dy_b, dx_b)
        angle_norm = angle_to_goal / torch.pi

        # Yaw Error (Orientation)
        yaw_diff = diff[:, 2]
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        yaw_error_norm = yaw_diff / torch.pi

        goal_context = torch.stack([dist_norm, angle_norm, yaw_error_norm], dim=1)

        # Fusion
        fused = torch.cat([global_feat, hist_feat, goal_context], dim=-1)

        return self.fusion_mlp(fused)