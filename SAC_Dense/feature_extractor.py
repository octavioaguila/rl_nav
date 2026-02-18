# point_net_extractor.py  ─────────────────────────────────────────────────────
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FeatureExtractor(BaseFeaturesExtractor):
    """
    Implements the two-stage PointNet-style encoder with the distance channel
    converted on-the-fly:
        d_norm ∈ [-1,1]  →  d_m = max_range*(d_norm+1)/2
        f_dist = 1 / (d_m - β)

    Args
    ----
    features_dim : size of the output embedding handed to the SAC heads
    n_lidar      : number of rays in the observation
    max_range    : same constant you use inside the env (default 6 m), this has to be the same as the cutoff in the XML file.
    beta         : small offset to avoid 1/0 (default 0.25 m)
    """
    def __init__(self, observation_space, *, features_dim: int = 256, n_lidar: int = 449, max_range: float = 20.0, beta: float = 0.25):
        super().__init__(observation_space, features_dim)

        self.n_lidar   = int(n_lidar)
        self.n_feat    = 3
        self.extra_dim = 5                 # goal (3) + velocity (2) - changed from 4 to 5
        self.max_range = float(max_range)
        self.beta      = float(beta)

        # ------------ shared point-wise MLP (32-64-128) ------------- #
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 32),  nn.LeakyReLU(),
            nn.Linear(32, 64), nn.LeakyReLU(),
            nn.Linear(64, 128), nn.LeakyReLU(),
        )

        # ------------ fusion MLP (128 + 4 → features_dim) ----------- #
        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 + self.extra_dim, 128), nn.LeakyReLU(),
            nn.Linear(128, 128),                  nn.LeakyReLU(),
            nn.Linear(128, 128),                  nn.LeakyReLU(),
            nn.Linear(128, features_dim),         nn.LeakyReLU(),
        )

    # ------------------------------------------------------------------ #
    def forward(self, obs):
        """
        obs: (B, n_lidar*3 + 5)
        returns: (B, features_dim)
        """
        B = obs.size(0)

        # ----------- split observation -------------------------------- #
        lidar_flat = obs[:, : self.n_lidar * self.n_feat]            # (B, N*3)
        goal_vel   = obs[:, self.n_lidar * self.n_feat :]            # (B, 5)

        # reshape LiDAR to (B, N, 3)
        pts = lidar_flat.view(B, self.n_lidar, self.n_feat)          # (B,N,3)

        # ------------- reconstruct distance & build new feature ----- #
        sincos = pts[..., :2]                                        # (B,N,2)
        d_norm = pts[..., 2]                                         # (B,N)

        d_m    = self.max_range * (d_norm + 1.0) * 0.5               # metres
        inv_d  = 1.0 / torch.clamp(d_m - self.beta, min=1e-4)        # (B,N)

        pts_proc = torch.cat([sincos, inv_d.unsqueeze(-1)], dim=-1)  # (B,N,3)

        # ------------- point-wise MLP  ------------------------------ #
        pts_feat = self.point_mlp(pts_proc.view(-1, 3))              # (B*N,128)
        pts_feat = pts_feat.view(B, self.n_lidar, 128)               # (B,N,128)

        global_feat, _ = torch.max(pts_feat, dim=1)                  # (B,128)

        fused = torch.cat([global_feat, goal_vel], dim=-1)           # (B, 133)
        return self.fusion_mlp(fused)                                # (B, features_dim)
