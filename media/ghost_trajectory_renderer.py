#!/usr/bin/env python3
"""
Ghost Trajectory Renderer
"""

import os
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for offscreen rendering
import matplotlib.pyplot as plt
from typing import List, Optional


def _get_robot_geom_ids(model: mujoco.MjModel, root_body_name: str = "mobile_base") -> set:
    """Find all geom IDs belonging to the robot body tree."""
    root_bid = model.body(root_body_name).id
    robot_body_ids = {root_bid}

    # Walk the body tree to find all children of the root body
    for bid in range(model.nbody):
        parent = bid
        while parent > 0:
            if parent == root_bid:
                robot_body_ids.add(bid)
                break
            parent = model.body_parentid[parent]

    # Collect all geom IDs belonging to these bodies
    robot_geom_ids = set()
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] in robot_body_ids:
            robot_geom_ids.add(gid)

    return robot_geom_ids


def render_ghost_trajectory(
    model: mujoco.MjModel,
    episode_qpos: List[np.ndarray],
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    subsample: int = 5,
    alpha: float = 0.4,
    temporal_fade: bool = False,
    cam_lookat: Optional[List[float]] = None,
    cam_distance: float = 18.0,
    cam_azimuth: float = 90.0,
    cam_elevation: float = -90.0,
    dpi: int = 300,
) -> None:
    """
    Render a ghosted trajectory image by compositing the robot at multiple timesteps.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model (loaded from XML). Must match the qpos recorded during inference.
    episode_qpos : list of np.ndarray
        List of qpos vectors recorded at each simulation step.
    output_path : str
        File path for the output PNG image.
    width, height : int
        Resolution of the rendered image in pixels.
    subsample : int
        Take every N-th state for rendering (controls trajectory density).
    alpha : float
        Base alpha blending coefficient per frame (0..1).
    temporal_fade : bool
        If True, earlier poses are more transparent and later poses more opaque,
        creating a fading trail effect.
    cam_lookat : list of float, optional
        Camera look-at point [x, y, z]. Defaults to center of world bounds.
    cam_distance : float
        Camera distance from the look-at point.
    cam_azimuth : float
        Camera azimuth angle in degrees. 90 = top-down along +Y.
    cam_elevation : float
        Camera elevation angle in degrees. -90 = looking straight down.
    dpi : int
        Output image DPI for matplotlib.
    """
    if len(episode_qpos) == 0:
        print("[Ghost Renderer] No qpos data to render. Skipping.")
        return

    # Default camera look-at: center of the world bounds [-4,-4] to [10,10]
    if cam_lookat is None:
        cam_lookat = [3.0, 3.0, 0.0]

    # Create fresh MjData — fully decoupled from simulation
    data = mujoco.MjData(model)

    # Resize the offscreen framebuffer to match the requested resolution
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)

    # Create offscreen renderer
    renderer = mujoco.Renderer(model, height=height, width=width)

    # Configure fixed camera
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = cam_lookat
    cam.distance = cam_distance
    cam.azimuth = cam_azimuth
    cam.elevation = cam_elevation

    # Configure visualization options — hide LiDAR/rangefinder rays
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    # ---- Identify robot geoms ----
    robot_geom_ids = _get_robot_geom_ids(model)

    # ---- Step 1: Render clean background (robot geoms made invisible) ----
    # Save original alpha values and make robot geoms transparent
    original_rgba = {}
    for gid in robot_geom_ids:
        original_rgba[gid] = model.geom_rgba[gid, 3].copy()
        model.geom_rgba[gid, 3] = 0.0  # Fully transparent

    # Place robot at first pose position (for correct shadows on obstacles, etc.)
    qpos_0 = episode_qpos[0]
    data.qpos[:len(qpos_0)] = qpos_0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    background = renderer.render().astype(np.float32) / 255.0  # (H, W, 3)

    # Restore robot geom alpha values
    for gid, a in original_rgba.items():
        model.geom_rgba[gid, 3] = a

    # Subsample the trajectory
    # Start from index 1 to skip the initial state (may contain reset artifact)
    sampled_indices = list(range(1, len(episode_qpos), subsample))
    if sampled_indices[-1] != len(episode_qpos) - 1:
        sampled_indices.append(len(episode_qpos) - 1)
    n_frames = len(sampled_indices)

    # Start with clean background as the base image
    final = background.copy()

    print(f"[Ghost Renderer] Rendering {n_frames} frames from {len(episode_qpos)} total steps (subsample={subsample})...")

    # ---- Step 2: Composite robot at each sampled pose ----
    for frame_idx, qpos_idx in enumerate(sampled_indices):
        # Compute per-frame alpha
        if temporal_fade:
            t = frame_idx / max(n_frames - 1, 1)
            alpha_i = alpha * (0.3 + 0.7 * t)
        else:
            alpha_i = alpha

        # Set state
        qpos = episode_qpos[qpos_idx]
        data.qpos[:len(qpos)] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # Render RGB frame
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        frame = renderer.render().astype(np.float32) / 255.0

        # Render segmentation to get pixel-perfect robot mask
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        seg = renderer.render()  # (H, W, 2): [geom_id, type]
        renderer.disable_segmentation_rendering()

        # Build robot mask from segmentation geom IDs
        geom_ids = seg[:, :, 0]
        robot_mask = np.zeros((height, width, 1), dtype=np.float32)
        for gid in robot_geom_ids:
            robot_mask[geom_ids == gid] = 1.0


        # Composite only robot pixels onto the final image
        final = final * (1.0 - robot_mask * alpha_i) + frame * robot_mask * alpha_i

    final_image = np.clip(final, 0.0, 1.0)

    # Save with matplotlib at high resolution
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.imshow(final_image)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Clean up renderer
    renderer.close()

    print(f"[Ghost Renderer] Saved ghost trajectory to: {output_path}")
