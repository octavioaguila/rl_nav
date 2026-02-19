#!/usr/bin/env python3
"""
Trajectory Video Renderer
"""

import numpy as np
import mujoco
import imageio
from typing import List, Optional


def render_trajectory_video(
    model: mujoco.MjModel,
    episode_qpos: List[np.ndarray],
    output_path: str,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    cam_lookat: Optional[List[float]] = None,
    cam_distance: float = 18.0,
    cam_azimuth: float = 90.0,
    cam_elevation: float = -90.0,
) -> None:
    """
    Render an MP4 video replaying the recorded qpos trajectory.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model (loaded from XML).
    episode_qpos : list of np.ndarray
        List of qpos vectors recorded at each simulation step.
    output_path : str
        File path for the output MP4 video.
    width, height : int
        Resolution of each video frame in pixels.
    fps : int
        Frames per second for the output video.
    cam_lookat : list of float, optional
        Camera look-at point [x, y, z].
    cam_distance : float
        Camera distance from the look-at point.
    cam_azimuth : float
        Camera azimuth angle in degrees.
    cam_elevation : float
        Camera elevation angle in degrees.
    """
    if len(episode_qpos) == 0:
        print("[Video Renderer] No qpos data to render. Skipping.")
        return

    if cam_lookat is None:
        cam_lookat = [3.0, 3.0, 0.0]

    # Create fresh MjData
    data = mujoco.MjData(model)

    # Resize the offscreen framebuffer
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

    # Hide LiDAR/rangefinder rays
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    n_frames = len(episode_qpos)
    print(f"[Video Renderer] Rendering {n_frames} frames at {fps} fps...")

    writer = imageio.get_writer(output_path, fps=fps)

    for qpos in episode_qpos:
        data.qpos[:len(qpos)] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        frame = renderer.render()  # uint8 (H, W, 3)
        writer.append_data(frame)

    writer.close()
    renderer.close()

    print(f"[Video Renderer] Saved video to: {output_path}")
