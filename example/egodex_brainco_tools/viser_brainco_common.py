"""Shared viser helpers for EgoDex -> BrainCo example viewers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from egodex_brainco_common import BRAINCO_ASSET_DIR, rotation_matrix_to_wxyz


def remove_handles(handles: dict, keys) -> None:
    for key in keys:
        handle = handles.pop(key, None)
        if handle is not None:
            handle.remove()


def set_urdf_visible(urdfs: dict, visible: bool) -> None:
    for urdf in urdfs.values():
        urdf.show_visual = bool(visible)


def load_brainco_urdfs(
    server,
    *,
    root_node_name: str = "/world/brainco/{side}/mesh",
    mesh_color_override=(0.75, 0.78, 0.82, 0.55),
    asset_dir: Path = BRAINCO_ASSET_DIR,
    warning_suffix: str = "",
) -> dict:
    """Load left/right BrainCo URDF meshes for a viser scene."""
    try:
        from viser.extras import ViserUrdf

        urdfs = {}
        for side in ("left", "right"):
            urdf_path = asset_dir / "brainco_hand" / f"brainco_{side}.urdf"
            urdfs[side] = ViserUrdf(
                server,
                urdf_path,
                root_node_name=root_node_name.format(side=side),
                mesh_color_override=mesh_color_override,
                load_meshes=True,
                load_collision_meshes=False,
            )
        return urdfs
    except Exception as exc:
        suffix = f"; {warning_suffix}" if warning_suffix else ""
        print(f"[WARN] Failed to load BrainCo URDF mesh{suffix}: {exc}", flush=True)
        return {}


def draw_points_and_lines(
    server,
    handles: dict,
    name: str,
    points: np.ndarray,
    color: np.ndarray,
    edges_fn: Callable[[np.ndarray], np.ndarray],
    *,
    point_size: float = 0.009,
    line_width: float = 2.5,
) -> None:
    remove_handles(handles, [f"{name}_points", f"{name}_lines"])
    handles[f"{name}_points"] = server.scene.add_point_cloud(
        f"{name}/points",
        points=np.asarray(points, dtype=np.float32),
        colors=np.tile(color[None, :], (points.shape[0], 1)),
        point_size=point_size,
    )
    handles[f"{name}_lines"] = server.scene.add_line_segments(
        f"{name}/lines",
        points=edges_fn(points),
        colors=color,
        line_width=line_width,
    )


def update_urdf_pose(urdf, qpos: np.ndarray, position: np.ndarray, rotation: np.ndarray) -> None:
    urdf.update_cfg(np.asarray(qpos, dtype=np.float32))
    wxyz = rotation_matrix_to_wxyz(rotation)
    for root_handle_name in ("_visual_root_frame", "_collision_root_frame"):
        root_handle = getattr(urdf, root_handle_name, None)
        if root_handle is not None:
            root_handle.position = np.asarray(position, dtype=np.float32)
            root_handle.wxyz = wxyz


def run_playback_loop(gui_play, gui_frame, gui_fps, num_frames: int, loop: bool, update_scene) -> None:
    last_time = time.time()
    while True:
        if gui_play.value:
            now = time.time()
            target_dt = 1.0 / max(float(gui_fps.value), 1.0)
            if now - last_time >= target_dt:
                next_frame = int(gui_frame.value) + 1
                if next_frame >= num_frames:
                    next_frame = 0 if loop else num_frames - 1
                    if not loop:
                        gui_play.value = False
                gui_frame.value = next_frame
                update_scene(next_frame)
                last_time = now
        else:
            last_time = time.time()
        time.sleep(0.001)
