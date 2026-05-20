#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize exported BrainCo loop JSON data with viser."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from egodex_brainco_common import (  # noqa: E402
    BRAINCO_ASSET_DIR,
    BRAINCO_HARDWARE_JOINT_ORDER,
    BRAINCO_MAX_QPOS,
    brainco_wrist_to_base_pose,
    rotation_matrix_to_wxyz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize exported BrainCo recomputed_ee_fullbody.json.")
    parser.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "egodex_example" / "clean_cups" / "0_brainco_loop" / "recomputed_ee_fullbody.json",
        help="Path to recomputed_ee_fullbody.json.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS. Defaults to JSON info.fps.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--frame", type=int, default=None, help="Visualize one fixed frame.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--no-mesh", action="store_true", help="Hide BrainCo URDF meshes.")
    parser.add_argument("--show-frames", action=argparse.BooleanOptionalAction, default=True, help="Show EE frames.")
    parser.add_argument("--mesh-alpha", type=float, default=0.6, help="BrainCo mesh alpha.")
    return parser.parse_args()


def rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def rpy_to_matrix(xyzrpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = xyzrpy[3:6]
    return (rot_z(float(yaw)) @ rot_y(float(pitch)) @ rot_x(float(roll))).astype(np.float32)


def pose_from_xyzrpy(xyzrpy: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(xyzrpy, dtype=np.float32)
    if pose.shape != (6,):
        raise ValueError(f"Expected xyzrpy pose shape (6,), got {pose.shape}")
    return pose[:3].copy(), rpy_to_matrix(pose)


def fig6d_to_qpos(fig6d: list[float] | np.ndarray, hand_action: str) -> np.ndarray:
    qpos = np.asarray(fig6d, dtype=np.float32)
    if qpos.shape != (6,):
        raise ValueError(f"Expected fig6d shape (6,), got {qpos.shape}")
    if hand_action == "normalized":
        qpos = np.clip(qpos, 0.0, 1.0) * BRAINCO_MAX_QPOS
    return qpos.astype(np.float32)


def remove_handles(handles: dict, keys) -> None:
    for key in keys:
        handle = handles.pop(key, None)
        if handle is not None:
            handle.remove()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if not args.json.exists():
        raise FileNotFoundError(args.json)

    try:
        import viser
    except ImportError as exc:
        raise SystemExit("Missing dependency: viser. Install it in the egodex environment.") from exc

    with args.json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {args.json}")

    info = data.get("info", {})
    fps = float(args.fps if args.fps is not None else info.get("fps", 30.0))
    hand_action = str(info.get("hand_action", "normalized"))
    ee_pose_frame = str(info.get("ee_pose_frame", "brainco_base"))
    wrist_y_offset = float(info.get("wrist_y_offset", 0.01))

    print(f"[INFO] Loaded: {args.json}", flush=True)
    print(f"[INFO] Frames: {len(frames)}", flush=True)
    print(f"[INFO] Hand action format: {hand_action}", flush=True)
    print(f"[INFO] EE pose frame: {ee_pose_frame}", flush=True)
    print(f"[INFO] Joint order: {info.get('joint_order', list(BRAINCO_HARDWARE_JOINT_ORDER))}", flush=True)
    print(f"[INFO] Open browser: http://localhost:{args.port}", flush=True)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("/world", axes_length=0.18, axes_radius=0.004)
    server.scene.add_grid("/world/grid", width=1.0, height=1.0)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = np.array([0.0, -1.2, 0.55])
        client.camera.look_at = np.array([0.0, 0.0, 0.0])
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    with server.gui.add_folder("Playback"):
        gui_play = server.gui.add_checkbox("Play", initial_value=args.frame is None)
        gui_frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(frames) - 1,
            step=1,
            initial_value=0 if args.frame is None else max(0, min(args.frame, len(frames) - 1)),
        )
        gui_fps = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=int(fps))
        gui_show_mesh = server.gui.add_checkbox("BrainCo mesh", initial_value=not args.no_mesh)
        gui_show_frames = server.gui.add_checkbox("EE frames", initial_value=args.show_frames)

    brainco_urdfs = {}
    if not args.no_mesh:
        try:
            from viser.extras import ViserUrdf

            alpha = float(np.clip(args.mesh_alpha, 0.0, 1.0))
            for side in ("left", "right"):
                urdf_path = BRAINCO_ASSET_DIR / "brainco_hand" / f"brainco_{side}.urdf"
                brainco_urdfs[side] = ViserUrdf(
                    server,
                    urdf_path,
                    root_node_name=f"/world/brainco/{side}/mesh",
                    mesh_color_override=(0.72, 0.76, 0.82, alpha),
                    load_meshes=True,
                    load_collision_meshes=False,
                )
        except Exception as exc:
            print(f"[WARN] Failed to load BrainCo URDF mesh: {exc}", flush=True)
            brainco_urdfs = {}

    handles = {}

    def update_scene(frame_idx: int) -> None:
        frame = frames[frame_idx]
        for side in ("left", "right"):
            pose_key = "state_ee"
            fig_key = f"{side}_fig6d_cmd" if f"{side}_fig6d_cmd" in frame else f"{side}_fig6d"
            stored_pos, stored_rot = pose_from_xyzrpy(frame[pose_key][side])
            if ee_pose_frame.startswith("brainco_wrist"):
                if "brainco_base" in frame and side in frame["brainco_base"]:
                    mesh_pos, mesh_rot = pose_from_xyzrpy(frame["brainco_base"][side])
                else:
                    mesh_pos, mesh_rot = brainco_wrist_to_base_pose(
                        stored_pos,
                        stored_rot,
                        side,
                        wrist_y_offset,
                    )
            else:
                mesh_pos, mesh_rot = stored_pos, stored_rot
            qpos = fig6d_to_qpos(frame[fig_key], hand_action)

            if side in brainco_urdfs:
                urdf = brainco_urdfs[side]
                urdf.update_cfg(qpos)
                for root_handle_name in ("_visual_root_frame", "_collision_root_frame"):
                    root_handle = getattr(urdf, root_handle_name, None)
                    if root_handle is not None:
                        root_handle.position = mesh_pos
                        root_handle.wxyz = rotation_matrix_to_wxyz(mesh_rot)
                urdf.show_visual = bool(gui_show_mesh.value)

            remove_handles(handles, [f"{side}_ee_frame"])
            if gui_show_frames.value:
                handles[f"{side}_ee_frame"] = server.scene.add_frame(
                    f"/world/brainco/{side}/ee_frame",
                    position=stored_pos,
                    wxyz=rotation_matrix_to_wxyz(stored_rot),
                    axes_length=0.09,
                    axes_radius=0.0035,
                )

        if not gui_show_mesh.value:
            for urdf in brainco_urdfs.values():
                urdf.show_visual = False

    def redraw(_event=None) -> None:
        update_scene(int(gui_frame.value))

    for control in (gui_frame, gui_show_mesh, gui_show_frames):
        control.on_update(redraw)

    update_scene(int(gui_frame.value))
    last_time = time.time()
    while True:
        if gui_play.value:
            now = time.time()
            target_dt = 1.0 / max(float(gui_fps.value), 1.0)
            if now - last_time >= target_dt:
                next_frame = int(gui_frame.value) + 1
                if next_frame >= len(frames):
                    next_frame = 0 if args.loop else len(frames) - 1
                    if not args.loop:
                        gui_play.value = False
                gui_frame.value = next_frame
                update_scene(next_frame)
                last_time = now
        else:
            last_time = time.time()
        time.sleep(0.001)


if __name__ == "__main__":
    main()
