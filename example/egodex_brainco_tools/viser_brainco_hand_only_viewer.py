#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize only retargeted BrainCo hands from an EgoDex HDF5 file."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
for path in (THIS_DIR, REPO_ROOT / "egodex_wuji_tools"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from egodex_brainco_common import (  # noqa: E402
    BRAINCO_ASSET_DIR,
    BraincoRetargeter,
    brainco_hand_only_scene_position,
    brainco_hand_only_scene_rotation,
    brainco_fk_points,
    egodex_vp25_to_brainco_local,
    make_brainco_line_segments,
    rotation_matrix_to_wxyz,
)
from egodex_wuji_common import egodex_vp25_positions, hand_confidence_ok, require_h5py  # noqa: E402
from viser_hdf5_skeleton_viewer import (  # noqa: E402
    extract_position,
    extract_rotation_matrix,
    get_confidence_group,
    get_transform_group,
    infer_num_frames,
    print_hdf5_tree,
)


COLORS = {
    "left": np.array([240, 110, 190], dtype=np.uint8),
    "right": np.array([110, 240, 180], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    default_config = THIS_DIR / "config" / "brainco_vector.yml"
    parser = argparse.ArgumentParser(
        description="Visualize only EgoDex-retargeted BrainCo hands with base and wrist frames."
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="Path to EgoDex hdf5 file.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence.")
    parser.add_argument("--print-tree", action="store_true", help="Print HDF5 tree and exit.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--frame", type=int, default=None, help="Visualize one fixed frame.")
    parser.add_argument("--no-filter", action="store_true", help="Disable retargeting low-pass filter.")
    parser.add_argument("--no-brainco-mesh", action="store_true", help="Do not load BrainCo URDF mesh.")
    parser.add_argument("--hide-fk", action="store_true", help="Hide compact BrainCo FK skeleton.")
    parser.add_argument("--hide-frames", action="store_true", help="Hide hand pose base and wrist frames.")
    parser.add_argument("--config", type=Path, default=default_config, help="BrainCo retargeting YAML.")
    parser.add_argument(
        "--wrist-y-offset",
        type=float,
        default=0.01,
        help="Offset from BrainCo palm/base frame to wrist frame along local +Y, in meters.",
    )
    parser.add_argument(
        "--wrist-axis-preset",
        choices=("egodex-to-brainco", "identity"),
        default="egodex-to-brainco",
        help="Fixed local wrist-axis conversion from EgoDex to BrainCo.",
    )
    return parser.parse_args()


def remove_handles(handles: dict, keys) -> None:
    for key in keys:
        handle = handles.pop(key, None)
        if handle is not None:
            handle.remove()


def set_urdf_visible(urdfs: dict, visible: bool) -> None:
    for urdf in urdfs.values():
        urdf.show_visual = bool(visible)


def rot_x(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def rot_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


LEFT_WRIST_FROM_BASE = rot_z(-np.pi / 2.0)
RIGHT_WRIST_FROM_BASE = rot_x(np.pi) @ rot_z(np.pi / 2.0)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)

    try:
        import viser
    except ImportError as exc:
        raise SystemExit("Missing dependency: viser. Install it in the egodex environment.") from exc

    h5py = require_h5py()
    h5_file = h5py.File(args.hdf5, "r")
    if args.print_tree:
        print_hdf5_tree(h5_file)
        return

    transform_group = get_transform_group(h5_file)
    get_confidence_group(h5_file)
    num_frames = infer_num_frames(transform_group)
    retargeter = BraincoRetargeter(args.config)

    print(f"[INFO] Loaded: {args.hdf5}", flush=True)
    print(f"[INFO] Retarget config: {args.config}", flush=True)
    print(f"[INFO] Number of frames: {num_frames}", flush=True)
    print(f"[INFO] Open browser: http://localhost:{args.port}", flush=True)
    print("[INFO] Scene uses the original HDF5/world coordinate frame.", flush=True)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.005)
    server.scene.add_grid("/world/grid", width=1.4, height=1.0)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = np.array([0.0, -1.5, 0.8])
        client.camera.look_at = np.array([0.0, 0.0, 0.18])
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    with server.gui.add_folder("Playback"):
        gui_play = server.gui.add_checkbox("Play", initial_value=args.frame is None)
        gui_frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0 if args.frame is None else max(0, min(args.frame, num_frames - 1)),
        )
        gui_fps = server.gui.add_slider("FPS", min=1, max=120, step=1, initial_value=int(args.fps))
        gui_min_conf = server.gui.add_slider(
            "Min confidence",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=args.min_conf,
        )
        gui_show_fk = server.gui.add_checkbox("Show BrainCo FK", initial_value=not args.hide_fk)
        gui_show_mesh = server.gui.add_checkbox("Show BrainCo mesh", initial_value=not args.no_brainco_mesh)
        gui_show_frames = server.gui.add_checkbox("Show pose/wrist frames", initial_value=not args.hide_frames)

    brainco_urdfs = {}
    if not args.no_brainco_mesh:
        try:
            from viser.extras import ViserUrdf

            for side in ("left", "right"):
                urdf_path = BRAINCO_ASSET_DIR / "brainco_hand" / f"brainco_{side}.urdf"
                brainco_urdfs[side] = ViserUrdf(
                    server,
                    urdf_path,
                    root_node_name=f"/world/brainco/{side}/mesh",
                    mesh_color_override=(0.75, 0.78, 0.82, 0.55),
                    load_meshes=True,
                    load_collision_meshes=False,
                )
        except Exception as exc:
            print(f"[WARN] Failed to load BrainCo URDF mesh; FK skeleton only: {exc}", flush=True)
            brainco_urdfs = {}

    handles = {}

    # def to_scene_position(position: np.ndarray) -> np.ndarray:
    #     return np.asarray(position, dtype=np.float32)

    # def to_scene_rotation(rotation: np.ndarray) -> np.ndarray:
    #     return np.asarray(rotation, dtype=np.float32)

        
    def to_scene_position(position: np.ndarray) -> np.ndarray:
        return brainco_hand_only_scene_position(position)


    def to_scene_rotation(rotation: np.ndarray) -> np.ndarray:
        return brainco_hand_only_scene_rotation(rotation)


    def draw_points_and_lines(name: str, points: np.ndarray, color: np.ndarray) -> None:
        remove_handles(handles, [f"{name}_points", f"{name}_lines"])
        handles[f"{name}_points"] = server.scene.add_point_cloud(
            f"{name}/points",
            points=points.astype(np.float32),
            colors=np.tile(color[None, :], (points.shape[0], 1)),
            point_size=0.011,
        )
        handles[f"{name}_lines"] = server.scene.add_line_segments(
            f"{name}/lines",
            points=make_brainco_line_segments(points),
            colors=color,
            line_width=2.5,
        )

    def update_scene(frame_idx: int) -> None:
        for side in ("left", "right"):
            hand_name = f"{side}Hand"
            fk_name = f"/world/brainco/{side}/fk"
            side_keys = {
                f"{fk_name}_points",
                f"{fk_name}_lines",
                f"{side}_pose_base_frame",
                f"{side}_wrist_frame",
            }
            if hand_name not in transform_group:
                remove_handles(handles, side_keys)
                continue
            if not hand_confidence_ok(h5_file, frame_idx, side, float(gui_min_conf.value)):
                remove_handles(handles, side_keys)
                if side in brainco_urdfs:
                    brainco_urdfs[side].show_visual = False
                continue
            
            # HDF5 原始 hand pose
            wrist_pos = extract_position(transform_group[hand_name], frame_idx)
            wrist_rot = extract_rotation_matrix(transform_group[hand_name], frame_idx)
            if wrist_pos is None or wrist_rot is None:
                remove_handles(handles, side_keys)
                continue

            vp25 = egodex_vp25_positions(h5_file, frame_idx, side)
            vp25_brainco_local, brainco_base_rot = egodex_vp25_to_brainco_local(
                vp25,
                wrist_pos,
                wrist_rot,
                side,
                args.wrist_axis_preset,
            )
            result = retargeter.retarget(side, vp25_brainco_local, apply_filter=not args.no_filter)
            fk_local = brainco_fk_points(retargeter.retargeting[side], result.qpos_full, side)

            base_pos = np.asarray(wrist_pos, dtype=np.float32) 
            base_rot = np.asarray(brainco_base_rot, dtype=np.float32) 
            scene_base_pos = to_scene_position(base_pos)
            scene_base_rot = to_scene_rotation(base_rot)
            scene_pose_rot = to_scene_rotation(wrist_rot)


            fk_base = fk_local - fk_local[0:1]
            fk_world = (fk_base @ base_rot.T) + base_pos
            fk_scene = to_scene_position(fk_world)

            if gui_show_fk.value:
                draw_points_and_lines(fk_name, fk_scene, COLORS[side])
            else:
                remove_handles(handles, [f"{fk_name}_points", f"{fk_name}_lines"])

            if side in brainco_urdfs:
                urdf = brainco_urdfs[side]
                urdf.update_cfg(result.qpos_hardware)
                for root_handle_name in ("_visual_root_frame", "_collision_root_frame"):
                    root_handle = getattr(urdf, root_handle_name, None)
                    if root_handle is not None:
                        root_handle.position = scene_base_pos
                        root_handle.wxyz = rotation_matrix_to_wxyz(scene_base_rot)
                urdf.show_visual = bool(gui_show_mesh.value)    

            remove_handles(handles, [f"{side}_pose_base_frame", f"{side}_wrist_frame"])
            if gui_show_frames.value:
                # 人工构造的 BrainCo wrist frame  手腕坐标系
                wrist_pos_world = base_pos + base_rot @ np.array(
                    [0.0, float(args.wrist_y_offset), 0.0],
                    dtype=np.float32,
                )
                wrist_rot_local = RIGHT_WRIST_FROM_BASE if side == "right" else LEFT_WRIST_FROM_BASE
                wrist_rot_world = base_rot @ wrist_rot_local
                handles[f"{side}_wrist_frame"] = server.scene.add_frame(
                    f"/world/brainco/{side}/wrist_frame",
                    position=to_scene_position(wrist_pos_world),
                    wxyz=rotation_matrix_to_wxyz(to_scene_rotation(wrist_rot_world)),
                    axes_length=0.085,
                    axes_radius=0.0035,
                )

        if not gui_show_mesh.value:
            set_urdf_visible(brainco_urdfs, False)

    def redraw(_event=None) -> None:
        update_scene(int(gui_frame.value))

    for control in (gui_frame, gui_min_conf, gui_show_fk, gui_show_mesh, gui_show_frames):
        control.on_update(redraw)

    update_scene(int(gui_frame.value))
    last_time = time.time()
    while True:
        if gui_play.value:
            now = time.time()
            target_dt = 1.0 / max(float(gui_fps.value), 1.0)
            if now - last_time >= target_dt:
                next_frame = int(gui_frame.value) + 1
                if next_frame >= num_frames:
                    next_frame = 0 if args.loop else num_frames - 1
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
