#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize EgoDex HDF5 skeleton and BrainCo retargeted hands with viser."""

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
    BRAINCO_WRIST_OFFSET_LOCAL,
    BraincoRetargeter,
    brainco_base_to_wrist_pose,
    brainco_fk_points,
    egodex_vp25_to_brainco_local,
    make_brainco_line_segments,
    make_vp25_line_segments,
    rotation_matrix_to_wxyz,
)
from egodex_wuji_common import egodex_vp25_positions, hand_confidence_ok, require_h5py  # noqa: E402
from viser_hdf5_skeleton_viewer import (  # noqa: E402
    SKELETON_EDGES,
    collect_joint_positions,
    extract_position,
    extract_rotation_matrix,
    get_confidence_group,
    get_transform_group,
    infer_num_frames,
    make_line_segments,
    normalize_scene_scale,
    offset_joint_positions,
    print_hdf5_tree,
)


COLORS = {
    "egodex": np.array([255, 180, 60], dtype=np.uint8),
    "egodex_left": np.array([255, 120, 180], dtype=np.uint8),
    "egodex_right": np.array([120, 220, 255], dtype=np.uint8),
    "brainco_left": np.array([240, 110, 190], dtype=np.uint8),
    "brainco_right": np.array([110, 240, 180], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    default_config = THIS_DIR / "config" / "brainco_vector.yml"
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--local-fk", action="store_true", help="Show BrainCo FK in local base frame.")
    parser.add_argument("--config", type=Path, default=default_config, help="BrainCo retargeting YAML.")
    parser.add_argument(
        "--wrist-y-offset",
        type=float,
        default=float(BRAINCO_WRIST_OFFSET_LOCAL[1]),
        help="Offset from BrainCo palm/base frame to displayed wrist frame along local +Y, in meters.",
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
    conf_group = get_confidence_group(h5_file)
    num_frames = infer_num_frames(transform_group)
    retargeter = BraincoRetargeter(args.config)

    egodex_offset = np.array([-0.55, 0.0, 0.0], dtype=np.float32)
    brainco_offset = np.array([0.55, 0.0, 0.0], dtype=np.float32)

    print(f"[INFO] Loaded: {args.hdf5}", flush=True)
    print(f"[INFO] Retarget config: {args.config}", flush=True)
    print(f"[INFO] Number of frames: {num_frames}", flush=True)
    print(f"[INFO] Open browser: http://localhost:{args.port}", flush=True)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.005)
    server.scene.add_grid("/world/grid", width=1.4, height=1.0)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = np.array([0.0, 2.8, -1.8])
        client.camera.look_at = np.array([0.0, 0.0, 1.1])
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
        gui_show_egodex = server.gui.add_checkbox("Show EgoDex skeleton", initial_value=True)
        gui_show_vp25 = server.gui.add_checkbox("Show EgoDex VP25 hands", initial_value=True)
        gui_show_brainco = server.gui.add_checkbox("Show BrainCo FK", initial_value=True)
        gui_show_mesh = server.gui.add_checkbox("Show BrainCo mesh", initial_value=not args.no_brainco_mesh)
        gui_world_pose = server.gui.add_checkbox("World wrist pose", initial_value=not args.local_fk)
        gui_show_frames = server.gui.add_checkbox("Show wrist frames", initial_value=True)

    brainco_urdfs = {}
    if not args.no_brainco_mesh:
        try:
            from viser.extras import ViserUrdf

            for side in ("left", "right"):
                urdf_path = BRAINCO_ASSET_DIR / "brainco_hand" / f"brainco_{side}.urdf"
                brainco_urdfs[side] = ViserUrdf(
                    server,
                    urdf_path,
                    root_node_name=f"/world/brainco/{side}_mesh",
                    mesh_color_override=(0.75, 0.78, 0.82, 0.45),
                    load_meshes=True,
                    load_collision_meshes=False,
                )
        except Exception as exc:
            print(f"[WARN] Failed to load BrainCo URDF mesh; FK skeleton only: {exc}", flush=True)
            brainco_urdfs = {}

    handles = {}

    def draw_points_and_lines(name: str, points: np.ndarray, color: np.ndarray, edges_fn, point_size=0.009):
        remove_handles(handles, [f"{name}_points", f"{name}_lines"])
        handles[f"{name}_points"] = server.scene.add_point_cloud(
            f"{name}/points",
            points=points.astype(np.float32),
            colors=np.tile(color[None, :], (points.shape[0], 1)),
            point_size=point_size,
        )
        handles[f"{name}_lines"] = server.scene.add_line_segments(
            f"{name}/lines",
            points=edges_fn(points),
            colors=color,
            line_width=2.5,
        )

    def update_scene(frame_idx: int) -> None:
        joint_positions, _joint_confidences = collect_joint_positions(
            transform_group=transform_group,
            conf_group=conf_group,
            frame_idx=frame_idx,
            min_conf=float(gui_min_conf.value),
        )
        if not joint_positions:
            return

        egodex_joint_positions = offset_joint_positions(joint_positions, egodex_offset)

        remove_handles(handles, ["egodex_points", "egodex_lines"])
        if gui_show_egodex.value:
            pts = np.asarray(list(egodex_joint_positions.values()), dtype=np.float32)
            radius = normalize_scene_scale(joint_positions)
            handles["egodex_points"] = server.scene.add_point_cloud(
                "/world/egodex/skeleton_points",
                points=pts,
                colors=np.tile(COLORS["egodex"][None, :], (pts.shape[0], 1)),
                point_size=max(radius, 0.007),
            )
            segments = make_line_segments(egodex_joint_positions, SKELETON_EDGES)
            if segments is not None:
                handles["egodex_lines"] = server.scene.add_line_segments(
                    "/world/egodex/skeleton_lines",
                    points=segments,
                    colors=np.array([80, 160, 255], dtype=np.uint8),
                    line_width=2.0,
                )

        frame_keys = []
        for side in ("left", "right"):
            hand_name = f"{side}Hand"
            if hand_name not in transform_group:
                continue
            ok = hand_confidence_ok(h5_file, frame_idx, side, float(gui_min_conf.value))
            if not ok:
                continue

            vp25 = egodex_vp25_positions(h5_file, frame_idx, side)
            wrist_pos = extract_position(transform_group[hand_name], frame_idx)
            wrist_rot = extract_rotation_matrix(transform_group[hand_name], frame_idx)
            if wrist_rot is not None and wrist_pos is not None:
                vp25_brainco_local, brainco_wrist_rot_for_retarget = egodex_vp25_to_brainco_local(
                    vp25,
                    wrist_pos,
                    wrist_rot,
                    side,
                    args.wrist_axis_preset,
                )
            else:
                vp25_brainco_local = vp25
                brainco_wrist_rot_for_retarget = None
            result = retargeter.retarget(side, vp25_brainco_local, apply_filter=not args.no_filter)
            fk_local = brainco_fk_points(retargeter.retargeting[side], result.qpos_full, side)

            if gui_show_vp25.value:
                draw_points_and_lines(
                    f"/world/egodex/{side}_vp25",
                    vp25 + egodex_offset,
                    COLORS[f"egodex_{side}"],
                    make_vp25_line_segments,
                )
            else:
                remove_handles(handles, [f"/world/egodex/{side}_vp25_points", f"/world/egodex/{side}_vp25_lines"])

            if wrist_rot is not None and gui_world_pose.value:
                brainco_wrist_rot = brainco_wrist_rot_for_retarget
                fk_base = fk_local - fk_local[0:1]
                fk_display = (fk_base @ brainco_wrist_rot.T) + wrist_pos + brainco_offset
                mesh_pos = wrist_pos + brainco_offset
                mesh_wxyz = rotation_matrix_to_wxyz(brainco_wrist_rot)
                frame_pos, frame_rot = brainco_base_to_wrist_pose(
                    mesh_pos,
                    brainco_wrist_rot,
                    side,
                    args.wrist_y_offset,
                )
            else:
                local_shift = brainco_offset + np.array([0.0, -0.16 if side == "left" else 0.16, 0.8])
                fk_display = fk_local + local_shift
                mesh_pos = local_shift
                mesh_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                frame_pos = mesh_pos
                frame_rot = np.eye(3, dtype=np.float32)

            if gui_show_brainco.value:
                draw_points_and_lines(
                    f"/world/brainco/{side}_fk",
                    fk_display,
                    COLORS[f"brainco_{side}"],
                    make_brainco_line_segments,
                    point_size=0.011,
                )
            else:
                remove_handles(handles, [f"/world/brainco/{side}_fk_points", f"/world/brainco/{side}_fk_lines"])

            if side in brainco_urdfs:
                urdf = brainco_urdfs[side]
                urdf.update_cfg(result.qpos_hardware)
                for root_handle_name in ("_visual_root_frame", "_collision_root_frame"):
                    root_handle = getattr(urdf, root_handle_name, None)
                    if root_handle is not None:
                        root_handle.position = mesh_pos
                        root_handle.wxyz = mesh_wxyz
                urdf.show_visual = bool(gui_show_mesh.value)

            frame_key = f"{side}_frame"
            frame_keys.append(frame_key)
            old = handles.pop(frame_key, None)
            if old is not None:
                old.remove()
            if gui_show_frames.value and wrist_rot is not None:
                handles[frame_key] = server.scene.add_frame(
                    f"/world/brainco/{side}_wrist_frame",
                    position=frame_pos,
                    wxyz=rotation_matrix_to_wxyz(frame_rot),
                    axes_length=0.09,
                    axes_radius=0.004,
                )

        for key in ("left_frame", "right_frame"):
            if key not in frame_keys and key in handles:
                handles.pop(key).remove()

        if not gui_show_mesh.value:
            set_urdf_visible(brainco_urdfs, False)

    def redraw(_event=None) -> None:
        update_scene(int(gui_frame.value))

    for control in (
        gui_frame,
        gui_min_conf,
        gui_show_egodex,
        gui_show_vp25,
        gui_show_brainco,
        gui_show_mesh,
        gui_world_pose,
        gui_show_frames,
    ):
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
