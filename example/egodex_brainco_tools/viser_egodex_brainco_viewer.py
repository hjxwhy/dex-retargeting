#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize EgoDex HDF5 skeleton and BrainCo retargeted hands with viser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent

from egodex_brainco_common import (  # noqa: E402
    BRAINCO_WRIST_OFFSET_LOCAL,
    BraincoRetargeter,
    brainco_base_to_wrist_pose,
    brainco_fk_points,
    make_brainco_line_segments,
    make_vp25_line_segments,
    rotation_matrix_to_wxyz,
    retarget_egodex_brainco_frame,
)
from egodex_wuji_common import (  # noqa: E402
    SKELETON_EDGES,
    collect_joint_positions,
    get_confidence_group,
    get_transform_group,
    infer_num_frames,
    make_line_segments,
    normalize_scene_scale,
    offset_joint_positions,
    print_hdf5_tree,
    require_h5py,
)
from viser_brainco_common import (  # noqa: E402
    draw_points_and_lines,
    load_brainco_urdfs,
    remove_handles,
    run_playback_loop,
    set_urdf_visible,
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

    egodex_offset = np.array([-0.55, 0.0, 0.0], dtype=np.float32) # for visualization only, does not affect retargeting results
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
        brainco_urdfs = load_brainco_urdfs(
            server,
            root_node_name="/world/brainco/{side}_mesh",
            mesh_color_override=(0.75, 0.78, 0.82, 0.45),
            warning_suffix="FK skeleton only",
        )

    handles = {}

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
            frame = retarget_egodex_brainco_frame(
                h5_file,
                transform_group,
                frame_idx,
                side,
                retargeter,
                min_conf=float(gui_min_conf.value),
                axis_preset=args.wrist_axis_preset,
                apply_filter=not args.no_filter,
            )
            if frame is None:
                continue
            fk_local = brainco_fk_points(retargeter.retargeting[side], frame.result.qpos_full, side)

            if gui_show_vp25.value:
                draw_points_and_lines(
                    server,
                    handles,
                    f"/world/egodex/{side}_vp25",
                    frame.vp25_world + egodex_offset,
                    COLORS[f"egodex_{side}"],
                    make_vp25_line_segments,
                )
            else:
                remove_handles(handles, [f"/world/egodex/{side}_vp25_points", f"/world/egodex/{side}_vp25_lines"])

            if gui_world_pose.value:
                brainco_wrist_rot = frame.brainco_base_rotation
                fk_base = fk_local - fk_local[0:1]
                fk_display = (fk_base @ brainco_wrist_rot.T) + frame.wrist_position + brainco_offset
                mesh_pos = frame.wrist_position + brainco_offset
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
                    server,
                    handles,
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
                urdf.update_cfg(frame.result.qpos_hardware)
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
            if gui_show_frames.value:
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
    run_playback_loop(gui_play, gui_frame, gui_fps, num_frames, args.loop, update_scene)


if __name__ == "__main__":
    main()
