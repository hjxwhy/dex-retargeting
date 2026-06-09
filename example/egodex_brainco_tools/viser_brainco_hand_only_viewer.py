#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize only retargeted BrainCo hands from an EgoDex HDF5 file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent

from egodex_brainco_common import (  # noqa: E402
    BraincoRetargeter,
    brainco_base_to_wrist_pose,
    brainco_hand_only_scene_position,
    brainco_hand_only_scene_rotation,
    brainco_fk_points,
    make_brainco_line_segments,
    rotation_matrix_to_wxyz,
    retarget_egodex_brainco_frame,
)
from egodex_wuji_common import (  # noqa: E402
    get_confidence_group,
    get_transform_group,
    infer_num_frames,
    print_hdf5_tree,
    require_h5py,
)
from viser_brainco_common import (  # noqa: E402
    draw_points_and_lines,
    load_brainco_urdfs,
    remove_handles,
    run_playback_loop,
    set_urdf_visible,
    update_urdf_pose,
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)

    try:
        import viser
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: viser. Install the example dependencies with "
            "`uv sync --extra torch-cu124 --extra example` or "
            "`uv sync --extra torch-cu128 --extra example`."
        ) from exc

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
        brainco_urdfs = load_brainco_urdfs(
            server,
            mesh_color_override=(0.75, 0.78, 0.82, 0.55),
            warning_suffix="FK skeleton only",
        )

    handles = {}

    # def to_scene_position(position: np.ndarray) -> np.ndarray:
    #     return np.asarray(position, dtype=np.float32)

    # def to_scene_rotation(rotation: np.ndarray) -> np.ndarray:
    #     return np.asarray(rotation, dtype=np.float32)

        
    def to_scene_position(position: np.ndarray) -> np.ndarray:
        return brainco_hand_only_scene_position(position)


    def to_scene_rotation(rotation: np.ndarray) -> np.ndarray:
        return brainco_hand_only_scene_rotation(rotation)


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
                remove_handles(handles, side_keys)
                if side in brainco_urdfs:
                    brainco_urdfs[side].show_visual = False
                continue

            fk_local = brainco_fk_points(retargeter.retargeting[side], frame.result.qpos_full, side)
            base_pos = frame.wrist_position
            base_rot = frame.brainco_base_rotation
            scene_base_pos = to_scene_position(base_pos)
            scene_base_rot = to_scene_rotation(base_rot)

            fk_base = fk_local - fk_local[0:1]
            fk_world = (fk_base @ base_rot.T) + base_pos
            fk_scene = to_scene_position(fk_world)

            if gui_show_fk.value:
                draw_points_and_lines(
                    server,
                    handles,
                    fk_name,
                    fk_scene,
                    COLORS[side],
                    make_brainco_line_segments,
                    point_size=0.011,
                )
            else:
                remove_handles(handles, [f"{fk_name}_points", f"{fk_name}_lines"])

            if side in brainco_urdfs:
                urdf = brainco_urdfs[side]
                update_urdf_pose(urdf, frame.result.qpos_hardware, scene_base_pos, scene_base_rot)
                urdf.show_visual = bool(gui_show_mesh.value)    

            remove_handles(handles, [f"{side}_pose_base_frame", f"{side}_wrist_frame"])
            if gui_show_frames.value:
                wrist_pos_world, wrist_rot_world = brainco_base_to_wrist_pose(
                    base_pos,
                    base_rot,
                    side,
                    args.wrist_y_offset,
                )
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
    run_playback_loop(gui_play, gui_frame, gui_fps, num_frames, args.loop, update_scene)


if __name__ == "__main__":
    main()
