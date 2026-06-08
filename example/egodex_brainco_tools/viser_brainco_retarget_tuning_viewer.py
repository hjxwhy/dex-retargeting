#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-scene BrainCo retarget tuning viewer.

Shows only the two EgoDex hands overlaid with retargeted BrainCo hand meshes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from egodex_brainco_common import (  # noqa: E402
    BRAINCO_CONFIG_PATH,
    BRAINCO_HARDWARE_JOINT_ORDER,
    BraincoRetargeter,
    brainco_fk_points,
    make_brainco_line_segments,
    make_vp25_line_segments,
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


EGODEX_COLORS = {
    "left": np.array([255, 120, 180], dtype=np.uint8),
    "right": np.array([120, 220, 255], dtype=np.uint8),
}
BRAINCO_COLORS = {
    "left": np.array([240, 90, 170], dtype=np.uint8),
    "right": np.array([80, 230, 170], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay EgoDex VP25 hands and retargeted BrainCo meshes for tuning."
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="Path to EgoDex HDF5 file.")
    parser.add_argument("--config", type=Path, default=BRAINCO_CONFIG_PATH, help="BrainCo retargeting YAML.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum hand confidence.")
    parser.add_argument("--print-tree", action="store_true", help="Print HDF5 tree and exit.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--frame", type=int, default=None, help="Visualize one fixed frame.")
    parser.add_argument("--no-filter", action="store_true", help="Disable dex-retargeting low-pass filter.")
    parser.add_argument("--no-mesh", action="store_true", help="Hide BrainCo URDF meshes.")
    parser.add_argument("--show-fk", action="store_true", help="Also show compact BrainCo FK skeleton.")
    parser.add_argument("--show-frames", action="store_true", help="Show raw EgoDex hand and BrainCo root frames.")
    parser.add_argument(
        "--wrist-axis-preset",
        choices=("egodex-to-brainco", "identity"),
        default="egodex-to-brainco",
        help="Fixed local wrist-axis conversion from EgoDex hand pose to BrainCo root pose.",
    )
    parser.add_argument(
        "--scene-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, -90.0, -90.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Fixed display rotation applied to both EgoDex and BrainCo, in degrees.",
    )
    parser.add_argument(
        "--scene-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, -0.3),
        metavar=("X", "Y", "Z"),
        help="Fixed display translation applied after scene rotation.",
    )
    parser.add_argument("--egodex-point-size", type=float, default=0.008, help="EgoDex point size.")
    parser.add_argument("--mesh-alpha", type=float, default=0.55, help="BrainCo mesh alpha, 0-1.")
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


def rpy_deg_to_matrix(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=np.float32))
    return (rot_x(float(roll)) @ rot_y(float(pitch)) @ rot_z(float(yaw))).astype(np.float32)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)
    if not args.config.exists():
        raise FileNotFoundError(args.config)

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

    scene_rot = rpy_deg_to_matrix(tuple(args.scene_rpy_deg))
    scene_offset = np.asarray(args.scene_offset, dtype=np.float32)

    def to_scene_position(position: np.ndarray) -> np.ndarray:
        position = np.asarray(position, dtype=np.float32)
        if position.ndim == 1:
            return (scene_rot @ position + scene_offset).astype(np.float32)
        if position.ndim == 2 and position.shape[1] == 3:
            return (position @ scene_rot.T + scene_offset[None, :]).astype(np.float32)
        raise ValueError(f"Expected position shape (3,) or (N, 3), got {position.shape}")

    def to_scene_rotation(rotation: np.ndarray) -> np.ndarray:
        return (scene_rot @ np.asarray(rotation, dtype=np.float32)).astype(np.float32)

    print(f"[INFO] Loaded HDF5: {args.hdf5}", flush=True)
    print(f"[INFO] Retarget config: {args.config}", flush=True)
    print(f"[INFO] Number of frames: {num_frames}", flush=True)
    print(f"[INFO] Open browser: http://localhost:{args.port}", flush=True)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_frame("/world", axes_length=0.18, axes_radius=0.004)
    server.scene.add_grid("/world/grid", width=1.0, height=1.0)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = np.array([0.0, -1.15, 0.35])
        client.camera.look_at = np.array([0.0, 0.0, 0.0])
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
        gui_min_conf = server.gui.add_slider("Min confidence", min=0.0, max=1.0, step=0.01, initial_value=args.min_conf)

    with server.gui.add_folder("Layers"):
        gui_show_egodex = server.gui.add_checkbox("EgoDex VP25 hands", initial_value=True)
        gui_show_mesh = server.gui.add_checkbox("BrainCo mesh", initial_value=not args.no_mesh)
        gui_show_fk = server.gui.add_checkbox("BrainCo FK", initial_value=args.show_fk)
        gui_show_frames = server.gui.add_checkbox("Frames", initial_value=args.show_frames)

    q_gain = {}
    q_bias = {}
    with server.gui.add_folder("Visual qpos tune"):
        for joint_name in BRAINCO_HARDWARE_JOINT_ORDER:
            q_gain[joint_name] = server.gui.add_slider(
                f"{joint_name} gain", min=0.0, max=2.0, step=0.01, initial_value=1.0
            )
            q_bias[joint_name] = server.gui.add_slider(
                f"{joint_name} bias", min=-0.8, max=0.8, step=0.01, initial_value=0.0
            )

    brainco_urdfs = {}
    if not args.no_mesh:
        alpha = float(np.clip(args.mesh_alpha, 0.0, 1.0))
        brainco_urdfs = load_brainco_urdfs(
            server,
            mesh_color_override=(0.72, 0.76, 0.82, alpha),
            warning_suffix="FK skeleton only",
        )

    handles = {}

    def tuned_qpos(qpos_hardware: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos_hardware, dtype=np.float32).copy()
        for idx, joint_name in enumerate(BRAINCO_HARDWARE_JOINT_ORDER):
            qpos[idx] = qpos[idx] * float(q_gain[joint_name].value) + float(q_bias[joint_name].value)
        return qpos

    def update_scene(frame_idx: int) -> None:
        for side in ("left", "right"):
            hand_name = f"{side}Hand"
            egodex_name = f"/world/egodex/{side}"
            brainco_name = f"/world/brainco/{side}/fk"
            side_keys = {
                f"{egodex_name}_points",
                f"{egodex_name}_lines",
                f"{brainco_name}_points",
                f"{brainco_name}_lines",
                f"{side}_egodex_frame",
                f"{side}_brainco_frame",
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

            qpos_display = tuned_qpos(frame.result.qpos_hardware)
            qpos_full_display = frame.result.qpos_full.copy()
            qpos_full_display[retargeter.hardware_order[side]] = qpos_display
            fk_local = brainco_fk_points(retargeter.retargeting[side], qpos_full_display, side)

            base_pos = frame.wrist_position
            base_rot = frame.brainco_base_rotation
            scene_base_pos = to_scene_position(base_pos)
            scene_brainco_rot = to_scene_rotation(base_rot)

            if gui_show_egodex.value:
                draw_points_and_lines(
                    server,
                    handles,
                    egodex_name,
                    to_scene_position(frame.vp25_world),
                    EGODEX_COLORS[side],
                    make_vp25_line_segments,
                    point_size=float(args.egodex_point_size),
                )
            else:
                remove_handles(handles, [f"{egodex_name}_points", f"{egodex_name}_lines"])

            fk_world = ((fk_local - fk_local[0:1]) @ base_rot.T) + base_pos
            if gui_show_fk.value:
                draw_points_and_lines(
                    server,
                    handles,
                    brainco_name,
                    to_scene_position(fk_world),
                    BRAINCO_COLORS[side],
                    make_brainco_line_segments,
                    point_size=0.010,
                )
            else:
                remove_handles(handles, [f"{brainco_name}_points", f"{brainco_name}_lines"])

            if side in brainco_urdfs:
                urdf = brainco_urdfs[side]
                update_urdf_pose(urdf, qpos_display, scene_base_pos, scene_brainco_rot)
                urdf.show_visual = bool(gui_show_mesh.value)

            remove_handles(handles, [f"{side}_egodex_frame", f"{side}_brainco_frame"])
            if gui_show_frames.value:
                handles[f"{side}_egodex_frame"] = server.scene.add_frame(
                    f"/world/egodex/{side}/hand_pose",
                    position=scene_base_pos,
                    wxyz=rotation_matrix_to_wxyz(to_scene_rotation(frame.wrist_rotation)),
                    axes_length=0.07,
                    axes_radius=0.003,
                )
                handles[f"{side}_brainco_frame"] = server.scene.add_frame(
                    f"/world/brainco/{side}/root_pose",
                    position=scene_base_pos,
                    wxyz=rotation_matrix_to_wxyz(scene_brainco_rot),
                    axes_length=0.07,
                    axes_radius=0.003,
                )

        if not gui_show_mesh.value:
            set_urdf_visible(brainco_urdfs, False)

    def redraw(_event=None) -> None:
        update_scene(int(gui_frame.value))

    for control in (gui_frame, gui_min_conf, gui_show_egodex, gui_show_mesh, gui_show_fk, gui_show_frames):
        control.on_update(redraw)
    for joint_name in BRAINCO_HARDWARE_JOINT_ORDER:
        q_gain[joint_name].on_update(redraw)
        q_bias[joint_name].on_update(redraw)

    update_scene(int(gui_frame.value))
    run_playback_loop(gui_play, gui_frame, gui_fps, num_frames, args.loop, update_scene)


if __name__ == "__main__":
    main()
