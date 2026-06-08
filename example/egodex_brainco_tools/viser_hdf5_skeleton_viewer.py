#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize EgoDex HDF5 human skeleton with viser.

Usage:
    python egodex_wuji_tools/viser_hdf5_skeleton_viewer.py \
        --hdf5 egodex_example/clean_cups/0.hdf5 \
        --fps 30

Then open:
    http://localhost:8080
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import viser

from egodex_wuji_common import (
    SKELETON_EDGES,
    collect_joint_positions,
    egodex_frame_to_mediapipe21,
    extract_position,
    extract_rotation_matrix,
    get_confidence_group,
    get_transform_group,
    hand_confidence_ok,
    infer_num_frames,
    make_line_segments,
    make_mediapipe_line_segments,
    normalize_scene_scale,
    offset_joint_positions,
    print_hdf5_tree,
    robot_fk_mediapipe21,
)


WUJIHAND_COLORS = {
    "left": np.array([255, 110, 180], dtype=np.uint8),
    "right": np.array([120, 240, 180], dtype=np.uint8),
}


def rotation_x(degrees):
    theta = np.deg2rad(degrees)
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


def rotation_y(degrees):
    theta = np.deg2rad(degrees)
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


def rotation_z(degrees):
    theta = np.deg2rad(degrees)
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


# Columns are Wuji wrist axes expressed in EgoDex wrist coordinates.
#
# User-provided conventions:
#   EgoDex left:  +x four-finger tips, +y palm side, +z pinky side
#   Wuji left:    +x palm side, +y pinky side, +z four-finger tips
#   EgoDex right: +z thumb side
#   Wuji right:   +x palm side, +y thumb side, +z four-finger tips
#
# For a world-from-EgoDex wrist rotation R_we, the matching Wuji wrist rotation is:
#     R_ww = R_we @ EGODEX_FROM_WUJI_WRIST[side]
EGODEX_FROM_WUJI_WRIST = {
    "left": np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
    "right": np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    ),
}


def import_retargeter(wuji_root):
    wuji_root = Path(wuji_root).resolve()
    if str(wuji_root) not in sys.path:
        sys.path.insert(0, str(wuji_root))

    try:
        from wuji_retargeting import Retargeter
    except ImportError as exc:
        raise SystemExit(
            "Cannot import wuji_retargeting. Pass --wuji-root /path/to/wuji-retargeting "
            "or install it with `pip install -e /path/to/wuji-retargeting`."
        ) from exc
    return Retargeter


def layout_offsets(layout):
    if layout == "top-bottom":
        return np.array([0.0, 0.0, 0.8], dtype=np.float32), np.array([0.0, 0.0, -0.8], dtype=np.float32)
    return np.array([-0.6, 0.0, 0.0], dtype=np.float32), np.array([0.6, 0.0, 0.0], dtype=np.float32)


def normalize_vector(vector, fallback):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def estimate_wuji_wrist_rotation_from_keypoints(raw_21, egodex_wrist_rot=None):
    """Estimate world-from-Wuji-wrist rotation from EgoDex hand geometry.

    MediaPipe/EgoDex 21-point indices:
      0 wrist, 5 index MCP, 9 middle MCP, 13 ring MCP, 17 pinky MCP,
      8/12/16/20 fingertips.

    Wuji convention:
      +x palm side, +y away from thumb side, +z four-finger fingertips.
    """
    wrist = raw_21[0]
    index_mcp = raw_21[5]
    pinky_mcp = raw_21[17]
    mcp_center = np.mean(raw_21[[5, 9, 13, 17]], axis=0)
    tip_center = np.mean(raw_21[[8, 12, 16, 20]], axis=0)

    y_axis = normalize_vector(pinky_mcp - index_mcp, np.array([0.0, 1.0, 0.0]))
    z_hint = tip_center - mcp_center
    z_axis = z_hint - np.dot(z_hint, y_axis) * y_axis
    z_axis = normalize_vector(z_axis, normalize_vector(tip_center - wrist, np.array([0.0, 0.0, 1.0])))
    x_axis = normalize_vector(np.cross(y_axis, z_axis), np.array([1.0, 0.0, 0.0]))

    if egodex_wrist_rot is not None:
        egodex_palm_side = -egodex_wrist_rot[:, 1]
        if np.dot(x_axis, egodex_palm_side) < 0:
            y_axis = -y_axis
            x_axis = -x_axis

    y_axis = normalize_vector(np.cross(z_axis, x_axis), y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)


def egodex_to_wuji_wrist_rotation(egodex_rot, side, axis_preset):
    if axis_preset == "identity":
        return egodex_rot
    return egodex_rot @ EGODEX_FROM_WUJI_WRIST[side]


def remove_handles(handles, keys):
    for key in keys:
        handle = handles.pop(key, None)
        if handle is not None:
            handle.remove()


def remove_wuji_side_handles(wuji_handles, side, include_pose=True, include_fk=True):
    keys = []
    if include_fk:
        keys.extend([f"{side}_points", f"{side}_bones"])
    if include_pose:
        keys.extend([f"{side}_wrist_frame", f"{side}_label"])
    remove_handles(wuji_handles, keys)


def set_wuji_urdf_visible(wuji_urdfs, visible):
    for wuji_urdf in wuji_urdfs.values():
        wuji_urdf.show_visual = bool(visible)


# -----------------------------
# Main visualization
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, required=True, help="Path to EgoDex hdf5 file.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum confidence threshold.")
    parser.add_argument("--print-tree", action="store_true", help="Print HDF5 tree and exit.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument("--frame", type=int, default=None, help="Visualize one fixed frame.")
    parser.add_argument(
        "--wuji-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "wuji-retargeting"),
        help="Path to wuji-retargeting repository.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Retarget YAML config. Default: wuji-retargeting/example/config/adaptive_analytical_avp.yaml",
    )
    parser.add_argument(
        "--layout",
        choices=["left-right", "top-bottom"],
        default="left-right",
        help="Place EgoDex and Wuji systems side by side or vertically.",
    )
    parser.add_argument("--disable-wuji", action="store_true", help="Only show EgoDex skeleton.")
    parser.add_argument(
        "--no-wuji-mesh",
        action="store_true",
        help="Do not load Wuji URDF meshes; show only FK skeletons.",
    )
    parser.add_argument(
        "--wrist-axis-preset",
        choices=["egodex-to-wuji", "identity"],
        default="egodex-to-wuji",
        help="Fixed wrist-axis conversion before applying EgoDex hand pose to Wuji.",
    )
    parser.add_argument(
        "--wrist-pose-source",
        choices=["geometry", "hdf5"],
        default="hdf5",
        help="Use hand keypoint geometry or HDF5 Hand transform rotation for Wuji wrist pose.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_path}")

    h5_file = h5py.File(hdf5_path, "r")

    if args.print_tree:
        print_hdf5_tree(h5_file)
        return

    transform_group = get_transform_group(h5_file)
    conf_group = get_confidence_group(h5_file)
    num_frames = infer_num_frames(transform_group)
    egodex_offset, wuji_offset = layout_offsets(args.layout)

    retargeters = {}
    if not args.disable_wuji:
        Retargeter = import_retargeter(args.wuji_root)
        config_path = Path(args.config) if args.config is not None else (
            Path(args.wuji_root) / "example" / "config" / "adaptive_analytical_avp.yaml"
        )
        config_path = config_path.resolve()
        print(f"[INFO] Retarget config: {config_path}", flush=True)
        retargeters = {
            "left": Retargeter.from_yaml(str(config_path), hand_side="left"),
            "right": Retargeter.from_yaml(str(config_path), hand_side="right"),
        }

    print(f"[INFO] Loaded: {hdf5_path}", flush=True)
    print(f"[INFO] Number of frames: {num_frames}", flush=True)
    print(f"[INFO] EgoDex offset: {egodex_offset}", flush=True)
    print(f"[INFO] Wuji offset: {wuji_offset}", flush=True)
    print(f"[INFO] Wrist pose source: {args.wrist_pose_source}", flush=True)
    print(f"[INFO] Wrist axis preset: {args.wrist_axis_preset}", flush=True)
    print(f"[INFO] Open browser: http://localhost:{args.port}", flush=True)

    server = viser.ViserServer(host=args.host, port=args.port)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.position = np.array([0.0, 2.8, -1.8])
        client.camera.look_at = np.array([0.0, 0.0, 2])
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    # Add world frame.
    server.scene.add_frame(
        "/world",
        axes_length=0.2,
        axes_radius=0.005,
    )
    server.scene.add_frame(
        "/world/egodex_skeleton_origin",
        position=egodex_offset,
        axes_length=0.12,
        axes_radius=0.003,
    )
    server.scene.add_frame(
        "/world/wuji_retargeted_origin",
        position=wuji_offset,
        axes_length=0.12,
        axes_radius=0.003,
    )

    # GUI controls.
    with server.gui.add_folder("Playback"):
        gui_play = server.gui.add_checkbox("Play", initial_value=args.frame is None)
        gui_frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0 if args.frame is None else max(0, min(args.frame, num_frames - 1)),
        )
        gui_fps = server.gui.add_slider(
            "FPS",
            min=1,
            max=120,
            step=1,
            initial_value=int(args.fps),
        )
        gui_min_conf = server.gui.add_slider(
            "Min confidence",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=args.min_conf,
        )
        gui_show_egodex = server.gui.add_checkbox("Show EgoDex skeleton", initial_value=True)
        gui_show_wuji = server.gui.add_checkbox("Show Wuji retargeted hands", initial_value=not args.disable_wuji)
        gui_show_wuji_mesh = server.gui.add_checkbox("Show Wuji mesh", initial_value=not args.no_wuji_mesh)
        gui_show_wuji_fk = server.gui.add_checkbox("Show Wuji FK skeleton", initial_value=True)
        gui_show_wrist_frames = server.gui.add_checkbox("Show wrist frames", initial_value=True)
        gui_show_labels = server.gui.add_checkbox("Show labels", initial_value=False)

    wuji_urdfs = {}
    if retargeters and not args.no_wuji_mesh:
        try:
            from viser.extras import ViserUrdf

            desc_root = Path(args.wuji_root) / "wuji_retargeting" / "wuji_hand_description"
            for side in ["left", "right"]:
                urdf_path = desc_root / "urdf" / f"{side}.urdf"
                wuji_urdfs[side] = ViserUrdf(
                    server,
                    urdf_path,
                    root_node_name=f"/world/wuji_retargeted/{side}_mesh",
                    mesh_color_override=(0.75, 0.78, 0.82, 0.38),
                    load_meshes=True,
                    load_collision_meshes=False,
                )
        except Exception as exc:
            print(f"[WARN] Failed to load Wuji URDF mesh; using FK skeleton only: {exc}", flush=True)
            wuji_urdfs = {}

    # Handles.
    point_handle = None
    line_handle = None
    palm_frame_handles = {}
    label_handles = {}
    wuji_handles = {}

    def update_scene(frame_idx):
        nonlocal point_handle, line_handle, palm_frame_handles, label_handles, wuji_handles

        joint_positions, joint_confidences = collect_joint_positions(
            transform_group=transform_group,
            conf_group=conf_group,
            frame_idx=frame_idx,
            min_conf=float(gui_min_conf.value),
        )

        if len(joint_positions) == 0:
            return

        egodex_joint_positions = offset_joint_positions(joint_positions, egodex_offset)

        pts = np.asarray(list(egodex_joint_positions.values()), dtype=np.float32)
        names = list(joint_positions.keys())

        radius = normalize_scene_scale(joint_positions)

        if point_handle is not None:
            point_handle.remove()
            point_handle = None

        if line_handle is not None:
            line_handle.remove()
            line_handle = None

        if gui_show_egodex.value:
            point_handle = server.scene.add_point_cloud(
                name="/world/egodex_skeleton/joints",
                points=pts,
                colors=np.tile(np.array([[255, 180, 60]], dtype=np.uint8), (pts.shape[0], 1)),
                point_size=0.008  #max(radius, 0.005),
            )

            segments = make_line_segments(egodex_joint_positions, SKELETON_EDGES)
            if segments is not None:
                line_handle = server.scene.add_line_segments(
                    name="/world/egodex_skeleton/bones",
                    points=segments,
                    colors=np.array([80, 160, 255], dtype=np.uint8),
                    line_width=2.0,
                )

        # Show palm frames if available.
        if not gui_show_wrist_frames.value:
            for handle in palm_frame_handles.values():
                handle.remove()
            palm_frame_handles = {}

        for hand_name in ["leftHand", "rightHand", "camera"]:
            if hand_name not in transform_group:
                continue
            if not gui_show_wrist_frames.value:
                continue

            try:
                pos = extract_position(transform_group[hand_name], frame_idx) + egodex_offset
                rot = extract_rotation_matrix(transform_group[hand_name], frame_idx)

                if rot is None:
                    continue

                frame_path = f"/world/egodex_skeleton/frames/{hand_name}"

                if frame_path in palm_frame_handles:
                    palm_frame_handles[frame_path].remove()

                palm_frame_handles[frame_path] = server.scene.add_frame(
                    frame_path,
                    wxyz=rotation_matrix_to_wxyz(rot),
                    position=pos,
                    axes_length=0.08 if "Hand" in hand_name else 0.15,
                    axes_radius=0.004,
                )
            except Exception:
                continue

        # Optional labels for important nodes.
        for handle in label_handles.values():
            handle.remove()
        label_handles = {}

        if gui_show_egodex.value and gui_show_labels.value:
            for important_name in ["leftHand", "rightHand", "camera", "hip"]:
                if important_name not in egodex_joint_positions:
                    continue

                label_path = f"/world/egodex_skeleton/labels/{important_name}"
                label_handles[label_path] = server.scene.add_label(
                    label_path,
                    text=important_name,
                    position=egodex_joint_positions[important_name],
                )

        if not gui_show_wuji.value:
            for key, handle in list(wuji_handles.items()):
                handle.remove()
            wuji_handles = {}
            set_wuji_urdf_visible(wuji_urdfs, False)
            return

        if gui_show_wuji.value and retargeters:
            for side in ["left", "right"]:
                hand_name = f"{side}Hand"
                if hand_name not in transform_group:
                    continue
                if not hand_confidence_ok(h5_file, frame_idx, side, float(gui_min_conf.value)):
                    continue

                try:
                    wrist_pos = extract_position(transform_group[hand_name], frame_idx)
                    egodex_wrist_rot = extract_rotation_matrix(transform_group[hand_name], frame_idx)
                    if egodex_wrist_rot is None:
                        continue
                    raw_21 = egodex_frame_to_mediapipe21(h5_file, frame_idx, side)
                    if args.wrist_pose_source == "geometry":
                        wuji_wrist_rot = estimate_wuji_wrist_rotation_from_keypoints(
                            raw_21,
                            egodex_wrist_rot,
                        )
                    else:
                        wuji_wrist_rot = egodex_to_wuji_wrist_rotation(
                            egodex_wrist_rot,
                            side,
                            args.wrist_axis_preset,
                        )
                    qpos, _verbose = retargeters[side].retarget_verbose(raw_21, apply_filter=False)
                    fk_local = robot_fk_mediapipe21(retargeters[side], qpos)
                    fk_wrist = fk_local - fk_local[0:1]
                    fk_world = (fk_wrist @ wuji_wrist_rot.T) + wrist_pos + wuji_offset
                    color = WUJIHAND_COLORS[side]
                    hand_root = f"/world/wuji_retargeted/{side}"
                    wrist_world = wrist_pos + wuji_offset

                    if side in wuji_urdfs:
                        wuji_urdf = wuji_urdfs[side]
                        wuji_urdf.update_cfg(qpos)
                        for root_handle_name in ["_visual_root_frame", "_collision_root_frame"]:
                            root_handle = getattr(wuji_urdf, root_handle_name, None)
                            if root_handle is not None:
                                root_handle.position = wrist_world
                                root_handle.wxyz = rotation_matrix_to_wxyz(wuji_wrist_rot)
                        wuji_urdf.show_visual = bool(gui_show_wuji_mesh.value)

                    if not gui_show_wuji_fk.value:
                        remove_wuji_side_handles(wuji_handles, side, include_pose=False, include_fk=True)

                    if gui_show_wuji_fk.value:
                        remove_wuji_side_handles(wuji_handles, side, include_pose=False, include_fk=True)
                        wuji_handles[f"{side}_points"] = server.scene.add_point_cloud(
                            name=f"{hand_root}/joints",
                            points=fk_world.astype(np.float32),
                            colors=np.tile(color[None, :], (fk_world.shape[0], 1)),
                            point_size=0.01,
                        )
                        wuji_handles[f"{side}_bones"] = server.scene.add_line_segments(
                            name=f"{hand_root}/bones",
                            points=make_mediapipe_line_segments(fk_world),
                            colors=color,
                            line_width=3.0,
                        )
                    remove_wuji_side_handles(wuji_handles, side, include_pose=True, include_fk=False)
                    wuji_handles[f"{side}_wrist_frame"] = server.scene.add_frame(
                        f"{hand_root}/wrist_pose",
                        wxyz=rotation_matrix_to_wxyz(wuji_wrist_rot),
                        position=wrist_world,
                        axes_length=0.10,
                        axes_radius=0.005,
                    )
                    if gui_show_labels.value:
                        wuji_handles[f"{side}_label"] = server.scene.add_label(
                            f"{hand_root}/label",
                            text=f"wuji {side}",
                            position=wrist_world + np.array([0.0, 0.0, 0.10]),
                        )
                except Exception as exc:
                    print(f"[WARN] Failed to retarget {side} hand at frame {frame_idx}: {exc}", flush=True)

    def rotation_matrix_to_wxyz(R):
        """
        Convert 3x3 rotation matrix to viser quaternion order: w, x, y, z.
        Avoid scipy dependency.
        """
        R = np.asarray(R, dtype=np.float64)
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        q = np.array([w, x, y, z], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        return q

    # First frame.
    update_scene(int(gui_frame.value))

    @gui_frame.on_update
    def _(_event):
        update_scene(int(gui_frame.value))

    @gui_min_conf.on_update
    def _(_event):
        update_scene(int(gui_frame.value))

    @gui_show_egodex.on_update
    def _(_event):
        update_scene(int(gui_frame.value))

    @gui_show_wuji.on_update
    def _(_event):
        update_scene(int(gui_frame.value))

    @gui_show_wuji_mesh.on_update
    def _(_event):
        update_scene(int(gui_frame.value))

    @gui_show_wuji_fk.on_update
    def _(_event):
        update_scene(int(gui_frame.value))

    @gui_show_wrist_frames.on_update
    def _(_event):
        update_scene(int(gui_frame.value))

    @gui_show_labels.on_update
    def _(_event):
        update_scene(int(gui_frame.value))

    # Main loop.
    last_time = time.time()

    while True:
        if gui_play.value:
            now = time.time()
            dt = now - last_time
            target_dt = 1.0 / max(float(gui_fps.value), 1.0)

            if dt >= target_dt:
                last_time = now

                next_frame = int(gui_frame.value) + 1
                if next_frame >= num_frames:
                    if args.loop:
                        next_frame = 0
                    else:
                        next_frame = num_frames - 1
                        gui_play.value = False

                gui_frame.value = next_frame
                update_scene(next_frame)

        time.sleep(0.001)


if __name__ == "__main__":
    main()
