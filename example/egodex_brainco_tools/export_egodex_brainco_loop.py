#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export EgoDex -> BrainCo retargeted loop data.

The JSON layout follows the recomputed end-effector/finger fields consumed by
convert_unitree_to_lerobot.py:

  state_ee/action_ee: left/right xyzrpy
  left_fig6d/right_fig6d: BrainCo 6D finger state
  left_fig6d_cmd/right_fig6d_cmd: BrainCo 6D finger command
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
for path in (THIS_DIR, REPO_ROOT / "egodex_wuji_tools"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from egodex_brainco_common import (  # noqa: E402
    BRAINCO_HARDWARE_JOINT_ORDER,
    BRAINCO_WRIST_OFFSET_LOCAL,
    BraincoRetargeter,
    brainco_base_to_wrist_pose,
    brainco_hand_only_scene_position,
    brainco_hand_only_scene_rotation,
    pose_xyzrpy,
    retarget_egodex_brainco_frame,
)
from egodex_wuji_common import require_h5py  # noqa: E402
from egodex_wuji_common import (  # noqa: E402
    get_transform_group,
)


def parse_args() -> argparse.Namespace:
    default_config = THIS_DIR / "config" / "brainco_vector.yml"
    parser = argparse.ArgumentParser(
        description="Export looped EgoDex -> BrainCo retargeted EE pose and 6D hand data."
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="Input EgoDex HDF5 file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--config", type=Path, default=default_config, help="BrainCo retargeting YAML.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS metadata.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum source frames before looping.")
    parser.add_argument("--loops", type=int, default=1, help="Repeat the retargeted sequence this many times.")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum hand confidence.")
    parser.add_argument(
        "--wrist-y-offset",
        type=float,
        default=float(BRAINCO_WRIST_OFFSET_LOCAL[1]),
        help="Offset from BrainCo palm/base frame to stored wrist frame along local +Y, in meters.",
    )
    parser.add_argument("--no-filter", action="store_true", help="Disable retargeting low-pass filter.")
    parser.add_argument(
        "--hand-action",
        choices=("normalized", "radians"),
        default="normalized",
        help="Store fig6d values as normalized BrainCo 0-1 commands or raw radians.",
    )
    parser.add_argument(
        "--wrist-axis-preset",
        choices=("egodex-to-brainco", "identity"),
        default="egodex-to-brainco",
        help="Fixed local wrist-axis conversion from EgoDex hand pose to BrainCo root pose.",
    )
    parser.add_argument(
        "--auto-scene-y-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only adjust the hand-only scene Y offset from the first valid stored wrist "
            "poses, equivalent to changing BRAINCO_HAND_ONLY_SCENE_OFFSET[1]."
        ),
    )
    parser.add_argument(
        "--scene-y-offset",
        type=float,
        default=None,
        help="Manual additional Y offset applied after the hand-only scene transform.",
    )
    return parser.parse_args()


def compute_initial_wrist_y_offset(base_frames: list[dict]) -> float:
    """Return additional scene Y offset from the first valid left/right wrist poses."""
    for frame in base_frames:
        valid = frame.get("valid", {})
        if not (valid.get("left") and valid.get("right")):
            continue
        left = np.asarray(frame["state_ee"]["left"][:3], dtype=np.float32)
        right = np.asarray(frame["state_ee"]["right"][:3], dtype=np.float32)
        return float(-0.5 * (left[1] + right[1]))
    return 0.0


def apply_pose_translation(base_frames: list[dict], translation: np.ndarray) -> None:
    for frame in base_frames:
        for key in ("state_ee", "action_ee", "brainco_base", "state_ee_torso", "action_ee_torso"):
            if key not in frame:
                continue
            for side in ("left", "right"):
                pose = np.asarray(frame[key][side], dtype=np.float32).copy()
                pose[:3] += translation
                frame[key][side] = pose


def to_json_compatible(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_compatible(v) for v in obj]
    return obj


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.loops < 1:
        raise SystemExit("--loops must be >= 1")
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)
    if not args.config.exists():
        raise FileNotFoundError(args.config)

    h5py = require_h5py()
    retargeter = BraincoRetargeter(args.config)

    frames = []
    with h5py.File(args.hdf5, "r") as h5_file:
        transform_group = get_transform_group(h5_file)
        total = int(h5_file["transforms"]["camera"].shape[0])
        end = total if args.max_frames is None else min(total, args.max_frames)
        source_frame_indices = np.asarray(list(range(0, end, args.stride)), dtype=np.int32)

        last_pose = {
            "left": np.zeros((6,), dtype=np.float32),
            "right": np.zeros((6,), dtype=np.float32),
        }
        last_base_pose = {
            "left": np.zeros((6,), dtype=np.float32),
            "right": np.zeros((6,), dtype=np.float32),
        }
        last_fig = {
            "left": np.zeros((6,), dtype=np.float32),
            "right": np.zeros((6,), dtype=np.float32),
        }
        last_fig_rad = {
            "left": np.zeros((6,), dtype=np.float32),
            "right": np.zeros((6,), dtype=np.float32),
        }
        last_valid = {"left": False, "right": False}

        base_frames = []
        for source_idx in source_frame_indices:
            source_idx_int = int(source_idx)
            frame_valid = {}
            pose = {}
            base_pose = {}
            fig = {}
            fig_rad = {}

            for side in ("left", "right"):
                frame = retarget_egodex_brainco_frame(
                    h5_file,
                    transform_group,
                    source_idx_int,
                    side,
                    retargeter,
                    min_conf=args.min_conf,
                    axis_preset=args.wrist_axis_preset,
                    apply_filter=not args.no_filter,
                )
                if frame is not None:
                    scene_base_pos = brainco_hand_only_scene_position(frame.wrist_position)
                    scene_base_rot = brainco_hand_only_scene_rotation(frame.brainco_base_rotation)
                    stored_wrist_pos, stored_wrist_rot = brainco_base_to_wrist_pose(
                        scene_base_pos,
                        scene_base_rot,
                        side,
                        args.wrist_y_offset,
                    )
                    last_base_pose[side] = pose_xyzrpy(scene_base_pos, scene_base_rot)
                    last_pose[side] = pose_xyzrpy(stored_wrist_pos, stored_wrist_rot)
                    last_fig_rad[side] = frame.result.qpos_hardware
                    last_fig[side] = (
                        frame.result.action_01
                        if args.hand_action == "normalized"
                        else frame.result.qpos_hardware
                    )
                    last_valid[side] = True

                pose[side] = last_pose[side].copy()
                base_pose[side] = last_base_pose[side].copy()
                fig[side] = last_fig[side].copy()
                fig_rad[side] = last_fig_rad[side].copy()
                frame_valid[side] = bool(last_valid[side])

            base_frames.append(
                {
                    "source_idx": source_idx_int,
                    "state_ee": {
                        "left": pose["left"],
                        "right": pose["right"],
                    },
                    "action_ee": {
                        "left": pose["left"],
                        "right": pose["right"],
                    },
                    "brainco_base": {
                        "left": base_pose["left"],
                        "right": base_pose["right"],
                    },
                    "state_torso": np.zeros((6,), dtype=np.float32),
                    "action_torso": np.zeros((6,), dtype=np.float32),
                    "state_d435": np.zeros((6,), dtype=np.float32),
                    "action_d435": np.zeros((6,), dtype=np.float32),
                    "state_ee_torso": {
                        "left": pose["left"],
                        "right": pose["right"],
                    },
                    "action_ee_torso": {
                        "left": pose["left"],
                        "right": pose["right"],
                    },
                    "state_base_pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                    "action_base_pose": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                    "waist": {
                        "state": np.zeros((3,), dtype=np.float32),
                        "action_used": np.zeros((3,), dtype=np.float32),
                    },
                    "velocity": {
                        "vx": 0.0,
                        "vy": 0.0,
                        "vyaw": 0.0,
                    },
                    "left_fig6d": fig["left"],
                    "right_fig6d": fig["right"],
                    "left_fig6d_cmd": fig["left"],
                    "right_fig6d_cmd": fig["right"],
                    "left_fig6d_rad": fig_rad["left"],
                    "right_fig6d_rad": fig_rad["right"],
                    "valid": frame_valid,
                }
            )

        additional_scene_offset = np.zeros((3,), dtype=np.float32)
        if args.scene_y_offset is not None:
            additional_scene_offset[1] = float(args.scene_y_offset)
        elif args.auto_scene_y_offset:
            additional_scene_offset[1] = compute_initial_wrist_y_offset(base_frames)
        if np.any(np.abs(additional_scene_offset) > 0.0):
            apply_pose_translation(base_frames, additional_scene_offset)

    out_idx = 0
    for loop_idx in range(args.loops):
        for base in base_frames:
            frame = dict(base)
            frame["idx"] = out_idx
            frame["loop_idx"] = loop_idx
            frames.append(frame)
            out_idx += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_data = {
        "info": {
            "format": "egodex_brainco_retarget_loop",
            "source_hdf5": str(args.hdf5),
            "config": str(args.config),
            "fps": float(args.fps),
            "loops": int(args.loops),
            "stride": int(args.stride),
            "hand_action": args.hand_action,
            "joint_order": list(BRAINCO_HARDWARE_JOINT_ORDER),
            "ee_pose_format": "xyzrpy",
            "ee_pose_frame": "brainco_wrist_in_hand_only_scene",
            "base_pose_frame": "brainco_base_in_hand_only_scene",
            "wrist_y_offset": float(args.wrist_y_offset),
            "scene_definition": "viser_brainco_hand_only_viewer",
            "auto_scene_y_offset": bool(args.auto_scene_y_offset),
            "additional_scene_offset": additional_scene_offset,
            "fig6d_dim_names": ["thumb_oc", "thumb_lat", "index", "middle", "ring", "little"],
        },
        "frames": frames,
    }
    json_path = args.output_dir / "recomputed_ee_fullbody.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(to_json_compatible(json_data), f, ensure_ascii=False, indent=2)

    npz_path = args.output_dir / "brainco_retarget_loop.npz"
    np.savez_compressed(
        npz_path,
        frame_indices=np.asarray([f["idx"] for f in frames], dtype=np.int32),
        source_frame_indices=np.asarray([f["source_idx"] for f in frames], dtype=np.int32),
        loop_indices=np.asarray([f["loop_idx"] for f in frames], dtype=np.int32),
        left_ee_pose_gripper_base=np.stack([f["state_ee"]["left"] for f in frames]).astype(np.float32),
        right_ee_pose_gripper_base=np.stack([f["state_ee"]["right"] for f in frames]).astype(np.float32),
        left_fig6d=np.stack([f["left_fig6d"] for f in frames]).astype(np.float32),
        right_fig6d=np.stack([f["right_fig6d"] for f in frames]).astype(np.float32),
        left_fig6d_cmd=np.stack([f["left_fig6d_cmd"] for f in frames]).astype(np.float32),
        right_fig6d_cmd=np.stack([f["right_fig6d_cmd"] for f in frames]).astype(np.float32),
        left_fig6d_rad=np.stack([f["left_fig6d_rad"] for f in frames]).astype(np.float32),
        right_fig6d_rad=np.stack([f["right_fig6d_rad"] for f in frames]).astype(np.float32),
        valid_left=np.asarray([f["valid"]["left"] for f in frames], dtype=bool),
        valid_right=np.asarray([f["valid"]["right"] for f in frames], dtype=bool),
        fps=np.asarray(args.fps, dtype=np.float32),
        joint_order=np.asarray(BRAINCO_HARDWARE_JOINT_ORDER),
    )

    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {npz_path}")
    print(f"[INFO] Source frames: {len(base_frames)}")
    print(f"[INFO] Looped frames: {len(frames)}")


if __name__ == "__main__":
    main()
