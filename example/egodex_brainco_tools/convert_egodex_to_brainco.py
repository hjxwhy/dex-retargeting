#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert EgoDex HDF5 hand trajectories to BrainCo 6-motor targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
for path in (THIS_DIR, REPO_ROOT / "egodex_wuji_tools"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from egodex_brainco_common import (  # noqa: E402
    BRAINCO_HARDWARE_JOINT_ORDER,
    BraincoRetargeter,
    egodex_vp25_to_brainco_local,
)
from egodex_wuji_common import (  # noqa: E402
    egodex_vp25_positions,
    hand_confidence_ok,
    require_h5py,
)
from viser_hdf5_skeleton_viewer import (  # noqa: E402
    extract_position,
    extract_rotation_matrix,
    get_transform_group,
)


def parse_args() -> argparse.Namespace:
    default_config = THIS_DIR / "config" / "brainco_vector.yml"
    parser = argparse.ArgumentParser(
        description="Convert EgoDex HDF5 frames to BrainCo qpos/action arrays."
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="Input EgoDex .hdf5 file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--config", type=Path, default=default_config, help="BrainCo retargeting YAML.")
    parser.add_argument("--fps", type=float, default=30.0, help="Source FPS metadata.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum source frames.")
    parser.add_argument("--min-conf", type=float, default=0.0, help="Minimum hand confidence.")
    parser.add_argument(
        "--hand",
        choices=("left", "right", "both"),
        default="both",
        help="Which hand to convert.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable dex-retargeting low-pass filter during conversion.",
    )
    parser.add_argument(
        "--wrist-axis-preset",
        choices=("egodex-to-brainco", "identity"),
        default="egodex-to-brainco",
        help="Fixed local wrist-axis conversion from EgoDex hand pose to BrainCo root pose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)

    h5py = require_h5py()
    sides = ("left", "right") if args.hand == "both" else (args.hand,)
    retargeter = BraincoRetargeter(args.config)

    with h5py.File(args.hdf5, "r") as h5_file:
        transform_group = get_transform_group(h5_file)
        total = int(h5_file["transforms"]["camera"].shape[0])
        end = total if args.max_frames is None else min(total, args.max_frames)
        frame_indices = np.asarray(list(range(0, end, args.stride)), dtype=np.int32)

        qpos = {side: np.zeros((len(frame_indices), 6), dtype=np.float32) for side in ("left", "right")}
        action = {side: np.zeros((len(frame_indices), 6), dtype=np.float32) for side in ("left", "right")}
        valid = {side: np.zeros((len(frame_indices),), dtype=bool) for side in ("left", "right")}
        last_qpos = {side: np.zeros((6,), dtype=np.float32) for side in ("left", "right")}
        last_action = {side: np.zeros((6,), dtype=np.float32) for side in ("left", "right")}

        for out_idx, frame_idx in enumerate(frame_indices):
            for side in sides:
                ok = hand_confidence_ok(h5_file, int(frame_idx), side, args.min_conf)
                if ok:
                    hand_name = f"{side}Hand"
                    vp25 = egodex_vp25_positions(h5_file, int(frame_idx), side)
                    wrist_pos = extract_position(transform_group[hand_name], int(frame_idx))
                    wrist_rot = extract_rotation_matrix(transform_group[hand_name], int(frame_idx))
                    if wrist_pos is None or wrist_rot is None:
                        continue
                    vp25, _brainco_rot = egodex_vp25_to_brainco_local(
                        vp25,
                        wrist_pos,
                        wrist_rot,
                        side,
                        args.wrist_axis_preset,
                    )
                    result = retargeter.retarget(
                        side,
                        vp25,
                        apply_filter=not args.no_filter,
                    )
                    last_qpos[side] = result.qpos_hardware
                    last_action[side] = result.action_01
                    valid[side][out_idx] = True

                qpos[side][out_idx] = last_qpos[side]
                action[side][out_idx] = last_action[side]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        left_qpos_rad=qpos["left"],
        right_qpos_rad=qpos["right"],
        left_action_01=action["left"],
        right_action_01=action["right"],
        valid_left=valid["left"],
        valid_right=valid["right"],
        frame_indices=frame_indices,
        fps=np.asarray(args.fps, dtype=np.float32),
        source_hdf5=np.asarray(str(args.hdf5)),
        joint_order=np.asarray(BRAINCO_HARDWARE_JOINT_ORDER),
    )
    print(f"[INFO] Wrote {args.output}")
    print(f"[INFO] Frames: {len(frame_indices)}")
    print(f"[INFO] Valid left/right: {valid['left'].sum()} / {valid['right'].sum()}")


if __name__ == "__main__":
    main()
