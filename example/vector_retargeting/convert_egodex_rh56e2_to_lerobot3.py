"""
Convert EgoDex RH56E2 retargeting results into a LeRobot v3.0-style dataset.

This script processes ALL HDF5 files in a directory, producing a single dataset
with multiple episodes, video files synced to parquet, and bimanual hand data.

Directory layout:
    <output_dir>/
        videos/observation.images.head_cam/chunk-000/file-{ep:03d}.mp4
        images/observation.images.head_cam/   (empty, placeholder)
        data/chunk-000/file-000.parquet
        meta/info.json
        meta/stats.json
        meta/tasks.parquet
        meta/episodes/chunk-000/file-000.parquet

Example:
    python3 example/vector_retargeting/convert_egodex_rh56e2_to_lerobot3.py \
        --hdf5-dir /home/user/ml-egodex/test/clean_cups \
        --output-dir /home/user/ml-egodex/test/clean_cups/lerobot3_rh56e2 \
        --fps 30.0
"""

from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro
from scipy.spatial.transform import Rotation

from dex_retargeting.constants import HandType, OPERATOR2MANO
from visualize_egodex_retargeting_rh56e2 import (
    EGODEX_TIP_JOINTS,
    HAND_JOINTS,
    RH56E2_DEFAULT_TIP_ORIGIN_SCALE,
    RH56E2_JOINT_LABELS,
    _build_rh56e2_retargeting,
)

# ── Coordinate conventions ──────────────────────────────────────────────────
# EgoDex transforms are in OpenCV camera body frame: x-right, y-down, z-forward.
# We convert to FLU: x-forward, y-left, z-up.
OPENCV_TO_FLU = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def _eef_pose6d_in_first_image_flu(
    transforms_group,
    frame_idx: int,
    hand_type: HandType,
    world_t_image0_inv: np.ndarray,
) -> np.ndarray:
    """Return [x, y, z, roll, pitch, yaw] in first-image FLU frame."""
    wrist_key = HAND_JOINTS[hand_type][0][0]
    image0_t_eef = world_t_image0_inv @ transforms_group[wrist_key][frame_idx]

    position = OPENCV_TO_FLU @ image0_t_eef[:3, 3]
    R_flu = OPENCV_TO_FLU @ image0_t_eef[:3, :3] @ OPENCV_TO_FLU.T
    rpy = Rotation.from_matrix(R_flu).as_euler("xyz").astype(np.float32)

    return np.concatenate([position, rpy]).astype(np.float32)


def _hands_from_bimanual() -> List[HandType]:
    return [HandType.right, HandType.left]


def _hand_wrist_key(hand_type: HandType) -> str:
    return HAND_JOINTS[hand_type][0][0]


def _compute_ref_value(
    transforms_group,
    frame_idx: int,
    hand_type: HandType,
    target_link_human_indices,
) -> np.ndarray:
    tip_joint_map = EGODEX_TIP_JOINTS[hand_type]
    operator2mano = OPERATOR2MANO[hand_type]
    wrist_key = _hand_wrist_key(hand_type)

    wrist_transform = transforms_group[wrist_key][frame_idx]
    wrist_transform_inv = np.linalg.inv(wrist_transform)

    tip_pos = np.zeros((21, 3), dtype=np.float32)
    for joint_name, mano_idx in tip_joint_map.items():
        joint_transform = transforms_group[joint_name][frame_idx]
        tip_pos[mano_idx] = (
            (wrist_transform_inv @ joint_transform)[:3, 3] @ operator2mano
        )

    origin_idx = target_link_human_indices[0]
    task_idx = target_link_human_indices[1]
    return tip_pos[task_idx] - tip_pos[origin_idx]


def _legacy9_to_hand6(robot_qpos: np.ndarray) -> np.ndarray:
    HAND6_FROM_LEGACY9 = [7, 8, 6, 5, 4, 3]
    if robot_qpos.shape[0] < 9:
        raise ValueError(
            f"RH56E2 retargeting qpos should be 9D, got {robot_qpos.shape}"
        )
    return robot_qpos[HAND6_FROM_LEGACY9].astype(np.float32)


def _prefixed(labels: List[str], hand_type: HandType) -> List[str]:
    return [f"{hand_type.name}/{label}" for label in labels]


def _feature_stats(matrix: np.ndarray) -> dict:
    if matrix.size == 0:
        return {
            "min": [], "max": [], "mean": [], "std": [], "count": 0,
            "q01": [], "q10": [], "q50": [], "q90": [], "q99": [],
        }
    q = np.quantile(matrix, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": matrix.min(axis=0).tolist(),
        "max": matrix.max(axis=0).tolist(),
        "mean": matrix.mean(axis=0).tolist(),
        "std": matrix.std(axis=0).tolist(),
        "count": int(matrix.shape[0]),
        "q01": q[0].tolist(), "q10": q[1].tolist(),
        "q50": q[2].tolist(), "q90": q[3].tolist(), "q99": q[4].tolist(),
    }


def _shift_rows(rows: List[List[float]], shift: int) -> List[List[float]]:
    if shift <= 0 or not rows:
        return rows
    return [rows[min(i + shift, len(rows) - 1)] for i in range(len(rows))]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── EEF 6D labels (matching导师要求) ────────────────────────────────────────
EEF6D_LABELS = ["x", "y", "z", "roll", "pitch", "yaw"]


def main(
    hdf5_dir: str,
    output_dir: str,
    fps: float = 30.0,
    task: str = "egodex rh56e2 bimanual retargeting",
    action_shift: int = 1,
    scaling_factor: Optional[float] = None,
    robot_wrist_z_offset: float = -0.0,
    retarget_origin_link: Optional[str] = None,
    tip_origin_scale: float = RH56E2_DEFAULT_TIP_ORIGIN_SCALE,
    max_frames: Optional[int] = None,
    debug_urdf_path: Optional[str] = None,
):
    """Convert all EgoDex HDF5 episodes in a directory to a single LeRobot 3.0 dataset.

    action_shift:
        1 means action[t] = observation[t+1], last frame copies itself.
        0 means action[t] = observation[t].
    """
    hands = _hands_from_bimanual()
    hdf5_files = sorted(glob.glob(str(Path(hdf5_dir) / "*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {hdf5_dir}")
    print(f"Found {len(hdf5_files)} HDF5 episodes in {hdf5_dir}")

    out_root = Path(output_dir).absolute()
    data_dir = out_root / "data" / "chunk-000"
    meta_dir = out_root / "meta"
    episodes_meta_dir = meta_dir / "episodes" / "chunk-000"
    video_dir = out_root / "videos" / "observation.images.head_cam" / "chunk-000"
    images_dir = out_root / "images" / "observation.images.head_cam"

    for d in [data_dir, meta_dir, episodes_meta_dir, video_dir, images_dir]:
        d.mkdir(parents=True, exist_ok=True)

    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    retargetings = {
        hand: _build_rh56e2_retargeting(
            robot_dir=robot_dir,
            hand_type=hand,
            scaling_factor=scaling_factor,
            robot_wrist_z_offset=robot_wrist_z_offset,
            retarget_origin_link=retarget_origin_link,
            tip_origin_scale=tip_origin_scale,
            debug_urdf_path=debug_urdf_path,
        )
        for hand in hands
    }

    # ── Accumulators across all episodes ────────────────────────────────────
    all_global_indices: List[int] = []
    all_episode_indices: List[int] = []
    all_frame_indices: List[int] = []
    all_timestamps: List[float] = []
    all_task_indices: List[int] = []
    all_is_first: List[bool] = []
    all_is_last: List[bool] = []
    all_is_terminal: List[bool] = []

    # Per-hand state vectors (6D ee + 6D finger)
    all_state_right: List[List[float]] = []
    all_state_left: List[List[float]] = []
    all_action_right: List[List[float]] = []
    all_action_left: List[List[float]] = []

    all_frame_counts: List[int] = []

    global_offset = 0

    for ep_idx, hdf5_path in enumerate(
        tqdm.tqdm(hdf5_files, desc="Processing episodes")
    ):
        hdf5_path = Path(hdf5_path)
        mp4_src = hdf5_path.with_suffix(".mp4")
        mp4_dst = video_dir / f"file-{ep_idx:03d}.mp4"
        if mp4_src.exists() and not mp4_dst.exists():
            shutil.copy2(str(mp4_src), str(mp4_dst))

        with h5py.File(hdf5_path, "r") as h5_file:
            transforms = h5_file["transforms"]

            required_keys = [_hand_wrist_key(h) for h in hands]
            if "camera" not in transforms:
                print(f"  SKIP {hdf5_path.name}: no camera transform")
                continue
            missing = [k for k in required_keys if k not in transforms]
            if missing:
                print(f"  SKIP {hdf5_path.name}: missing wrist transforms {missing}")
                continue

            num_frames = min(
                transforms[key].shape[0] for key in ["camera"] + required_keys
            )
            if max_frames is not None:
                num_frames = min(num_frames, max_frames)

            world_t_image0_inv = np.linalg.inv(transforms["camera"][0])

            # ── Per-episode state storage (need full list for action shift) ─
            ep_state_right: List[List[float]] = []
            ep_state_left: List[List[float]] = []

            for frame_idx in range(num_frames):
                per_frame_right: List[float] = []
                per_frame_left: List[float] = []

                for hand in hands:
                    retargeting = retargetings[hand]
                    retarget_indices = retargeting.optimizer.target_link_human_indices
                    ref_value = _compute_ref_value(
                        transforms, frame_idx, hand, retarget_indices
                    )
                    legacy_qpos = retargeting.retarget(ref_value)
                    hand6 = _legacy9_to_hand6(legacy_qpos)
                    eef6d = _eef_pose6d_in_first_image_flu(
                        transforms, frame_idx, hand, world_t_image0_inv
                    )

                    if hand == HandType.right:
                        per_frame_right.extend(eef6d.tolist())
                        per_frame_right.extend(hand6.tolist())
                    else:
                        per_frame_left.extend(eef6d.tolist())
                        per_frame_left.extend(hand6.tolist())

                ep_state_right.append(per_frame_right)
                ep_state_left.append(per_frame_left)

            # ── Action shift: action[t] = state[t+1], last = self ──────────
            ep_action_right = _shift_rows(ep_state_right, action_shift)
            ep_action_left = _shift_rows(ep_state_left, action_shift)

            # ── Append to global accumulators ───────────────────────────────
            for frame_idx in range(num_frames):
                all_global_indices.append(global_offset + frame_idx)
                all_episode_indices.append(ep_idx)
                all_frame_indices.append(frame_idx)
                all_timestamps.append(frame_idx / fps)
                all_task_indices.append(0)
                all_is_first.append(frame_idx == 0)
                all_is_last.append(frame_idx == num_frames - 1)
                all_is_terminal.append(frame_idx == num_frames - 1)

            all_state_right.extend(ep_state_right)
            all_state_left.extend(ep_state_left)
            all_action_right.extend(ep_action_right)
            all_action_left.extend(ep_action_left)
            all_frame_counts.append(num_frames)
            global_offset += num_frames

    total_frames = global_offset
    print(f"\nTotal episodes: {len(hdf5_files)}, Total frames: {total_frames}")

    # ── Write parquet ───────────────────────────────────────────────────────
    table_columns = {
        "index": pa.array(all_global_indices, type=pa.int64()),
        "episode_index": pa.array(all_episode_indices, type=pa.int64()),
        "frame_index": pa.array(all_frame_indices, type=pa.int64()),
        "timestamp": pa.array(all_timestamps, type=pa.float32()),
        "task_index": pa.array(all_task_indices, type=pa.int64()),
        "is_first": pa.array(all_is_first, type=pa.bool_()),
        "is_last": pa.array(all_is_last, type=pa.bool_()),
        "is_terminal": pa.array(all_is_terminal, type=pa.bool_()),
        "observation.state.right_ee_pose_gripper_base": pa.array(
            [row[:6] for row in all_state_right], type=pa.list_(pa.float32())
        ),
        "observation.state.right_fig6d": pa.array(
            [row[6:] for row in all_state_right], type=pa.list_(pa.float32())
        ),
        "observation.state.left_ee_pose_gripper_base": pa.array(
            [row[:6] for row in all_state_left], type=pa.list_(pa.float32())
        ),
        "observation.state.left_fig6d": pa.array(
            [row[6:] for row in all_state_left], type=pa.list_(pa.float32())
        ),
        "action.right_ee_pose_gripper_base": pa.array(
            [row[:6] for row in all_action_right], type=pa.list_(pa.float32())
        ),
        "action.right_fig6d": pa.array(
            [row[6:] for row in all_action_right], type=pa.list_(pa.float32())
        ),
        "action.left_ee_pose_gripper_base": pa.array(
            [row[:6] for row in all_action_left], type=pa.list_(pa.float32())
        ),
        "action.left_fig6d": pa.array(
            [row[6:] for row in all_action_left], type=pa.list_(pa.float32())
        ),
    }

    parquet_path = data_dir / "file-000.parquet"
    pq.write_table(pa.table(table_columns), parquet_path)

    # ── tasks.parquet ───────────────────────────────────────────────────────
    tasks_path = meta_dir / "tasks.parquet"
    pq.write_table(
        pa.table({
            "task_index": pa.array([0], type=pa.int64()),
            "task": pa.array([task], type=pa.string()),
        }),
        tasks_path,
    )

    # ── episodes parquet ────────────────────────────────────────────────────
    ep_from = 0
    ep_episode_index = []
    ep_tasks = []
    ep_length = []
    ep_data_chunk = []
    ep_data_file = []
    ep_from_index = []
    ep_to_index = []
    ep_from_ts = []
    ep_to_ts = []
    for ep_idx, count in enumerate(all_frame_counts):
        ep_episode_index.append(ep_idx)
        ep_tasks.append([task])
        ep_length.append(count)
        ep_data_chunk.append(0)
        ep_data_file.append(0)
        ep_from_index.append(ep_from)
        ep_to_index.append(ep_from + count - 1)
        ep_from_ts.append(ep_from / fps)
        ep_to_ts.append((ep_from + count - 1) / fps)
        ep_from += count

    episodes_path = episodes_meta_dir / "file-000.parquet"
    pq.write_table(
        pa.table({
            "episode_index": pa.array(ep_episode_index, type=pa.int64()),
            "tasks": pa.array(ep_tasks, type=pa.list_(pa.string())),
            "length": pa.array(ep_length, type=pa.int64()),
            "data_chunk_index": pa.array(ep_data_chunk, type=pa.int64()),
            "data_file_index": pa.array(ep_data_file, type=pa.int64()),
            "from_index": pa.array(ep_from_index, type=pa.int64()),
            "to_index": pa.array(ep_to_index, type=pa.int64()),
            "from_timestamp": pa.array(ep_from_ts, type=pa.float32()),
            "to_timestamp": pa.array(ep_to_ts, type=pa.float32()),
        }),
        episodes_path,
    )

    # ── stats.json ──────────────────────────────────────────────────────────
    state_right_arr = np.asarray(all_state_right, dtype=np.float32)
    state_left_arr = np.asarray(all_state_left, dtype=np.float32)

    stats = {
        "observation.state.right_ee_pose_gripper_base": _feature_stats(
            state_right_arr[:, :6]
        ),
        "observation.state.right_fig6d": _feature_stats(state_right_arr[:, 6:]),
        "observation.state.left_ee_pose_gripper_base": _feature_stats(
            state_left_arr[:, :6]
        ),
        "observation.state.left_fig6d": _feature_stats(state_left_arr[:, 6:]),
        "action.right_ee_pose_gripper_base": _feature_stats(
            np.asarray(all_action_right, dtype=np.float32)[:, :6]
        ),
        "action.right_fig6d": _feature_stats(
            np.asarray(all_action_right, dtype=np.float32)[:, 6:]
        ),
        "action.left_ee_pose_gripper_base": _feature_stats(
            np.asarray(all_action_left, dtype=np.float32)[:, :6]
        ),
        "action.left_fig6d": _feature_stats(
            np.asarray(all_action_left, dtype=np.float32)[:, 6:]
        ),
    }
    _write_json(meta_dir / "stats.json", stats)

    # ── info.json ───────────────────────────────────────────────────────────
    eef_labels = EEF6D_LABELS
    fig_labels = RH56E2_JOINT_LABELS

    right_ee_names = _prefixed(eef_labels, HandType.right)
    right_fig_names = _prefixed(fig_labels, HandType.right)
    left_ee_names = _prefixed(eef_labels, HandType.left)
    left_fig_names = _prefixed(fig_labels, HandType.left)

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "is_first": {"dtype": "bool", "shape": [1], "names": None},
        "is_last": {"dtype": "bool", "shape": [1], "names": None},
        "is_terminal": {"dtype": "bool", "shape": [1], "names": None},
        # ── Video feature ───────────────────────────────────────────────────
        "observation.images.head_cam": {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": None,
            "fps": fps,
            "codec": "av1",
        },
        # ── Observation state ───────────────────────────────────────────────
        "observation.state.right_ee_pose_gripper_base": {
            "dtype": "float32",
            "shape": [6],
            "names": {"axes": right_ee_names},
            "fps": fps,
        },
        "observation.state.right_fig6d": {
            "dtype": "float32",
            "shape": [6],
            "names": {"axes": right_fig_names},
            "fps": fps,
        },
        "observation.state.left_ee_pose_gripper_base": {
            "dtype": "float32",
            "shape": [6],
            "names": {"axes": left_ee_names},
            "fps": fps,
        },
        "observation.state.left_fig6d": {
            "dtype": "float32",
            "shape": [6],
            "names": {"axes": left_fig_names},
            "fps": fps,
        },
        # ── Action ──────────────────────────────────────────────────────────
        "action.right_ee_pose_gripper_base": {
            "dtype": "float32",
            "shape": [6],
            "names": {"axes": right_ee_names},
            "fps": fps,
        },
        "action.right_fig6d": {
            "dtype": "float32",
            "shape": [6],
            "names": {"axes": right_fig_names},
            "fps": fps,
        },
        "action.left_ee_pose_gripper_base": {
            "dtype": "float32",
            "shape": [6],
            "names": {"axes": left_ee_names},
            "fps": fps,
        },
        "action.left_fig6d": {
            "dtype": "float32",
            "shape": [6],
            "names": {"axes": left_fig_names},
            "fps": fps,
        },
    }

    splits = {}
    if total_frames > 0:
        splits["train"] = f"0:{len(hdf5_files)}"

    info = {
        "codebase_version": "v3.0",
        "robot_type": "inspire_rh56e2_bimanual",
        "total_episodes": len(hdf5_files),
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": splits,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/observation.images.head_cam/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "source": {
            "hdf5_dir": str(Path(hdf5_dir).absolute()),
            "num_episodes": len(hdf5_files),
            "action_shift": action_shift,
            "scaling_factor": scaling_factor,
            "robot_wrist_z_offset": robot_wrist_z_offset,
            "retarget_origin_link": retarget_origin_link,
            "tip_origin_scale": tip_origin_scale,
            "image_frame_convention": "opencv -> FLU (x-forward, y-left, z-up)",
            "eef_format": "x, y, z, roll, pitch, yaw",
            "fig6d_order": RH56E2_JOINT_LABELS,
        },
    }
    _write_json(meta_dir / "info.json", info)

    print(f"\nDataset saved to: {out_root}")
    print(f"  Parquet: {parquet_path}")
    print(f"  Episodes: {episodes_path}")
    print(f"  Tasks: {tasks_path}")
    print(f"  Info: {meta_dir / 'info.json'}")
    print(f"  Stats: {meta_dir / 'stats.json'}")
    print(f"  Videos: {video_dir}")
    print(f"  Total episodes: {len(hdf5_files)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Action shift: {action_shift}")


if __name__ == "__main__":
    tyro.cli(main)
