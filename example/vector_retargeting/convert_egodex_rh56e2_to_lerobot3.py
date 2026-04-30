"""
Convert EgoDex RH56E2 retargeting results into a LeRobot v3.0-style dataset.

This script intentionally reuses the RH56E2 retargeting path from
visualize_egodex_retargeting_rh56e2.py, because that path is the tuned and
visually accepted one.

Default output:
    <output_dir>/
        data/chunk-000/file-000.parquet
        meta/info.json
        meta/stats.json
        meta/tasks.parquet
        meta/episodes/chunk-000/file-000.parquet

Example:
    .venv/bin/python example/vector_retargeting/convert_egodex_rh56e2_to_lerobot3.py \
        --hdf5-path /home/user/ml-egodex/test/clean_cups/0.hdf5 \
        --output-dir /home/user/ml-egodex/test/clean_cups/lerobot3_rh56e2 \
        --hand-type right \
        --robot-wrist-z-offset -0.00
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

from dex_retargeting.constants import HandType, OPERATOR2MANO
from visualize_egodex_retargeting_rh56e2 import (
    EGODEX_TIP_JOINTS,
    HAND_JOINTS,
    RH56E2_DEFAULT_TIP_ORIGIN_SCALE,
    RH56E2_JOINT_LABELS,
    _build_rh56e2_retargeting,
    _rh56e2_lerobot_joint_names,
)

HandMode = Literal["right", "left", "bimanual"]
ActionMode = Literal["hand", "eef_hand"]
ImageFrameConvention = Literal["opencv", "identity"]

HAND6_FROM_LEGACY9 = [7, 8, 6, 5, 4, 3]
EEF_POSE9_LABELS = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_x_axis_x",
    "eef_x_axis_y",
    "eef_x_axis_z",
    "eef_y_axis_x",
    "eef_y_axis_y",
    "eef_y_axis_z",
]


def _hands_from_mode(hand_type: HandMode) -> List[HandType]:
    if hand_type == "right":
        return [HandType.right]
    if hand_type == "left":
        return [HandType.left]
    return [HandType.left, HandType.right]


def _feature_stats(matrix: np.ndarray) -> dict:
    if matrix.size == 0:
        return {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
            "count": 0,
            "q01": [],
            "q10": [],
            "q50": [],
            "q90": [],
            "q99": [],
        }

    q = np.quantile(matrix, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": matrix.min(axis=0).tolist(),
        "max": matrix.max(axis=0).tolist(),
        "mean": matrix.mean(axis=0).tolist(),
        "std": matrix.std(axis=0).tolist(),
        "count": int(matrix.shape[0]),
        "q01": q[0].tolist(),
        "q10": q[1].tolist(),
        "q50": q[2].tolist(),
        "q90": q[3].tolist(),
        "q99": q[4].tolist(),
    }


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
    if robot_qpos.shape[0] < 9:
        raise ValueError(
            f"RH56E2 retargeting qpos should be 9D, got {robot_qpos.shape}"
        )
    return robot_qpos[HAND6_FROM_LEGACY9].astype(np.float32)


def _image_basis_to_flu(convention: ImageFrameConvention) -> np.ndarray:
    if convention == "identity":
        return np.eye(3, dtype=np.float32)

    # OpenCV camera/image axes: x right, y down, z forward.
    # Target first-image frame: x forward, y left, z up.
    return np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _rotation_to_6d_axes(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation[:, 0], rotation[:, 1]]).astype(np.float32)


def _eef_pose9_in_first_image_flu(
    transforms_group,
    frame_idx: int,
    hand_type: HandType,
    world_t_image0_inv: np.ndarray,
    image_to_flu: np.ndarray,
) -> np.ndarray:
    wrist_key = _hand_wrist_key(hand_type)
    image0_t_eef = world_t_image0_inv @ transforms_group[wrist_key][frame_idx]
    rotation = image_to_flu @ image0_t_eef[:3, :3] @ image_to_flu.T
    position = image_to_flu @ image0_t_eef[:3, 3]
    return np.concatenate([position, _rotation_to_6d_axes(rotation)]).astype(np.float32)


def _prefixed(labels: List[str], hand_type: HandType) -> List[str]:
    return [f"{hand_type.name}/{label}" for label in labels]


def _state_names(hands: List[HandType], action_mode: ActionMode) -> List[str]:
    names: List[str] = []
    for hand in hands:
        if action_mode == "eef_hand":
            names.extend(_prefixed(EEF_POSE9_LABELS, hand))
        names.extend(_prefixed(RH56E2_JOINT_LABELS, hand))
    return names


def _hand_names(hands: List[HandType]) -> List[str]:
    names: List[str] = []
    for hand in hands:
        names.extend(_prefixed(RH56E2_JOINT_LABELS, hand))
    return names


def _eef_names(hands: List[HandType]) -> List[str]:
    names: List[str] = []
    for hand in hands:
        names.extend(_prefixed(EEF_POSE9_LABELS, hand))
    return names


def _shift_rows(rows: List[List[float]], shift: int) -> List[List[float]]:
    if shift < 0:
        raise ValueError("action_shift must be >= 0")
    if not rows or shift == 0:
        return rows
    last = len(rows) - 1
    return [rows[min(i + shift, last)] for i in range(len(rows))]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(
    hdf5_path: str,
    output_dir: str,
    hand_type: HandMode = "right",
    fps: float = 30.0,
    episode_index: int = 0,
    task: str = "egodex rh56e2 retargeting",
    action_mode: ActionMode = "eef_hand",
    action_shift: int = 0,
    scaling_factor: Optional[float] = None,
    robot_wrist_z_offset: float = -0.0,
    retarget_origin_link: Optional[str] = None,
    tip_origin_scale: float = RH56E2_DEFAULT_TIP_ORIGIN_SCALE,
    image_frame_convention: ImageFrameConvention = "opencv",
    max_frames: Optional[int] = None,
    debug_urdf_path: Optional[str] = None,
):
    """Convert one EgoDex HDF5 episode to a LeRobot 3.0-style RH56E2 dataset.

    action_mode:
        "eef_hand" writes observation.state/action as per-hand
        [eef_pose9, hand6]. "hand" writes only the per-hand hand6 vectors.
    action_shift:
        0 keeps action[t] aligned with observation[t]. 1 uses next-frame action.
    image_frame_convention:
        "opencv" converts first-image axes from x-right/y-down/z-forward to
        x-forward/y-left/z-up. "identity" only makes poses relative to frame 0.
    """
    if action_shift < 0:
        raise ValueError("action_shift must be >= 0")

    hands = _hands_from_mode(hand_type)
    out_root = Path(output_dir).absolute()
    data_dir = out_root / "data" / "chunk-000"
    meta_dir = out_root / "meta"
    episodes_meta_dir = meta_dir / "episodes" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta_dir.mkdir(parents=True, exist_ok=True)

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

    frame_indices: List[int] = []
    global_indices: List[int] = []
    episode_indices: List[int] = []
    timestamps: List[float] = []
    task_indices: List[int] = []
    is_first: List[bool] = []
    is_last: List[bool] = []
    is_terminal: List[bool] = []

    states: List[List[float]] = []
    hand_vectors: List[List[float]] = []
    eef_poses: List[List[float]] = []
    hand_vectors_by_hand: Dict[HandType, List[List[float]]] = {
        hand: [] for hand in hands
    }
    eef_poses_by_hand: Dict[HandType, List[List[float]]] = {
        hand: [] for hand in hands
    }

    with h5py.File(hdf5_path, "r") as h5_file:
        transforms = h5_file["transforms"]
        if "camera" not in transforms:
            raise ValueError("Input file does not contain transforms/camera.")

        required_wrist_keys = [_hand_wrist_key(hand) for hand in hands]
        missing = [key for key in required_wrist_keys if key not in transforms]
        if missing:
            raise ValueError(f"Input file is missing required wrist transforms: {missing}")

        num_frames = min(
            transforms[key].shape[0] for key in ["camera"] + required_wrist_keys
        )
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)

        world_t_image0_inv = np.linalg.inv(transforms["camera"][0])
        image_to_flu = _image_basis_to_flu(image_frame_convention)

        for frame_idx in tqdm.trange(num_frames, desc="Converting RH56E2 retargeting"):
            per_frame_state: List[float] = []
            per_frame_hand: List[float] = []
            per_frame_eef: List[float] = []

            for hand in hands:
                retargeting = retargetings[hand]
                retarget_indices = retargeting.optimizer.target_link_human_indices
                ref_value = _compute_ref_value(
                    transforms,
                    frame_idx,
                    hand,
                    retarget_indices,
                )
                legacy_qpos = retargeting.retarget(ref_value)
                hand6 = _legacy9_to_hand6(legacy_qpos)
                eef_pose9 = _eef_pose9_in_first_image_flu(
                    transforms,
                    frame_idx,
                    hand,
                    world_t_image0_inv,
                    image_to_flu,
                )

                if action_mode == "eef_hand":
                    per_frame_state.extend(eef_pose9.tolist())
                per_frame_state.extend(hand6.tolist())
                per_frame_hand.extend(hand6.tolist())
                per_frame_eef.extend(eef_pose9.tolist())
                hand_vectors_by_hand[hand].append(hand6.tolist())
                eef_poses_by_hand[hand].append(eef_pose9.tolist())

            frame_indices.append(frame_idx)
            global_indices.append(frame_idx)
            episode_indices.append(episode_index)
            timestamps.append(frame_idx / fps)
            task_indices.append(0)
            is_first.append(frame_idx == 0)
            is_last.append(frame_idx == num_frames - 1)
            is_terminal.append(frame_idx == num_frames - 1)
            states.append(per_frame_state)
            hand_vectors.append(per_frame_hand)
            eef_poses.append(per_frame_eef)

    actions = _shift_rows(states, action_shift)
    action_hands = _shift_rows(hand_vectors, action_shift)
    action_eefs = _shift_rows(eef_poses, action_shift)

    table_columns = {
        "index": pa.array(global_indices, type=pa.int64()),
        "episode_index": pa.array(episode_indices, type=pa.int64()),
        "frame_index": pa.array(frame_indices, type=pa.int64()),
        "timestamp": pa.array(timestamps, type=pa.float32()),
        "task_index": pa.array(task_indices, type=pa.int64()),
        "is_first": pa.array(is_first, type=pa.bool_()),
        "is_last": pa.array(is_last, type=pa.bool_()),
        "is_terminal": pa.array(is_terminal, type=pa.bool_()),
        "observation.state": pa.array(states, type=pa.list_(pa.float32())),
        "observation.eef_pose": pa.array(eef_poses, type=pa.list_(pa.float32())),
        "observation.hand": pa.array(hand_vectors, type=pa.list_(pa.float32())),
        "action": pa.array(actions, type=pa.list_(pa.float32())),
        "action.eef_pose": pa.array(action_eefs, type=pa.list_(pa.float32())),
        "action.hand": pa.array(action_hands, type=pa.list_(pa.float32())),
    }
    for hand in hands:
        suffix = hand.name
        table_columns[f"observation.hand_{suffix}"] = pa.array(
            hand_vectors_by_hand[hand], type=pa.list_(pa.float32())
        )
        table_columns[f"observation.eef_pose_{suffix}"] = pa.array(
            eef_poses_by_hand[hand], type=pa.list_(pa.float32())
        )
        table_columns[f"action_{suffix}"] = pa.array(
            _shift_rows(hand_vectors_by_hand[hand], action_shift),
            type=pa.list_(pa.float32()),
        )

    parquet_path = data_dir / "file-000.parquet"
    pq.write_table(pa.table(table_columns), parquet_path)

    tasks_path = meta_dir / "tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array([task], type=pa.string()),
            }
        ),
        tasks_path,
    )

    episodes_path = episodes_meta_dir / "file-000.parquet"
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([episode_index], type=pa.int64()),
                "tasks": pa.array([[task]], type=pa.list_(pa.string())),
                "length": pa.array([len(frame_indices)], type=pa.int64()),
                "data_chunk_index": pa.array([0], type=pa.int64()),
                "data_file_index": pa.array([0], type=pa.int64()),
                "from_index": pa.array([0], type=pa.int64()),
                "to_index": pa.array([len(frame_indices) - 1], type=pa.int64()),
                "from_timestamp": pa.array([0.0], type=pa.float32()),
                "to_timestamp": pa.array(
                    [timestamps[-1] if timestamps else 0.0], type=pa.float32()
                ),
            }
        ),
        episodes_path,
    )

    state_axis_names = _state_names(hands, action_mode)
    hand_axis_names = _hand_names(hands)
    eef_axis_names = _eef_names(hands)

    stats = {
        "observation.state": _feature_stats(np.asarray(states, dtype=np.float32)),
        "observation.hand": _feature_stats(np.asarray(hand_vectors, dtype=np.float32)),
        "observation.eef_pose": _feature_stats(np.asarray(eef_poses, dtype=np.float32)),
        "action": _feature_stats(np.asarray(actions, dtype=np.float32)),
        "action.hand": _feature_stats(np.asarray(action_hands, dtype=np.float32)),
        "action.eef_pose": _feature_stats(np.asarray(action_eefs, dtype=np.float32)),
    }
    for hand in hands:
        stats[f"observation.hand_{hand.name}"] = _feature_stats(
            np.asarray(hand_vectors_by_hand[hand], dtype=np.float32)
        )
        stats[f"observation.eef_pose_{hand.name}"] = _feature_stats(
            np.asarray(eef_poses_by_hand[hand], dtype=np.float32)
        )
        stats[f"action_{hand.name}"] = _feature_stats(
            np.asarray(
                _shift_rows(hand_vectors_by_hand[hand], action_shift),
                dtype=np.float32,
            )
        )
    _write_json(meta_dir / "stats.json", stats)

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "is_first": {"dtype": "bool", "shape": [1], "names": None},
        "is_last": {"dtype": "bool", "shape": [1], "names": None},
        "is_terminal": {"dtype": "bool", "shape": [1], "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_axis_names)],
            "names": {"axes": state_axis_names},
            "fps": int(fps),
        },
        "observation.hand": {
            "dtype": "float32",
            "shape": [len(hand_axis_names)],
            "names": {"axes": hand_axis_names},
            "fps": int(fps),
        },
        "observation.eef_pose": {
            "dtype": "float32",
            "shape": [len(eef_axis_names)],
            "names": {"axes": eef_axis_names},
            "fps": int(fps),
        },
        "action": {
            "dtype": "float32",
            "shape": [len(state_axis_names)],
            "names": {"axes": state_axis_names},
            "fps": int(fps),
        },
        "action.hand": {
            "dtype": "float32",
            "shape": [len(hand_axis_names)],
            "names": {"axes": hand_axis_names},
            "fps": int(fps),
        },
        "action.eef_pose": {
            "dtype": "float32",
            "shape": [len(eef_axis_names)],
            "names": {"axes": eef_axis_names},
            "fps": int(fps),
        },
    }
    for hand in hands:
        hand_names = _prefixed(RH56E2_JOINT_LABELS, hand)
        eef_names = _prefixed(EEF_POSE9_LABELS, hand)
        features[f"observation.hand_{hand.name}"] = {
            "dtype": "float32",
            "shape": [len(hand_names)],
            "names": {"axes": hand_names},
            "fps": int(fps),
        }
        features[f"observation.eef_pose_{hand.name}"] = {
            "dtype": "float32",
            "shape": [len(eef_names)],
            "names": {"axes": eef_names},
            "fps": int(fps),
        }
        features[f"action_{hand.name}"] = {
            "dtype": "float32",
            "shape": [len(hand_names)],
            "names": {"axes": hand_names},
            "fps": int(fps),
        }

    info = {
        "codebase_version": "v3.0",
        "robot_type": f"inspire_rh56e2_{hand_type}",
        "total_episodes": 1,
        "total_frames": len(frame_indices),
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": int(fps),
        "splits": {},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": features,
        "source": {
            "hdf5_path": str(Path(hdf5_path).absolute()),
            "hand_type": hand_type,
            "action_mode": action_mode,
            "action_shift": action_shift,
            "scaling_factor": scaling_factor,
            "robot_wrist_z_offset": robot_wrist_z_offset,
            "retarget_origin_link": retarget_origin_link,
            "tip_origin_scale": tip_origin_scale,
            "image_frame_convention": image_frame_convention,
            "image_frame_axes": "x->forward, y->left, z->up",
            "hand6_order": RH56E2_JOINT_LABELS,
            "legacy9_joint_names": {
                hand.name: retargetings[hand].optimizer.robot.dof_joint_names
                for hand in hands
            },
            "lerobot6_joint_names": {
                hand.name: _rh56e2_lerobot_joint_names(hand) for hand in hands
            },
            "legacy9_to_hand6_indices": HAND6_FROM_LEGACY9,
        },
    }
    _write_json(meta_dir / "info.json", info)

    print(f"Saved data: {parquet_path}")
    print(f"Saved episodes: {episodes_path}")
    print(f"Saved tasks: {tasks_path}")
    print(f"Saved info: {meta_dir / 'info.json'}")
    print(f"Saved stats: {meta_dir / 'stats.json'}")
    print(f"Action mode: {action_mode}")
    print(f"Hand 6D order: {RH56E2_JOINT_LABELS}")


if __name__ == "__main__":
    tyro.cli(main)
