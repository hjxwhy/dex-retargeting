"""
Visualize a LeRobot v3.0 RH56E2 dataset in Rerun.

The goal is to inspect the data after conversion, not to re-run retargeting:
    - 3D end-effector pose in the first-image FLU frame
    - state/action curves for eef_pose
    - state/action curves for RH56E2 hand6
    - per-frame numeric values

Example:
    .venv/bin/python example/vector_retargeting/visualize_lerobot3_rh56e2.py \
        --dataset-dir /home/user/ml-egodex/test/clean_cups/lerobot3_rh56e2

Then open:
    rerun /home/user/ml-egodex/test/clean_cups/lerobot3_rh56e2/lerobot3_rh56e2_vis.rrd
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import tqdm
import tyro

HAND6_LABELS = ["thumb_oc", "thumb_lat", "index", "middle", "ring", "little"]
EEF9_LABELS = [
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

COLOR_STATE = [80, 200, 120, 255]
COLOR_ACTION = [220, 80, 170, 255]
COLOR_X = [255, 60, 60, 255]
COLOR_Y = [80, 220, 80, 255]
COLOR_Z = [80, 120, 255, 255]
COLOR_POINT = [255, 220, 80, 255]
DEFAULT_VIDEO_ENTITY = "exterior_1_left"


def _check_rerun_sdk() -> None:
    if not all(hasattr(rr, name) for name in ("init", "save", "log")):
        raise ImportError("当前导入的 rerun 不是 Rerun SDK，请在项目 .venv 中运行。")
    if not all(hasattr(rr, name) for name in ("Arrows3D", "Scalars", "AnyValues")):
        raise ImportError("当前 Rerun SDK 缺少本脚本需要的 3D/curve archetypes。")


def _read_info(dataset_dir: Path) -> dict:
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Cannot find LeRobot info.json: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _resolve_data_path(dataset_dir: Path, info: dict) -> Path:
    data_path = info.get("data_path")
    if isinstance(data_path, str):
        candidate = dataset_dir / data_path.format(chunk_index=0, file_index=0)
        if candidate.exists():
            return candidate

    matches = sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet"))
    if matches:
        return matches[0]

    matches = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find parquet data file under: {dataset_dir / 'data'}")


def _resolve_video_path(dataset_dir: Path, info: dict, explicit_path: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path).expanduser().absolute())

    source = info.get("source", {})
    hdf5_path = source.get("hdf5_path")
    if isinstance(hdf5_path, str):
        candidates.append(Path(hdf5_path).with_suffix(".mp4"))

    parquet_stem = _resolve_data_path(dataset_dir, info).stem
    candidates.append(dataset_dir / f"{parquet_stem}.mp4")
    candidates.append(dataset_dir.parent / f"{parquet_stem}.mp4")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _probe_video_size(video_path: Path) -> Tuple[int, int]:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        text=True,
    )
    payload = json.loads(output)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"Cannot probe video size for {video_path}")
    stream = streams[0]
    return int(stream["width"]), int(stream["height"])


class _FFmpegFrameReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.width, self.height = _probe_video_size(video_path)
        self.frame_bytes = self.width * self.height * 3
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read(self) -> Optional[np.ndarray]:
        assert self._proc.stdout is not None
        buf = self._proc.stdout.read(self.frame_bytes)
        if len(buf) != self.frame_bytes:
            return None
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
        return frame.copy()

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def _feature_axes(info: dict, feature: str) -> List[str]:
    feature_info = info.get("features", {}).get(feature, {})
    names = feature_info.get("names")
    if isinstance(names, dict):
        axes = names.get("axes")
        if isinstance(axes, list):
            return axes
    return []


def _hands_from_axes(axes: List[str]) -> List[str]:
    hands: List[str] = []
    for axis in axes:
        if "/" not in axis:
            continue
        hand = axis.split("/", 1)[0]
        if hand not in hands:
            hands.append(hand)
    return hands


def _rows(table_dict: dict, column: str) -> Optional[List[List[float]]]:
    values = table_dict.get(column)
    if values is None:
        return None
    return [list(map(float, row)) for row in values]


def _combined_feature_by_hand(
    rows: List[List[float]],
    axes: List[str],
    hand: str,
) -> List[List[float]]:
    indices = [
        idx
        for idx, axis in enumerate(axes)
        if axis.startswith(f"{hand}/")
    ]
    return [[row[idx] for idx in indices] for row in rows]


def _column_or_split(
    table_dict: dict,
    column: str,
    combined_column: str,
    combined_axes: List[str],
    hand: str,
) -> List[List[float]]:
    direct = _rows(table_dict, column)
    if direct is not None:
        return direct

    combined = _rows(table_dict, combined_column)
    if combined is None:
        raise KeyError(f"Missing both {column} and {combined_column}")
    return _combined_feature_by_hand(combined, combined_axes, hand)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return v
    return v / norm


def _eef9_to_pose(eef9: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    if len(eef9) != 9:
        raise ValueError(f"Expected eef_pose9, got {len(eef9)} values")
    position = np.asarray(eef9[:3], dtype=np.float32)
    x_axis = _normalize(np.asarray(eef9[3:6], dtype=np.float32))
    y_axis = np.asarray(eef9[6:9], dtype=np.float32)
    y_axis = _normalize(y_axis - x_axis * float(np.dot(x_axis, y_axis)))
    z_axis = _normalize(np.cross(x_axis, y_axis))
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    return position, rotation


def _safe_key(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")


def _log_series_style(path: str, name: str, color: List[int]) -> None:
    if hasattr(rr, "SeriesLines"):
        rr.log(
            path,
            rr.SeriesLines(names=[name], colors=[color], widths=[1.5]),
            static=True,
        )


def _log_scalar(path: str, value: float, name: str, color: List[int]) -> None:
    _log_series_style(path, name, color)
    rr.log(path, rr.Scalars([float(value)]))


def _log_pose_axes(
    path: str,
    position: np.ndarray,
    rotation: np.ndarray,
    axis_scale: float,
) -> None:
    vectors = [
        rotation[:, 0] * axis_scale,
        rotation[:, 1] * axis_scale,
        rotation[:, 2] * axis_scale,
    ]
    origins = [position, position, position]
    rr.log(
        path,
        rr.Arrows3D(
            origins=origins,
            vectors=vectors,
            colors=[COLOR_X, COLOR_Y, COLOR_Z],
            labels=["X (+forward)", "Y (+left)", "Z (+up)"],
            radii=[0.003, 0.003, 0.003],
        ),
    )
    rr.log(f"{path}/origin", rr.Points3D([position], radii=0.007, colors=COLOR_POINT))


def _log_reference_axes(axis_scale: float) -> None:
    origin = np.zeros(3, dtype=np.float32)
    rr.log(
        "world/first_image_flu_axes",
        rr.Arrows3D(
            origins=[origin, origin, origin],
            vectors=[
                [axis_scale, 0.0, 0.0],
                [0.0, axis_scale, 0.0],
                [0.0, 0.0, axis_scale],
            ],
            colors=[COLOR_X, COLOR_Y, COLOR_Z],
            labels=["X (+forward)", "Y (+left)", "Z (+up)"],
            radii=[0.002, 0.002, 0.002],
        ),
        static=True,
    )


def _log_eef_curves(
    hand: str,
    state_eef: List[float],
    action_eef: List[float],
) -> None:
    for idx, label in enumerate(EEF9_LABELS):
        base = f"curves/{hand}/eef_pose/{label}"
        _log_scalar(f"{base}/state", state_eef[idx], f"{label}/state", COLOR_STATE)
        _log_scalar(f"{base}/action", action_eef[idx], f"{label}/action", COLOR_ACTION)


def _log_hand_curves(
    hand: str,
    state_hand: List[float],
    action_hand: List[float],
) -> None:
    for idx, label in enumerate(HAND6_LABELS):
        base = f"curves/{hand}/hand6/{label}"
        _log_scalar(f"{base}/state", state_hand[idx], f"{label}/state", COLOR_STATE)
        _log_scalar(f"{base}/action", action_hand[idx], f"{label}/action", COLOR_ACTION)


def _log_state_values(
    hand: str,
    frame_index: int,
    state_eef: List[float],
    action_eef: List[float],
    state_hand: List[float],
    action_hand: List[float],
) -> None:
    values: Dict[str, float | int] = {"frame": int(frame_index)}
    for label, value in zip(EEF9_LABELS, state_eef):
        values[f"state_{_safe_key(label)}"] = float(value)
    for label, value in zip(EEF9_LABELS, action_eef):
        values[f"action_{_safe_key(label)}"] = float(value)
    for label, value in zip(HAND6_LABELS, state_hand):
        values[f"state_{_safe_key(label)}"] = float(value)
    for label, value in zip(HAND6_LABELS, action_hand):
        values[f"action_{_safe_key(label)}"] = float(value)
    rr.log(f"state_values/{hand}", rr.AnyValues(**values))


def main(
    dataset_dir: str = "/home/user/ml-egodex/test/clean_cups/lerobot3_rh56e2",
    rrd_path: Optional[str] = None,
    video_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    axis_scale: float = 0.08,
    video_entity: str = DEFAULT_VIDEO_ENTITY,
):
    """Visualize converted RH56E2 LeRobot data in Rerun."""
    _check_rerun_sdk()

    dataset_root = Path(dataset_dir).absolute()
    info = _read_info(dataset_root)
    parquet_path = _resolve_data_path(dataset_root, info)
    resolved_video_path = _resolve_video_path(dataset_root, info, video_path)
    if rrd_path is None:
        rrd_path = str(dataset_root / "lerobot3_rh56e2_vis.rrd")

    table = pq.read_table(parquet_path)
    table_dict = table.to_pydict()
    num_frames = table.num_rows
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)

    eef_axes = _feature_axes(info, "observation.eef_pose")
    hand_axes = _feature_axes(info, "observation.hand")
    hands = _hands_from_axes(eef_axes) or _hands_from_axes(hand_axes)
    if not hands:
        raise ValueError("Cannot infer hand names from meta/info.json feature axes.")

    observation_eef = _rows(table_dict, "observation.eef_pose")
    action_eef = _rows(table_dict, "action.eef_pose")
    if observation_eef is None or action_eef is None:
        raise KeyError("Dataset must contain observation.eef_pose and action.eef_pose.")

    frame_indices = table_dict.get("frame_index", list(range(table.num_rows)))

    rr.init("lerobot3_rh56e2_vis")
    rr.save(rrd_path)
    _log_reference_axes(axis_scale)

    video_reader = None
    if resolved_video_path is not None:
        video_reader = _FFmpegFrameReader(resolved_video_path)
        print(f"Video source: {resolved_video_path}")
        print(f"Video entity: {video_entity}")
    else:
        print("Video source: not found; continuing without left video.")

    hand_data = {}
    for hand in hands:
        hand_data[hand] = {
            "state_eef": _column_or_split(
                table_dict,
                f"observation.eef_pose_{hand}",
                "observation.eef_pose",
                eef_axes,
                hand,
            ),
            "action_eef": _combined_feature_by_hand(action_eef, eef_axes, hand),
            "state_hand": _column_or_split(
                table_dict,
                f"observation.hand_{hand}",
                "observation.hand",
                hand_axes,
                hand,
            ),
            "action_hand": _column_or_split(
                table_dict,
                f"action_{hand}",
                "action.hand",
                hand_axes,
                hand,
            ),
        }

    for row_idx in tqdm.trange(num_frames, desc="Logging LeRobot RH56E2 to Rerun"):
        frame_index = int(frame_indices[row_idx])
        rr.set_time("frame", sequence=frame_index)

        if video_reader is not None:
            frame = video_reader.read()
            if frame is not None:
                rr.log(video_entity, rr.Image(frame))

        for hand in hands:
            state_eef = hand_data[hand]["state_eef"][row_idx]
            action_eef_for_pose = hand_data[hand]["action_eef"][row_idx]
            state_hand = hand_data[hand]["state_hand"][row_idx]
            action_hand = hand_data[hand]["action_hand"][row_idx]

            position, rotation = _eef9_to_pose(state_eef)
            rr.log(
                f"poses/{hand}/ee_pose",
                rr.Transform3D(translation=position, mat3x3=rotation),
            )
            _log_pose_axes(f"poses/{hand}/ee_pose_axes", position, rotation, axis_scale)
            _log_eef_curves(hand, state_eef, action_eef_for_pose)
            _log_hand_curves(hand, state_hand, action_hand)
            _log_state_values(
                hand,
                frame_index,
                state_eef,
                action_eef_for_pose,
                state_hand,
                action_hand,
            )

    if video_reader is not None:
        video_reader.close()

    print(f"Read LeRobot data: {parquet_path}")
    print(f"Saved Rerun recording: {rrd_path}")
    print(f"View: rerun {rrd_path}")
    print("Expected checks:")
    print(f"  - {video_entity}: left video if mp4 was found")
    print("  - poses/*/ee_pose_axes: X forward, Y left, Z up")
    print("  - curves/*/eef_pose/*: state/action eef curves")
    print("  - curves/*/hand6/*: state/action RH56E2 6D finger curves")


if __name__ == "__main__":
    tyro.cli(main)
