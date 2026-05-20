"""Shared helpers for adapting EgoDex hand poses to wuji-retargeting.

The helpers in this file intentionally live outside ``wuji-retargeting`` so the
upstream package can remain unchanged.  EgoDex exposes ARKit/Vision Pro-style
SE(3) transforms, while wuji-retargeting consumes MediaPipe-style ``(21, 3)``
landmarks.  The conversion mirrors the VisionPro adapter already shipped by
wuji-retargeting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np


HandSide = Literal["left", "right"]


# Same mapping as wuji-retargeting/example/input_devices/visionpro.py.
VP_TO_MEDIAPIPE = (
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
)


MEDIAPIPE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


def egodex_vp25_joint_names(hand_side: HandSide) -> list[str]:
    """Return EgoDex transform names in the 25-joint VisionPro order."""
    prefix = hand_side

    return [
        f"{prefix}Hand",
        f"{prefix}ThumbKnuckle",
        f"{prefix}ThumbIntermediateBase",
        f"{prefix}ThumbIntermediateTip",
        f"{prefix}ThumbTip",
        f"{prefix}IndexFingerMetacarpal",
        f"{prefix}IndexFingerKnuckle",
        f"{prefix}IndexFingerIntermediateBase",
        f"{prefix}IndexFingerIntermediateTip",
        f"{prefix}IndexFingerTip",
        f"{prefix}MiddleFingerMetacarpal",
        f"{prefix}MiddleFingerKnuckle",
        f"{prefix}MiddleFingerIntermediateBase",
        f"{prefix}MiddleFingerIntermediateTip",
        f"{prefix}MiddleFingerTip",
        f"{prefix}RingFingerMetacarpal",
        f"{prefix}RingFingerKnuckle",
        f"{prefix}RingFingerIntermediateBase",
        f"{prefix}RingFingerIntermediateTip",
        f"{prefix}RingFingerTip",
        f"{prefix}LittleFingerMetacarpal",
        f"{prefix}LittleFingerKnuckle",
        f"{prefix}LittleFingerIntermediateBase",
        f"{prefix}LittleFingerIntermediateTip",
        f"{prefix}LittleFingerTip",
    ]


def iter_hdf5_files(path: str | Path, recursive: bool = False) -> list[Path]:
    """Return sorted HDF5 files from a file or directory path."""
    path = Path(path)
    if path.is_file():
        return [path]
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    return sorted(path.glob(pattern))


def require_h5py():
    """Import h5py lazily so CLI help works before dependencies are installed."""
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: h5py. Install EgoDex requirements, for example: "
            "pip install h5py numpy"
        ) from exc
    return h5py


def _num_frames(h5_file) -> int:
    return int(h5_file["transforms"]["camera"].shape[0])


def _has_hand(h5_file, hand_side: HandSide) -> bool:
    transforms = h5_file["transforms"]
    return all(name in transforms for name in egodex_vp25_joint_names(hand_side))


def egodex_vp25_positions(h5_file, frame_idx: int, hand_side: HandSide) -> np.ndarray:
    """Extract one EgoDex hand frame as VisionPro-style ``(25, 3)`` positions."""
    transforms = h5_file["transforms"]
    positions = np.zeros((25, 3), dtype=np.float32)
    for idx, name in enumerate(egodex_vp25_joint_names(hand_side)):
        positions[idx] = transforms[name][frame_idx, :3, 3]
    return positions


def vp25_to_mediapipe21(vp25_positions: np.ndarray) -> np.ndarray:
    """Convert VisionPro-style ``(25, 3)`` positions to MediaPipe ``(21, 3)``."""
    if vp25_positions.shape != (25, 3):
        raise ValueError(f"Expected (25, 3), got {vp25_positions.shape}")
    return vp25_positions[np.asarray(VP_TO_MEDIAPIPE, dtype=np.int64)].astype(np.float32)


def egodex_frame_to_mediapipe21(h5_file, frame_idx: int, hand_side: HandSide) -> np.ndarray:
    """Extract one EgoDex hand frame directly as MediaPipe ``(21, 3)``."""
    return vp25_to_mediapipe21(egodex_vp25_positions(h5_file, frame_idx, hand_side))


def hand_confidence_ok(
    h5_file,
    frame_idx: int,
    hand_side: HandSide,
    threshold: float,
) -> bool:
    """Check EgoDex confidence values for the selected 21 landmarks."""
    if threshold <= 0.0 or "confidences" not in h5_file:
        return True

    confidences = h5_file["confidences"]
    names = egodex_vp25_joint_names(hand_side)
    selected_names = [names[idx] for idx in VP_TO_MEDIAPIPE]
    values = []
    for name in selected_names:
        if name not in confidences:
            return True
        values.append(float(confidences[name][frame_idx]))
    return min(values) >= threshold


def load_egodex_replay(
    hdf5_path: str | Path,
    hand_sides: Iterable[HandSide] = ("left", "right"),
    fps: float = 30.0,
    stride: int = 1,
    max_frames: int | None = None,
    confidence_threshold: float = 0.0,
    include_metadata: bool = True,
) -> list[dict]:
    """Load one EgoDex HDF5 file into wuji-retargeting replay dictionaries."""
    h5py = require_h5py()
    hand_sides = tuple(hand_sides)
    empty = np.zeros((21, 3), dtype=np.float32)
    replay: list[dict] = []

    with h5py.File(hdf5_path, "r") as h5_file:
        total_frames = _num_frames(h5_file)
        end = total_frames if max_frames is None else min(total_frames, max_frames)
        available = {side: _has_hand(h5_file, side) for side in ("left", "right")}

        for frame_idx in range(0, end, stride):
            frame = {
                "t": frame_idx / fps,
                "left_fingers": empty.copy(),
                "right_fingers": empty.copy(),
            }

            for side in hand_sides:
                if not available[side]:
                    continue
                if not hand_confidence_ok(h5_file, frame_idx, side, confidence_threshold):
                    continue
                frame[f"{side}_fingers"] = egodex_frame_to_mediapipe21(
                    h5_file, frame_idx, side
                )

            if include_metadata:
                frame["egodex_frame"] = frame_idx
                frame["egodex_source"] = str(hdf5_path)
            replay.append(frame)

    return replay


def robot_fk_mediapipe21(retargeter, qpos: np.ndarray) -> np.ndarray:
    """Compute Wuji robot FK keypoints in the same 21-point topology."""
    robot = retargeter.optimizer.robot
    robot.compute_forward_kinematics(qpos)

    names = ["palm_link"]
    for finger_idx in range(1, 6):
        names.extend(f"finger{finger_idx}_link{link_idx}" for link_idx in range(1, 5))

    positions = np.zeros((21, 3), dtype=np.float64)
    for idx, name in enumerate(names):
        link_id = robot.get_link_index(name)
        positions[idx] = robot.get_link_pose(link_id)[:3, 3]
    return positions
