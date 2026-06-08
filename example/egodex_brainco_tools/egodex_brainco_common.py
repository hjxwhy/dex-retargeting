#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for EgoDex -> BrainCo hand retargeting."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parents[1]
DEX_RETARGETING_SRC = REPO_ROOT / "src"
BRAINCO_ASSET_DIR = REPO_ROOT / "assets" / "robots" / "hands"
BRAINCO_CONFIG_PATH = TOOL_DIR / "config" / "brainco_vector.yml"

HandSide = Literal["left", "right"]

BRAINCO_NUM_MOTORS = 6
BRAINCO_HARDWARE_JOINT_ORDER = (
    "thumb",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
)
BRAINCO_MAX_QPOS = np.array([1.52, 1.05, 1.47, 1.47, 1.47, 1.47], dtype=np.float32)

# Columns are BrainCo wrist axes expressed in EgoDex wrist coordinates.
#
# EgoDex convention used here:
#   left:  +x four-finger tips, +y palm side, +z pinky/outer side
#   right: +x forearm side, +y back-of-hand side, +z thumb side
#
# BrainCo convention provided for this adapter:
#   left:  +x back of hand, +y forearm, +z thumb side
#   right: +x back of hand, +y forearm, +z pinky side
#
# Therefore, for the left hand:
#   x_brainco = -y_egodex
#   y_brainco = -x_egodex
#   z_brainco = -z_egodex
#
# For the right hand:
#   x_brainco = +y_egodex
#   y_brainco = +x_egodex
#   z_brainco = -z_egodex
#
# For a world-from-EgoDex wrist rotation R_we, the matching BrainCo root
# rotation is:
#   R_wb = R_we @ EGODEX_FROM_BRAINCO_WRIST[side]
EGODEX_FROM_BRAINCO_WRIST = {
    "left": np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    ),
    "right": np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    ),
}


def _rot_x(theta: float) -> np.ndarray:
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


def _rot_z(theta: float) -> np.ndarray:
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


def _rot_y(theta: float) -> np.ndarray:
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


BRAINCO_WRIST_FROM_BASE = {
    "left": _rot_z(-np.pi / 2.0),
    "right": _rot_x(np.pi) @ _rot_z(np.pi / 2.0),
}
BRAINCO_WRIST_OFFSET_LOCAL = np.array([0.0, 0.01, 0.0], dtype=np.float32)
BRAINCO_HAND_ONLY_SCENE_ROT = _rot_y(-np.pi / 2.0) @ _rot_z(-np.pi / 2.0)
BRAINCO_HAND_ONLY_SCENE_OFFSET = np.array([0.0, -0.0, -0.3], dtype=np.float32)

BRAINCO_API_JOINT_NAMES = {
    "left": [
        "left_thumb_metacarpal_joint",
        "left_thumb_proximal_joint",
        "left_index_proximal_joint",
        "left_middle_proximal_joint",
        "left_ring_proximal_joint",
        "left_pinky_proximal_joint",
    ],
    "right": [
        "right_thumb_metacarpal_joint",
        "right_thumb_proximal_joint",
        "right_index_proximal_joint",
        "right_middle_proximal_joint",
        "right_ring_proximal_joint",
        "right_pinky_proximal_joint",
    ],
}

BRAINCO_FK_LINKS = {
    "left": [
        "base_link",
        "left_thumb_metacarpal_Link",
        "left_thumb_proximal_Link",
        "left_thumb_distal_Link",
        "left_thumb_tip",
        "left_index_proximal_Link",
        "left_index_distal_Link",
        "left_index_tip",
        "left_middle_proximal_Link",
        "left_middle_distal_Link",
        "left_middle_tip",
        "left_ring_proximal_Link",
        "left_ring_distal_Link",
        "left_ring_tip",
        "left_pinky_proximal_Link",
        "left_pinky_distal_Link",
        "left_pinky_tip",
    ],
    "right": [
        "base_link",
        "right_thumb_metacarpal_link",
        "right_thumb_proximal_link",
        "right_thumb_distal_link",
        "right_thumb_tip",
        "right_index_proximal_link",
        "right_index_distal_link",
        "right_index_tip",
        "right_middle_proximal_link",
        "right_middle_distal_link",
        "right_middle_tip",
        "right_ring_proximal_link",
        "right_ring_distal_link",
        "right_ring_tip",
        "right_pinky_proximal_link",
        "right_pinky_distal_link",
        "right_pinky_tip",
    ],
}

BRAINCO_FK_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (0, 8),
    (8, 9),
    (9, 10),
    (0, 11),
    (11, 12),
    (12, 13),
    (0, 14),
    (14, 15),
    (15, 16),
)

VP25_HAND_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (0, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (0, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (5, 10),
    (10, 15),
    (15, 20),
)


@dataclass
class BraincoRetargetResult:
    qpos_full: np.ndarray
    qpos_hardware: np.ndarray
    action_01: np.ndarray


@dataclass
class BraincoFrameRetarget:
    side: HandSide
    wrist_position: np.ndarray
    wrist_rotation: np.ndarray
    brainco_base_rotation: np.ndarray
    vp25_world: np.ndarray
    vp25_local: np.ndarray
    result: BraincoRetargetResult


def ensure_import_paths() -> None:
    """Make local dex-retargeting imports available."""
    for path in (DEX_RETARGETING_SRC,):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_brainco_retargeting_pair(config_path: Path | str | None = None):
    """Load BrainCo left/right SeqRetargeting objects with absolute paths."""
    ensure_import_paths()
    from dex_retargeting.retargeting_config import RetargetingConfig

    config_path = Path(config_path) if config_path is not None else BRAINCO_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"BrainCo config not found: {config_path}")

    RetargetingConfig.set_default_urdf_dir(BRAINCO_ASSET_DIR)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    left = RetargetingConfig.from_dict(cfg["left"]).build()
    right = RetargetingConfig.from_dict(cfg["right"]).build()
    return {"left": left, "right": right}


class BraincoRetargeter:
    """Small wrapper matching robot_hand_brainco_virtual.py's mapping logic."""

    def __init__(self, config_path: Path | str | None = None):
        self.retargeting = load_brainco_retargeting_pair(config_path)
        self.indices = {
            side: self.retargeting[side].optimizer.target_link_human_indices
            for side in ("left", "right")
        }
        self.hardware_order = {
            side: [
                self.retargeting[side].joint_names.index(name)
                for name in BRAINCO_API_JOINT_NAMES[side]
            ]
            for side in ("left", "right")
        }

    def reset(self) -> None:
        for retargeting in self.retargeting.values():
            retargeting.reset()

    def retarget(
        self,
        side: HandSide,
        vp25_positions: np.ndarray,
        apply_filter: bool = True,
    ) -> BraincoRetargetResult:
        if vp25_positions.shape != (25, 3):
            raise ValueError(f"Expected VP25 shape (25, 3), got {vp25_positions.shape}")

        indices = self.indices[side]
        ref_value = vp25_positions[indices[1, :]] - vp25_positions[indices[0, :]]
        return self.retarget_ref_value(side, ref_value, apply_filter=apply_filter)

    def retarget_ref_value(
        self,
        side: HandSide,
        ref_value: np.ndarray,
        apply_filter: bool = True,
    ) -> BraincoRetargetResult:
        retargeting = self.retargeting[side]
        ref_value = np.asarray(ref_value, dtype=np.float32)

        old_filter = retargeting.filter
        if not apply_filter:
            retargeting.filter = None
        try:
            qpos_full = retargeting.retarget(ref_value)
        finally:
            retargeting.filter = old_filter

        qpos_hardware = qpos_full[self.hardware_order[side]]
        action_01 = normalize_brainco_qpos(qpos_hardware)
        return BraincoRetargetResult(
            qpos_full=np.asarray(qpos_full, dtype=np.float32),
            qpos_hardware=np.asarray(qpos_hardware, dtype=np.float32),
            action_01=action_01,
        )


def normalize_brainco_qpos(qpos_hardware: np.ndarray) -> np.ndarray:
    """Convert BrainCo hardware joint radians to [0, 1] close ratios."""
    qpos_hardware = np.asarray(qpos_hardware, dtype=np.float32)
    if qpos_hardware.shape != (BRAINCO_NUM_MOTORS,):
        raise ValueError(f"Expected qpos shape (6,), got {qpos_hardware.shape}")
    return np.clip(qpos_hardware / BRAINCO_MAX_QPOS, 0.0, 1.0).astype(np.float32)


def brainco_fig6d_to_qpos(fig6d: list[float] | np.ndarray, hand_action: str) -> np.ndarray:
    """Convert exported BrainCo fig6d data to hardware joint radians."""
    qpos = np.asarray(fig6d, dtype=np.float32)
    if qpos.shape != (BRAINCO_NUM_MOTORS,):
        raise ValueError(f"Expected fig6d shape (6,), got {qpos.shape}")
    if hand_action == "normalized":
        qpos = np.clip(qpos, 0.0, 1.0) * BRAINCO_MAX_QPOS
    elif hand_action != "radians":
        raise ValueError(f"Unknown BrainCo hand action format: {hand_action}")
    return qpos.astype(np.float32)


def rotation_matrix_to_rpy_xyz(rotation: np.ndarray) -> np.ndarray:
    """Return XYZ roll/pitch/yaw from a rotation matrix."""
    r = np.asarray(rotation, dtype=np.float64)
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-8
    if not singular:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def rpy_xyz_to_matrix(xyzrpy: list[float] | np.ndarray) -> np.ndarray:
    """Convert an xyzrpy pose or rpy triplet to a rotation matrix."""
    values = np.asarray(xyzrpy, dtype=np.float32)
    if values.shape == (6,):
        roll, pitch, yaw = values[3:6]
    elif values.shape == (3,):
        roll, pitch, yaw = values
    else:
        raise ValueError(f"Expected rpy shape (3,) or xyzrpy shape (6,), got {values.shape}")
    return (_rot_z(float(yaw)) @ _rot_y(float(pitch)) @ _rot_x(float(roll))).astype(np.float32)


def pose_xyzrpy(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(position, dtype=np.float32),
            rotation_matrix_to_rpy_xyz(rotation),
        ]
    ).astype(np.float32)


def pose_from_xyzrpy(xyzrpy: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(xyzrpy, dtype=np.float32)
    if pose.shape != (6,):
        raise ValueError(f"Expected xyzrpy pose shape (6,), got {pose.shape}")
    return pose[:3].copy(), rpy_xyz_to_matrix(pose)


def egodex_to_brainco_wrist_rotation(
    egodex_wrist_rotation: np.ndarray,
    side: HandSide,
    axis_preset: str = "egodex-to-brainco",
) -> np.ndarray:
    """Convert an EgoDex wrist rotation into the BrainCo root frame."""
    egodex_wrist_rotation = np.asarray(egodex_wrist_rotation, dtype=np.float32)
    if axis_preset == "identity":
        return egodex_wrist_rotation
    if axis_preset != "egodex-to-brainco":
        raise ValueError(f"Unknown wrist axis preset: {axis_preset}")
    return egodex_wrist_rotation @ EGODEX_FROM_BRAINCO_WRIST[side]


def egodex_vp25_to_brainco_local(
    vp25_positions: np.ndarray,
    egodex_wrist_position: np.ndarray,
    egodex_wrist_rotation: np.ndarray,
    side: HandSide,
    axis_preset: str = "egodex-to-brainco",
) -> tuple[np.ndarray, np.ndarray]:
    """Express EgoDex VP25 points in the BrainCo base local frame.

    dex-retargeting optimizes robot FK vectors in the robot base frame. For
    HDF5 playback the VP25 points are in the dataset/world frame, so they must
    be rotated back into BrainCo base coordinates before building target
    vectors. Otherwise an open hand can look like a short or reversed target
    vector and the optimizer tends to close the hand.
    """
    vp25_positions = np.asarray(vp25_positions, dtype=np.float32)
    wrist_position = np.asarray(egodex_wrist_position, dtype=np.float32)
    brainco_wrist_rotation = egodex_to_brainco_wrist_rotation(
        egodex_wrist_rotation,
        side,
        axis_preset,
    )
    vp25_local = (vp25_positions - wrist_position[None, :]) @ brainco_wrist_rotation
    return vp25_local.astype(np.float32), brainco_wrist_rotation.astype(np.float32)


def retarget_egodex_brainco_frame(
    h5_file,
    transform_group,
    frame_idx: int,
    side: HandSide,
    retargeter: BraincoRetargeter,
    *,
    min_conf: float = 0.0,
    axis_preset: str = "egodex-to-brainco",
    apply_filter: bool = True,
) -> BraincoFrameRetarget | None:
    """Retarget one EgoDex HDF5 hand frame into BrainCo local targets."""
    from egodex_wuji_common import (
        egodex_vp25_positions,
        extract_position,
        extract_rotation_matrix,
        hand_confidence_ok,
    )

    hand_name = f"{side}Hand"
    if hand_name not in transform_group:
        return None
    if not hand_confidence_ok(h5_file, frame_idx, side, min_conf):
        return None

    wrist_position = extract_position(transform_group[hand_name], frame_idx)
    wrist_rotation = extract_rotation_matrix(transform_group[hand_name], frame_idx)
    if wrist_position is None or wrist_rotation is None:
        return None

    vp25_world = egodex_vp25_positions(h5_file, frame_idx, side)
    vp25_local, brainco_base_rotation = egodex_vp25_to_brainco_local(
        vp25_world,
        wrist_position,
        wrist_rotation,
        side,
        axis_preset,
    )
    result = retargeter.retarget(side, vp25_local, apply_filter=apply_filter)
    return BraincoFrameRetarget(
        side=side,
        wrist_position=np.asarray(wrist_position, dtype=np.float32),
        wrist_rotation=np.asarray(wrist_rotation, dtype=np.float32),
        brainco_base_rotation=brainco_base_rotation,
        vp25_world=vp25_world,
        vp25_local=vp25_local,
        result=result,
    )


def brainco_base_to_wrist_pose(
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    side: HandSide,
    wrist_y_offset: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert BrainCo palm/base pose to the exported wrist pose."""
    base_position = np.asarray(base_position, dtype=np.float32)
    base_rotation = np.asarray(base_rotation, dtype=np.float32)
    offset = BRAINCO_WRIST_OFFSET_LOCAL.copy()
    if wrist_y_offset is not None:
        offset[1] = float(wrist_y_offset)
    wrist_position = base_position + base_rotation @ offset
    wrist_rotation = base_rotation @ BRAINCO_WRIST_FROM_BASE[side]
    return wrist_position.astype(np.float32), wrist_rotation.astype(np.float32)


def brainco_wrist_to_base_pose(
    wrist_position: np.ndarray,
    wrist_rotation: np.ndarray,
    side: HandSide,
    wrist_y_offset: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert stored wrist pose back to the BrainCo palm/base pose for URDF mesh."""
    wrist_position = np.asarray(wrist_position, dtype=np.float32)
    wrist_rotation = np.asarray(wrist_rotation, dtype=np.float32)
    base_rotation = wrist_rotation @ BRAINCO_WRIST_FROM_BASE[side].T
    offset = BRAINCO_WRIST_OFFSET_LOCAL.copy()
    if wrist_y_offset is not None:
        offset[1] = float(wrist_y_offset)
    base_position = wrist_position - base_rotation @ offset
    return base_position.astype(np.float32), base_rotation.astype(np.float32)


def brainco_hand_only_scene_position(position: np.ndarray) -> np.ndarray:
    """Apply viser_brainco_hand_only_viewer.py's fixed display transform."""
    position = np.asarray(position, dtype=np.float32)
    if position.ndim == 1:
        return (BRAINCO_HAND_ONLY_SCENE_ROT @ position + BRAINCO_HAND_ONLY_SCENE_OFFSET).astype(np.float32)
    if position.ndim == 2 and position.shape[1] == 3:
        return (position @ BRAINCO_HAND_ONLY_SCENE_ROT.T + BRAINCO_HAND_ONLY_SCENE_OFFSET[None, :]).astype(np.float32)
    raise ValueError(f"Expected position shape (3,) or (N, 3), got {position.shape}")


def brainco_hand_only_scene_rotation(rotation: np.ndarray) -> np.ndarray:
    """Apply viser_brainco_hand_only_viewer.py's fixed display rotation."""
    return (BRAINCO_HAND_ONLY_SCENE_ROT @ np.asarray(rotation, dtype=np.float32)).astype(np.float32)


def brainco_fk_points(retargeting, qpos_full: np.ndarray, side: HandSide) -> np.ndarray:
    """Return BrainCo FK points for drawing a compact hand skeleton."""
    robot = retargeting.optimizer.robot
    robot.compute_forward_kinematics(qpos_full)
    points = []
    for link_name in BRAINCO_FK_LINKS[side]:
        link_id = robot.get_link_index(link_name)
        points.append(robot.get_link_pose(link_id)[:3, 3])
    return np.asarray(points, dtype=np.float32)


def make_line_segments(points: np.ndarray, edges) -> np.ndarray:
    return np.asarray([[points[a], points[b]] for a, b in edges], dtype=np.float32)


def make_vp25_line_segments(points: np.ndarray) -> np.ndarray:
    return make_line_segments(points, VP25_HAND_EDGES)


def make_brainco_line_segments(points: np.ndarray) -> np.ndarray:
    return make_line_segments(points, BRAINCO_FK_EDGES)


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to viser quaternion order: w, x, y, z."""
    rotation = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(rotation)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-8)
