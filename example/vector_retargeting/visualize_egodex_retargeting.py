"""
Visualize EgoDex hand motion and Inspire hand retargeting side-by-side in Rerun.

Source hand (blue) is shown centered at origin; retargeted robot hand (orange) is
offset +0.3 m in X so both are visible simultaneously. Scrub the "frame" timeline
to compare poses frame-by-frame.

Usage:
    uv run python example/vector_retargeting/visualize_egodex_retargeting.py \
        --hdf5_path /home/unitree/remote_jensen2/egocentric_datasets/EgoDex/data/clean_cups/0.hdf5
"""
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import rerun as rr
import tqdm
import tyro

from dex_retargeting.constants import (
    HandType,
    RobotName,
    RetargetingType,
    get_default_config_path,
    OPERATOR2MANO,
)
from dex_retargeting.retargeting_config import RetargetingConfig

# ---------------------------------------------------------------------------
# Source hand: EgoDex right-hand joint names and skeleton connectivity
# ---------------------------------------------------------------------------

# All joints to extract from EgoDex (25 joints total)
RIGHT_HAND_JOINTS = [
    "rightHand",  # 0: wrist
    "rightThumbKnuckle",            # 1
    "rightThumbIntermediateBase",   # 2
    "rightThumbIntermediateTip",    # 3
    "rightThumbTip",                # 4
    "rightIndexFingerMetacarpal",   # 5
    "rightIndexFingerKnuckle",      # 6
    "rightIndexFingerIntermediateBase",  # 7
    "rightIndexFingerIntermediateTip",   # 8
    "rightIndexFingerTip",          # 9
    "rightMiddleFingerMetacarpal",  # 10
    "rightMiddleFingerKnuckle",     # 11
    "rightMiddleFingerIntermediateBase", # 12
    "rightMiddleFingerIntermediateTip",  # 13
    "rightMiddleFingerTip",         # 14
    "rightRingFingerMetacarpal",    # 15
    "rightRingFingerKnuckle",       # 16
    "rightRingFingerIntermediateBase",   # 17
    "rightRingFingerIntermediateTip",    # 18
    "rightRingFingerTip",           # 19
    "rightLittleFingerMetacarpal",  # 20
    "rightLittleFingerKnuckle",     # 21
    "rightLittleFingerIntermediateBase", # 22
    "rightLittleFingerIntermediateTip",  # 23
    "rightLittleFingerTip",         # 24
]
RIGHT_HAND_JOINT_IDX = {name: i for i, name in enumerate(RIGHT_HAND_JOINTS)}

RIGHT_HAND_BONES = [
    # Thumb
    ("rightHand", "rightThumbKnuckle"),
    ("rightThumbKnuckle", "rightThumbIntermediateBase"),
    ("rightThumbIntermediateBase", "rightThumbIntermediateTip"),
    ("rightThumbIntermediateTip", "rightThumbTip"),
    # Index
    ("rightHand", "rightIndexFingerMetacarpal"),
    ("rightIndexFingerMetacarpal", "rightIndexFingerKnuckle"),
    ("rightIndexFingerKnuckle", "rightIndexFingerIntermediateBase"),
    ("rightIndexFingerIntermediateBase", "rightIndexFingerIntermediateTip"),
    ("rightIndexFingerIntermediateTip", "rightIndexFingerTip"),
    # Middle
    ("rightHand", "rightMiddleFingerMetacarpal"),
    ("rightMiddleFingerMetacarpal", "rightMiddleFingerKnuckle"),
    ("rightMiddleFingerKnuckle", "rightMiddleFingerIntermediateBase"),
    ("rightMiddleFingerIntermediateBase", "rightMiddleFingerIntermediateTip"),
    ("rightMiddleFingerIntermediateTip", "rightMiddleFingerTip"),
    # Ring
    ("rightHand", "rightRingFingerMetacarpal"),
    ("rightRingFingerMetacarpal", "rightRingFingerKnuckle"),
    ("rightRingFingerKnuckle", "rightRingFingerIntermediateBase"),
    ("rightRingFingerIntermediateBase", "rightRingFingerIntermediateTip"),
    ("rightRingFingerIntermediateTip", "rightRingFingerTip"),
    # Little (pinky)
    ("rightHand", "rightLittleFingerMetacarpal"),
    ("rightLittleFingerMetacarpal", "rightLittleFingerKnuckle"),
    ("rightLittleFingerKnuckle", "rightLittleFingerIntermediateBase"),
    ("rightLittleFingerIntermediateBase", "rightLittleFingerIntermediateTip"),
    ("rightLittleFingerIntermediateTip", "rightLittleFingerTip"),
]

LEFT_HAND_JOINTS = [j.replace("right", "left").replace("Right", "Left") for j in RIGHT_HAND_JOINTS]
LEFT_HAND_JOINT_IDX = {name: i for i, name in enumerate(LEFT_HAND_JOINTS)}
LEFT_HAND_BONES = [(p.replace("right", "left").replace("Right", "Left"),
                    c.replace("right", "left").replace("Right", "Left"))
                   for p, c in RIGHT_HAND_BONES]

HAND_JOINTS = {HandType.right: (RIGHT_HAND_JOINTS, RIGHT_HAND_JOINT_IDX, RIGHT_HAND_BONES),
               HandType.left:  (LEFT_HAND_JOINTS, LEFT_HAND_JOINT_IDX, LEFT_HAND_BONES)}

# ---------------------------------------------------------------------------
# Robot hand: Inspire hand link names and skeleton connectivity (from URDF)
# ---------------------------------------------------------------------------

# All URDF links we want to visualize
INSPIRE_RIGHT_LINKS = [
    "base1", "hand_base_link",
    "thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_tip",
]

INSPIRE_RIGHT_BONES = [
    ("base1", "hand_base_link"),
    ("hand_base_link", "thumb_proximal_base"),
    ("thumb_proximal_base", "thumb_proximal"),
    ("thumb_proximal", "thumb_intermediate"),
    ("thumb_intermediate", "thumb_distal"),
    ("thumb_distal", "thumb_tip"),
    ("hand_base_link", "index_proximal"),
    ("index_proximal", "index_intermediate"),
    ("index_intermediate", "index_tip"),
    ("hand_base_link", "middle_proximal"),
    ("middle_proximal", "middle_intermediate"),
    ("middle_intermediate", "middle_tip"),
    ("hand_base_link", "ring_proximal"),
    ("ring_proximal", "ring_intermediate"),
    ("ring_intermediate", "ring_tip"),
    ("hand_base_link", "pinky_proximal"),
    ("pinky_proximal", "pinky_intermediate"),
    ("pinky_intermediate", "pinky_tip"),
]

# Vector retargeting uses wrist (MANO 0) + thumb knuckle (MANO 1) + thumb intermediate (MANO 2) + 5 fingertips (MANO 4,8,12,16,20)
EGODEX_RIGHT_TIP_JOINTS = {
    "rightHand": 0,
    "rightThumbKnuckle": 1,
    "rightThumbIntermediateBase": 2,
    "rightThumbTip": 4,
    "rightIndexFingerTip": 8,
    "rightMiddleFingerTip": 12,
    "rightRingFingerTip": 16,
    "rightLittleFingerTip": 20,
}
EGODEX_LEFT_TIP_JOINTS = {k.replace("right", "left").replace("Right", "Left"): v
                          for k, v in EGODEX_RIGHT_TIP_JOINTS.items()}
EGODEX_TIP_JOINTS = {HandType.right: EGODEX_RIGHT_TIP_JOINTS,
                     HandType.left: EGODEX_LEFT_TIP_JOINTS}

# Rerun colors
COLOR_SOURCE = [100, 180, 255, 255]   # blue
COLOR_ROBOT  = [255, 160, 50,  255]   # orange

ROBOT_X_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def main(
    hdf5_path: str,
    hand_type: HandType = HandType.right,
    robot_name: RobotName = RobotName.inspire,
    rrd_path: Optional[str] = None,
    scaling_factor: Optional[float] = None,
):
    """
    Visualize EgoDex source hand and Vector-retargeted robot hand in Rerun.

    Saves a .rrd file (default: alongside the hdf5 file) then prints the command
    to open it. Pass --rrd_path to override the save location.

    Args:
        hdf5_path: Path to a single EgoDex .hdf5 episode file.
        hand_type: Which hand to visualize (right or left).
        robot_name: Target robot hand (default: inspire).
        rrd_path: Output .rrd path. Defaults to <hdf5_stem>_retargeting.rrd next to the input file.
        scaling_factor: Override the scaling factor from config. If None, uses the config default.
    """
    if rrd_path is None:
        p = Path(hdf5_path)
        rrd_path = str(p.parent / (p.stem + "_retargeting.rrd"))

    # ---- Setup retargeting ------------------------------------------------
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    config_path = get_default_config_path(robot_name, RetargetingType.vector, hand_type)
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    override = {"scaling_factor": scaling_factor} if scaling_factor is not None else None
    retargeting = RetargetingConfig.load_from_file(config_path, override=override).build()
    robot = retargeting.optimizer.robot
    sf = retargeting.optimizer.scaling  # actual scaling factor used

    retarget_indices = retargeting.optimizer.target_link_human_indices
    origin_idx = retarget_indices[0]
    task_idx   = retarget_indices[1]

    # Precompute robot link frame ids
    robot_link_ids = {name: robot.get_link_index(name) for name in INSPIRE_RIGHT_LINKS}

    # ---- Hand joint data --------------------------------------------------
    hand_joints, hand_joint_idx, hand_bones = HAND_JOINTS[hand_type]
    tip_joint_map = EGODEX_TIP_JOINTS[hand_type]
    operator2mano = OPERATOR2MANO[hand_type]
    wrist_key = hand_joints[0]

    # ---- Rerun init: always save to .rrd (avoids blocking spawn) ----------
    rr.init("egodex_retargeting")
    rr.save(rrd_path)  # must be called before logging

    # ---- Process frames ---------------------------------------------------
    with h5py.File(hdf5_path, "r") as f:
        num_frames = f["transforms"][wrist_key].shape[0]

        for frame_idx in tqdm.trange(num_frames, desc="Logging to Rerun"):
            rr.set_time("frame", sequence=frame_idx)

            # Compute wrist inverse transform for this frame
            wrist_T = f["transforms"][wrist_key][frame_idx]          # (4, 4)
            wrist_T_inv = np.linalg.inv(wrist_T)

            # -- Source hand skeleton (wrist-local MANO frame, scaled to robot size) --
            hand_pos = np.array(
                [
                    (wrist_T_inv @ f["transforms"][j][frame_idx])[:3, 3] @ operator2mano
                    for j in hand_joints
                ],
                dtype=np.float32,
            )
            hand_pos *= sf  # scale to robot size so tips align with robot hand

            rr.log(
                "source_hand/joints",
                rr.Points3D(hand_pos, radii=0.005, colors=COLOR_SOURCE),
            )
            bone_strips = [
                [hand_pos[hand_joint_idx[p]], hand_pos[hand_joint_idx[c]]]
                for p, c in hand_bones
            ]
            rr.log(
                "source_hand/bones",
                rr.LineStrips3D(bone_strips, radii=0.002, colors=COLOR_SOURCE),
            )

            # -- Retargeting (using frame-normalized positions) --
            tip_pos = np.zeros((21, 3), dtype=np.float32)
            for joint_name, mano_idx in tip_joint_map.items():
                joint_T = f["transforms"][joint_name][frame_idx]
                tip_pos[mano_idx] = (wrist_T_inv @ joint_T)[:3, 3] @ operator2mano
            ref_value = tip_pos[task_idx] - tip_pos[origin_idx]
            robot_qpos = retargeting.retarget(ref_value)

            # -- Robot hand skeleton: re-root at `base1` so the optimizer's free-joint
            #    pose + URDF-root (`base1`) offset are factored out, putting the robot
            #    in the same wrist-local / MANO-aligned frame as the source hand.
            #    Then apply ROBOT_X_OFFSET so the two skeletons render side-by-side.
            robot.compute_forward_kinematics(robot_qpos)
            base_pos = robot.get_link_pose(robot_link_ids["base1"])[:3, 3].astype(np.float32)
            link_pos = {
                name: (robot.get_link_pose(link_id)[:3, 3].astype(np.float32)
                       - base_pos + ROBOT_X_OFFSET)
                for name, link_id in robot_link_ids.items()
            }
            # breakpoint()

            rr.log(
                "robot_hand/joints",
                rr.Points3D(list(link_pos.values()), radii=0.005, colors=COLOR_ROBOT),
            )
            robot_strips = [
                [link_pos[p], link_pos[c]]
                for p, c in INSPIRE_RIGHT_BONES
                if p in link_pos and c in link_pos
            ]
            rr.log(
                "robot_hand/bones",
                rr.LineStrips3D(robot_strips, radii=0.002, colors=COLOR_ROBOT),
            )

    print(f"\nSaved: {rrd_path}")
    print(f"View:  rerun {rrd_path}")


if __name__ == "__main__":
    tyro.cli(main)
