# 可视化 EgoDex 手部数据重定向到 Inspire RH56E2 灵巧手的结果（左右手同时渲染）。
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import tempfile

import h5py
import numpy as np
import rerun as rr
import tqdm
import tyro

from dex_retargeting.constants import HandType, OPERATOR2MANO
from dex_retargeting.retargeting_config import RetargetingConfig

# ---------------------------------------------------------------------------
# EgoDex 手部关节与骨架连接
# ---------------------------------------------------------------------------

RIGHT_HAND_JOINTS = [
    "rightHand",
    "rightThumbKnuckle",
    "rightThumbIntermediateBase",
    "rightThumbIntermediateTip",
    "rightThumbTip",
    "rightIndexFingerMetacarpal",
    "rightIndexFingerKnuckle",
    "rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip",
    "rightIndexFingerTip",
    "rightMiddleFingerMetacarpal",
    "rightMiddleFingerKnuckle",
    "rightMiddleFingerIntermediateBase",
    "rightMiddleFingerIntermediateTip",
    "rightMiddleFingerTip",
    "rightRingFingerMetacarpal",
    "rightRingFingerKnuckle",
    "rightRingFingerIntermediateBase",
    "rightRingFingerIntermediateTip",
    "rightRingFingerTip",
    "rightLittleFingerMetacarpal",
    "rightLittleFingerKnuckle",
    "rightLittleFingerIntermediateBase",
    "rightLittleFingerIntermediateTip",
    "rightLittleFingerTip",
]
RIGHT_HAND_JOINT_IDX = {name: i for i, name in enumerate(RIGHT_HAND_JOINTS)}

RIGHT_HAND_BONES = [
    ("rightHand", "rightThumbKnuckle"),
    ("rightThumbKnuckle", "rightThumbIntermediateBase"),
    ("rightThumbIntermediateBase", "rightThumbIntermediateTip"),
    ("rightThumbIntermediateTip", "rightThumbTip"),
    ("rightHand", "rightIndexFingerMetacarpal"),
    ("rightIndexFingerMetacarpal", "rightIndexFingerKnuckle"),
    ("rightIndexFingerKnuckle", "rightIndexFingerIntermediateBase"),
    ("rightIndexFingerIntermediateBase", "rightIndexFingerIntermediateTip"),
    ("rightIndexFingerIntermediateTip", "rightIndexFingerTip"),
    ("rightHand", "rightMiddleFingerMetacarpal"),
    ("rightMiddleFingerMetacarpal", "rightMiddleFingerKnuckle"),
    ("rightMiddleFingerKnuckle", "rightMiddleFingerIntermediateBase"),
    ("rightMiddleFingerIntermediateBase", "rightMiddleFingerIntermediateTip"),
    ("rightMiddleFingerIntermediateTip", "rightMiddleFingerTip"),
    ("rightHand", "rightRingFingerMetacarpal"),
    ("rightRingFingerMetacarpal", "rightRingFingerKnuckle"),
    ("rightRingFingerKnuckle", "rightRingFingerIntermediateBase"),
    ("rightRingFingerIntermediateBase", "rightRingFingerIntermediateTip"),
    ("rightRingFingerIntermediateTip", "rightRingFingerTip"),
    ("rightHand", "rightLittleFingerMetacarpal"),
    ("rightLittleFingerMetacarpal", "rightLittleFingerKnuckle"),
    ("rightLittleFingerKnuckle", "rightLittleFingerIntermediateBase"),
    ("rightLittleFingerIntermediateBase", "rightLittleFingerIntermediateTip"),
    ("rightLittleFingerIntermediateTip", "rightLittleFingerTip"),
]

LEFT_HAND_JOINTS = [
    joint.replace("right", "left").replace("Right", "Left")
    for joint in RIGHT_HAND_JOINTS
]
LEFT_HAND_JOINT_IDX = {name: i for i, name in enumerate(LEFT_HAND_JOINTS)}
LEFT_HAND_BONES = [
    (
        parent.replace("right", "left").replace("Right", "Left"),
        child.replace("right", "left").replace("Right", "Left"),
    )
    for parent, child in RIGHT_HAND_BONES
]

HAND_JOINTS = {
    HandType.right: (RIGHT_HAND_JOINTS, RIGHT_HAND_JOINT_IDX, RIGHT_HAND_BONES),
    HandType.left: (LEFT_HAND_JOINTS, LEFT_HAND_JOINT_IDX, LEFT_HAND_BONES),
}

# EgoDex MANO 关节索引映射（用于提取指尖位置）
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
EGODEX_LEFT_TIP_JOINTS = {
    joint.replace("right", "left").replace("Right", "Left"): mano_idx
    for joint, mano_idx in EGODEX_RIGHT_TIP_JOINTS.items()
}
EGODEX_TIP_JOINTS = {
    HandType.right: EGODEX_RIGHT_TIP_JOINTS,
    HandType.left: EGODEX_LEFT_TIP_JOINTS,
}

# ---------------------------------------------------------------------------
# RH56E2 机器人关节与可视化骨架
# ---------------------------------------------------------------------------

RH56E2_URDF_RELATIVE_PATHS = {
    HandType.right: (
        "inspire_RH56E2/RH56E2_R_2025_9_11/urdf/RH56E2_R_2025_9_11.urdf"
    ),
    HandType.left: (
        "inspire_RH56E2/RH56E2_L_2025_9_10/urdf/RH56E2_L_2025_9_10.urdf"
    ),
}

# LeRobot 3.0 输出时的 6D 语义顺序：
# [thumb_oc, thumb_lat, index, middle, ring, little]
RH56E2_JOINT_LABELS = [
    "thumb_oc",
    "thumb_lat",
    "index",
    "middle",
    "ring",
    "little",
]

RH56E2_DEFAULT_SCALING_FACTOR = 1.0
RH56E2_THUMB_4_MIMIC_MULTIPLIER = 0.8392 * 0.891
RH56E2_RETARGET_ROOT_LINK = "base1"
RH56E2_DEFAULT_TIP_ORIGIN_SCALE = 0.25

# 渲染颜色：右手蓝/橙，左手紫/青
COLOR_SOURCE = [100, 180, 255, 255]
COLOR_ROBOT = [255, 160, 50, 255]
COLOR_SOURCE_LEFT = [200, 100, 255, 255]
COLOR_ROBOT_LEFT = [80, 220, 160, 255]
DEFAULT_ROBOT_WRIST_Z_OFFSET = 0.0


def _prefix(hand_type: HandType) -> str:
    return hand_type.name


def _default_retarget_origin_link(hand_type: HandType) -> str:
    return f"{_prefix(hand_type)}_plam_1"


# Legacy 9D 关节顺序（用于 retargeting optimizer）
def _rh56e2_joint_names(hand_type: HandType) -> List[str]:
    prefix = _prefix(hand_type)
    return [
        "dummy_x_rotation_joint",
        "dummy_y_rotation_joint",
        "dummy_z_rotation_joint",
        f"{prefix}_little_1_joint",
        f"{prefix}_ring_1_joint",
        f"{prefix}_middle_1_joint",
        f"{prefix}_index_1_joint",
        f"{prefix}_thumb_2_joint",
        f"{prefix}_thumb_1_joint",
    ]


# LeRobot 6D 关节顺序（最终输出顺序）
def _rh56e2_lerobot_joint_names(hand_type: HandType) -> List[str]:
    prefix = _prefix(hand_type)
    return [
        f"{prefix}_thumb_2_joint",
        f"{prefix}_thumb_1_joint",
        f"{prefix}_index_1_joint",
        f"{prefix}_middle_1_joint",
        f"{prefix}_ring_1_joint",
        f"{prefix}_little_1_joint",
    ]


def _rh56e2_task_link_names(hand_type: HandType) -> List[str]:
    prefix = _prefix(hand_type)
    return [
        f"{prefix}_thumb_tip",
        f"{prefix}_index_tip",
        f"{prefix}_middle_tip",
        f"{prefix}_ring_tip",
        f"{prefix}_little_tip",
        f"{prefix}_thumb_1",
        f"{prefix}_thumb_2",
    ]


def _rh56e2_visual_links(hand_type: HandType) -> List[str]:
    prefix = _prefix(hand_type)
    return [
        RH56E2_RETARGET_ROOT_LINK,
        "base_link",
        f"{prefix}_plam_1",
        f"{prefix}_plam_2",
        f"{prefix}_thumb_1",
        f"{prefix}_thumb_2",
        f"{prefix}_thumb_3",
        f"{prefix}_thumb_4",
        f"{prefix}_thumb_tip",
        f"{prefix}_index_1",
        f"{prefix}_index_2",
        f"{prefix}_index_tip",
        f"{prefix}_middle_1",
        f"{prefix}_middle_2",
        f"{prefix}_middle_tip",
        f"{prefix}_ring_1",
        f"{prefix}_ring_2",
        f"{prefix}_ring_tip",
        f"{prefix}_little_1",
        f"{prefix}_little_2",
        f"{prefix}_little_tip",
    ]


def _rh56e2_visual_bones(hand_type: HandType) -> List[Tuple[str, str]]:
    prefix = _prefix(hand_type)
    return [
        (RH56E2_RETARGET_ROOT_LINK, "base_link"),
        ("base_link", f"{prefix}_plam_1"),
        (f"{prefix}_plam_1", f"{prefix}_plam_2"),
        ("base_link", f"{prefix}_thumb_1"),
        (f"{prefix}_thumb_1", f"{prefix}_thumb_2"),
        (f"{prefix}_thumb_2", f"{prefix}_thumb_3"),
        (f"{prefix}_thumb_3", f"{prefix}_thumb_4"),
        (f"{prefix}_thumb_4", f"{prefix}_thumb_tip"),
        ("base_link", f"{prefix}_index_1"),
        (f"{prefix}_index_1", f"{prefix}_index_2"),
        (f"{prefix}_index_2", f"{prefix}_index_tip"),
        ("base_link", f"{prefix}_middle_1"),
        (f"{prefix}_middle_1", f"{prefix}_middle_2"),
        (f"{prefix}_middle_2", f"{prefix}_middle_tip"),
        ("base_link", f"{prefix}_ring_1"),
        (f"{prefix}_ring_1", f"{prefix}_ring_2"),
        (f"{prefix}_ring_2", f"{prefix}_ring_tip"),
        ("base_link", f"{prefix}_little_1"),
        (f"{prefix}_little_1", f"{prefix}_little_2"),
        (f"{prefix}_little_2", f"{prefix}_little_tip"),
    ]


def _joint_child_origin(urdf_text: str, joint_name: str) -> Tuple[str, str, str]:
    """从 URDF 文本中解析指定关节的 parent link 和 origin xyz/rpy。"""
    pattern = rf'<joint\s+name="{re.escape(joint_name)}"[\s\S]*?</joint>'
    match = re.search(pattern, urdf_text)
    if match is None:
        raise ValueError(f"找不到 RH56E2 关节: {joint_name}")
    block = match.group(0)
    parent = re.search(r'<parent\s+link="([^"]+)"\s*/>', block)
    origin = re.search(r'<origin\s+xyz="([^"]+)"\s+rpy="([^"]+)"\s*/>', block)
    if parent is None or origin is None:
        raise ValueError(f"无法解析 RH56E2 关节 parent/origin: {joint_name}")
    return parent.group(1), origin.group(1), origin.group(2)


def _scale_xyz(xyz: str, scale: float) -> str:
    """对 xyz 字符串按比例缩放。"""
    values = np.fromstring(xyz, sep=" ")
    if values.shape != (3,):
        raise ValueError(f"无法解析 xyz: {xyz}")
    return " ".join(f"{value:.8g}" for value in values * scale)


def _rh56e2_virtual_tip_xml(
    urdf_text: str,
    hand_type: HandType,
    tip_origin_scale: float,
) -> str:
    """为每个手指生成虚拟 tip link，位置由 force sensor 关节的 origin 按比例缩放得到。"""
    prefix = _prefix(hand_type)
    sensor_joint_map = {
        "thumb": f"{prefix}_thumb_force_sensor_4_joint",
        "index": f"{prefix}_index_force_sensor_3_joint",
        "middle": f"{prefix}_middle_force_sensor_3_joint",
        "ring": f"{prefix}_ring_force_sensor_3_joint",
        "little": f"{prefix}_little_force_sensor_3_joint",
    }
    blocks = []
    for finger_name, sensor_joint_name in sensor_joint_map.items():
        parent_link, xyz, rpy = _joint_child_origin(urdf_text, sensor_joint_name)
        tip_xyz = _scale_xyz(xyz, tip_origin_scale)
        tip_link = f"{prefix}_{finger_name}_tip"
        blocks.append(
            f"""
  <link name="{tip_link}"/>
  <joint name="{tip_link}_joint" type="fixed">
    <origin xyz="{tip_xyz}" rpy="{rpy}"/>
    <parent link="{parent_link}"/>
    <child link="{tip_link}"/>
  </joint>"""
        )
    return "\n".join(blocks)


def _rh56e2_retarget_root_xml() -> str:
    """生成 retargeting root link，robot_wrist_z_offset 已在 URDF 层面处理。"""
    return f"""
  <link name="{RH56E2_RETARGET_ROOT_LINK}"/>
  <joint name="{RH56E2_RETARGET_ROOT_LINK}_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="{RH56E2_RETARGET_ROOT_LINK}"/>
    <child link="base_link"/>
  </joint>"""


def _write_rh56e2_retarget_urdf(
    urdf_path: Path,
    hand_type: HandType,
    robot_wrist_z_offset: float,
    tip_origin_scale: float,
) -> Path:
    """写出临时 URDF，补齐 root/tip 语义点并压平拇指 mimic。"""
    prefix = _prefix(hand_type)
    urdf_text = urdf_path.read_text(encoding="utf-8")
    # 合并 thumb_3 + thumb_4 的 mimic 为一个等效 multiplier
    urdf_text = urdf_text.replace(
        f'joint="{prefix}_thumb_3_joint"\n        multiplier="0.891"',
        (
            f'joint="{prefix}_thumb_2_joint"\n'
            f'        multiplier="{RH56E2_THUMB_4_MIMIC_MULTIPLIER:.7f}"'
        ),
    )
    retarget_links_xml = "\n".join(
        [
            _rh56e2_retarget_root_xml(),
            _rh56e2_virtual_tip_xml(urdf_text, hand_type, tip_origin_scale),
        ]
    )
    urdf_text = urdf_text.replace("</robot>", f"{retarget_links_xml}\n</robot>")

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".urdf",
        prefix=f"{urdf_path.stem}_retarget_",
        dir=urdf_path.parent,
        delete=False,
    )
    with tmp:
        tmp.write(urdf_text)
    return Path(tmp.name)


def _build_rh56e2_retargeting(
    robot_dir: Path,
    hand_type: HandType,
    scaling_factor: Optional[float],
    robot_wrist_z_offset: float,
    retarget_origin_link: Optional[str],
    tip_origin_scale: float,
    debug_urdf_path: Optional[str],
):
    """构建 RH56E2 vector retargeting 对象，返回配置好的 Retargeting 实例。"""
    urdf_path = robot_dir / RH56E2_URDF_RELATIVE_PATHS[hand_type]
    patched_urdf_path = _write_rh56e2_retarget_urdf(
        urdf_path,
        hand_type,
        robot_wrist_z_offset,
        tip_origin_scale,
    )
    if debug_urdf_path is not None:
        debug_path = Path(debug_urdf_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(patched_urdf_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Debug retarget URDF: {debug_path}")

    scale = (
        RH56E2_DEFAULT_SCALING_FACTOR
        if scaling_factor is None
        else float(scaling_factor)
    )
    config_dict: Dict = {
        "type": "vector",
        "urdf_path": str(patched_urdf_path),
        "add_dummy_free_joint": True,
        "target_joint_names": _rh56e2_joint_names(hand_type),
        "target_origin_link_names": [
            retarget_origin_link or _default_retarget_origin_link(hand_type)
        ]
        * 7,
        "target_task_link_names": _rh56e2_task_link_names(hand_type),
        # target_link_human_indices: [[origin_idx]*7, [task_idx]*7]
        # origin: wrist(0), task: thumb_tip(4), index_tip(8), middle_tip(12), ring_tip(16), little_tip(20), thumb_kuckle(1), thumb_base(2)
        "target_link_human_indices": [[0, 0, 0, 0, 0, 0, 0], [4, 8, 12, 16, 20, 1, 2]],
        "scaling_factor": scale,
        "low_pass_alpha": 0.2,
    }

    try:
        return RetargetingConfig.from_dict(config_dict).build()
    finally:
        patched_urdf_path.unlink(missing_ok=True)


def _check_rerun_sdk() -> None:
    """检查 rerun SDK 是否为 Rerun SDK（而非已废弃的 tartley/rerun）。"""
    if not all(hasattr(rr, name) for name in ("init", "save", "log")):
        raise ImportError(
            "当前导入的 rerun 不是 Rerun SDK，请在项目 .venv 中运行，"
            "例如：.venv/bin/python example/vector_retargeting/"
            "visualize_egodex_retargeting_rh56e2.py"
        )


def main(
    hdf5_path: str = "/home/user/ml-egodex/test/clean_cups/0.hdf5",
    hand_type: HandType = HandType.right,
    rrd_path: Optional[str] = None,
    scaling_factor: Optional[float] = None,
    robot_wrist_z_offset: float = DEFAULT_ROBOT_WRIST_Z_OFFSET,
    retarget_origin_link: Optional[str] = None,
    tip_origin_scale: float = RH56E2_DEFAULT_TIP_ORIGIN_SCALE,
    max_frames: Optional[int] = None,
    debug_urdf_path: Optional[str] = None,
):
    """
    在 Rerun 中可视化 EgoDex 源手和重定向后的 RH56E2 灵巧手。

    Args:
        hdf5_path: 单个 EgoDex .hdf5 episode 文件路径。
        hand_type: 处理右手或左手。
        rrd_path: 输出 .rrd 文件路径；默认写到输入 hdf5 同目录。
        scaling_factor: 覆盖 RH56E2 vector retargeting 缩放系数。
        robot_wrist_z_offset: 显示时给 RH56E2 增加的 z 偏移，用于调节和人手的重合效果。
        retarget_origin_link: vector retargeting 的机器人 origin link；默认使用当前手的 plam_1。
        tip_origin_scale: 虚拟 tip 相对 force sensor 固定关节 origin 的缩放，默认 0.25 以折中贴合和弯曲幅度。
        max_frames: 只处理前 N 帧，便于快速检查。
        debug_urdf_path: 可选，保存补齐 root/tip 后的临时 URDF，方便放进 URDF viewer 检查。
    """
    _check_rerun_sdk()

    if rrd_path is None:
        hdf5_file = Path(hdf5_path)
        rrd_path = str(hdf5_file.parent / f"{hdf5_file.stem}_retargeting_rh56e2.rrd")

    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    retargeting = _build_rh56e2_retargeting(
        robot_dir,
        hand_type,
        scaling_factor,
        robot_wrist_z_offset,
        retarget_origin_link,
        tip_origin_scale,
        debug_urdf_path,
    )
    robot = retargeting.optimizer.robot
    scale = retargeting.optimizer.scaling

    retarget_indices = retargeting.optimizer.target_link_human_indices
    origin_idx = retarget_indices[0]
    task_idx = retarget_indices[1]

    robot_links = _rh56e2_visual_links(hand_type)
    robot_bones = _rh56e2_visual_bones(hand_type)
    robot_link_ids = {name: robot.get_link_index(name) for name in robot_links}

    hand_joints, hand_joint_idx, hand_bones = HAND_JOINTS[hand_type]
    tip_joint_map = EGODEX_TIP_JOINTS[hand_type]
    operator2mano = OPERATOR2MANO[hand_type]
    wrist_key = hand_joints[0]
    actual_origin_link = retargeting.optimizer.origin_link_names[0]
    robot_display_offset = np.array(
        [0.0, 0.0, robot_wrist_z_offset], dtype=np.float32
    )

    print("Legacy-style optimizer joints:", _rh56e2_joint_names(hand_type))
    print(
        "LeRobot 6D joint order:",
        list(zip(RH56E2_JOINT_LABELS, _rh56e2_lerobot_joint_names(hand_type))),
    )
    print(f"Scaling factor: {scale}")
    print(f"Retarget origin link: {actual_origin_link}")
    print(f"Tip origin scale: {tip_origin_scale}")
    print(f"Robot wrist z offset: {robot_wrist_z_offset}")

    rr.init("egodex_retargeting_rh56e2")
    rr.save(rrd_path)

    with h5py.File(hdf5_path, "r") as h5_file:
        num_frames = h5_file["transforms"][wrist_key].shape[0]
        if max_frames is not None:
            num_frames = min(num_frames, max_frames)

        for frame_idx in tqdm.trange(num_frames, desc="Logging RH56E2 to Rerun"):
            rr.set_time("frame", sequence=frame_idx)

            wrist_transform = h5_file["transforms"][wrist_key][frame_idx]
            wrist_transform_inv = np.linalg.inv(wrist_transform)

            # 渲染源手骨架
            hand_pos = np.array(
                [
                    (
                        wrist_transform_inv
                        @ h5_file["transforms"][joint_name][frame_idx]
                    )[:3, 3]
                    @ operator2mano
                    for joint_name in hand_joints
                ],
                dtype=np.float32,
            )
            hand_pos *= scale

            rr.log(
                "source_hand/joints",
                rr.Points3D(hand_pos, radii=0.005, colors=COLOR_SOURCE),
            )
            hand_strips = [
                [hand_pos[hand_joint_idx[parent]], hand_pos[hand_joint_idx[child]]]
                for parent, child in hand_bones
            ]
            rr.log(
                "source_hand/bones",
                rr.LineStrips3D(hand_strips, radii=0.002, colors=COLOR_SOURCE),
            )

            # 计算指尖位置，执行重定向，渲染机器人的手骨架
            tip_pos = np.zeros((21, 3), dtype=np.float32)
            for joint_name, mano_idx in tip_joint_map.items():
                joint_transform = h5_file["transforms"][joint_name][frame_idx]
                tip_pos[mano_idx] = (
                    (wrist_transform_inv @ joint_transform)[:3, 3] @ operator2mano
                )
            ref_value = tip_pos[task_idx] - tip_pos[origin_idx]
            robot_qpos = retargeting.retarget(ref_value)

            robot.compute_forward_kinematics(robot_qpos)
            base_pos = robot.get_link_pose(robot_link_ids[actual_origin_link])[
                :3, 3
            ].astype(np.float32)
            link_pos = {
                name: (
                    robot.get_link_pose(link_id)[:3, 3].astype(np.float32)
                    - base_pos
                    + robot_display_offset
                )
                for name, link_id in robot_link_ids.items()
            }

            rr.log(
                "robot_hand/joints",
                rr.Points3D(list(link_pos.values()), radii=0.005, colors=COLOR_ROBOT),
            )
            robot_strips = [
                [link_pos[parent], link_pos[child]]
                for parent, child in robot_bones
                if parent in link_pos and child in link_pos
            ]
            rr.log(
                "robot_hand/bones",
                rr.LineStrips3D(robot_strips, radii=0.002, colors=COLOR_ROBOT),
            )

    print(f"\nSaved: {rrd_path}")
    print(f"View:  rerun {rrd_path}")


if __name__ == "__main__":
    tyro.cli(main)
