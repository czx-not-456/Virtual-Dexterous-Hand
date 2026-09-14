from __future__ import annotations

import numpy as np
from src.hand_model.robot_hand import RobotHandModel


def coupling_error(q: np.ndarray, robot: RobotHandModel) -> float:
    """名义协同误差。

    对真实欠驱动手而言，单腱 + 弹簧允许接触后产生被动顺应；因此本项只作为
    运动学映射的软约束，不把三关节锁成刚性固定比例。
    """
    idx = {name: i for i, name in enumerate(robot.joint_order)}
    err = 0.0
    for finger, cfg in robot.coupling.items():
        mcp = q[idx[f"{finger}_mcp"]]
        pip = q[idx[f"{finger}_pip"]]
        dip = q[idx[f"{finger}_dip"]]
        err += float((pip - cfg["pip_over_mcp"] * mcp) ** 2)
        err += float((dip - cfg["dip_over_mcp"] * mcp) ** 2)
    return err


def fingertip_collision_penalty(
    q: np.ndarray,
    robot: RobotHandModel,
    intent: str,
    adjacent_tip_min_m: float = 0.010,
    non_target_thumb_tip_min_m: float = 0.014,
    penalty_gain: float = 1.0,
) -> float:
    """简化的指尖自碰撞安全项。

    参考混合关节-笛卡尔映射中“精细捏合要允许拇指/食指接近，同时保持其他
    指间安全距离”的思想。当前工程没有使用完整 mesh 距离场，因此这是低成本
    指尖近似，不等同于论文中的完整自碰撞模型。
    """
    tips = robot.fingertips(q)
    pairs = [
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "little"),
    ]
    penalty = 0.0
    for a, b in pairs:
        d = float(np.linalg.norm(tips[a] - tips[b]))
        if d < adjacent_tip_min_m:
            penalty += (adjacent_tip_min_m - d) ** 2

    # 精细捏合允许 thumb-index 接近；其余拇指-非目标手指仍需留安全距离。
    thumb_targets = ("middle", "ring", "little") if intent == "PINCH" else ("index", "middle", "ring", "little")
    for f in thumb_targets:
        d = float(np.linalg.norm(tips["thumb"] - tips[f]))
        if d < non_target_thumb_tip_min_m:
            penalty += (non_target_thumb_tip_min_m - d) ** 2
    return float(penalty_gain) * penalty


def vector_shape_error(
    robot_tips: dict[str, np.ndarray],
    human_tips: dict[str, np.ndarray],
    scale: float,
    vector_pairs: list[list[str]] | list[tuple[str, str]],
) -> float:
    """关键指尖相对向量误差。

    对应文献中以关键点间向量同时约束“形状”和“精细末端关系”的思路。
    使用相对向量而非单纯绝对位置，可降低手掌基准偏差对映射的影响。
    """
    errs: list[float] = []
    for pair in vector_pairs:
        a, b = str(pair[0]), str(pair[1])
        vr = np.asarray(robot_tips[b]) - np.asarray(robot_tips[a])
        vh = (np.asarray(human_tips[b]) - np.asarray(human_tips[a])) * float(scale)
        errs.append(float(np.sum((vr - vh) ** 2)))
    return float(np.mean(errs)) if errs else 0.0
