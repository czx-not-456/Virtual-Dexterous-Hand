from __future__ import annotations

import math
import numpy as np

from src.common import load_yaml
from .kinematics import HandKinematics


class RobotHandModel:
    def __init__(self, config_path: str = "configs/robot_hand.yaml") -> None:
        cfg = load_yaml(config_path)
        self.cfg = cfg
        self.joint_order = cfg["joint_order"]
        self.ranges = np.array([
            [
                math.radians(float(cfg["joint_ranges_deg"][j][0])),
                math.radians(float(cfg["joint_ranges_deg"][j][1])),
            ]
            for j in self.joint_order
        ])
        self.kin = HandKinematics(
            cfg["finger_lengths_m"],
            cfg["base_offsets_m"],
            cfg["base_yaw_deg"],
            self.joint_order,
            cfg.get("base_euler_deg"),
        )
        self.coupling = cfg.get("coupling", {})
        arch = cfg.get("architecture", {})
        self.nominal_dof = int(arch.get("nominal_dof", len(self.joint_order)))
        self.effective_actuators = int(arch.get("effective_actuators", len(self.joint_order)))
        self.actuator_order = list(arch.get("actuator_order", self.joint_order))
        self.anatomical_aliases = dict(cfg.get("anatomical_aliases", {}))
        self.spring_stiffness = dict(cfg.get("non_thumb_spring_stiffness_Nm_rad", {}))

    def fingertips(self, q: np.ndarray) -> dict[str, np.ndarray]:
        return self.kin.fingertips(q)

    def actuator_targets_from_joint_targets(self, q: np.ndarray) -> np.ndarray:
        """把 15 维目标关节姿态投影为 7 维驱动目标。

        文献结构：拇指 3 个关节独立驱动，四根非拇指各由单腱驱动三关节。
        当前投影使用项目配置中的工程协同系数，不代表 CasiaHand 官方腱轮半径。
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.joint_order),):
            raise ValueError(f"expected {len(self.joint_order)} joint targets, got {q.shape}")
        idx = {name: i for i, name in enumerate(self.joint_order)}
        u = [
            float(q[idx["thumb_mcp"]]),
            float(q[idx["thumb_pip"]]),
            float(q[idx["thumb_dip"]]),
        ]
        for finger in ("index", "middle", "ring", "little"):
            c = self.coupling[finger]
            mcp = float(q[idx[f"{finger}_mcp"]])
            pip = float(q[idx[f"{finger}_pip"]])
            dip = float(q[idx[f"{finger}_dip"]])
            # 与 MJCF fixed tendon 的 joint coefficients 保持一致。
            u.append(mcp + float(c["pip_over_mcp"]) * pip + float(c["dip_over_mcp"]) * dip)
        return np.asarray(u, dtype=float)

    def actuator_ctrl_ranges(self) -> np.ndarray:
        """根据 15 维关节限位计算 7 维驱动目标范围。"""
        idx = {name: i for i, name in enumerate(self.joint_order)}
        ranges = [
            self.ranges[idx["thumb_mcp"]],
            self.ranges[idx["thumb_pip"]],
            self.ranges[idx["thumb_dip"]],
        ]
        for finger in ("index", "middle", "ring", "little"):
            c = self.coupling[finger]
            lo = (
                self.ranges[idx[f"{finger}_mcp"], 0]
                + float(c["pip_over_mcp"]) * self.ranges[idx[f"{finger}_pip"], 0]
                + float(c["dip_over_mcp"]) * self.ranges[idx[f"{finger}_dip"], 0]
            )
            hi = (
                self.ranges[idx[f"{finger}_mcp"], 1]
                + float(c["pip_over_mcp"]) * self.ranges[idx[f"{finger}_pip"], 1]
                + float(c["dip_over_mcp"]) * self.ranges[idx[f"{finger}_dip"], 1]
            )
            ranges.append(np.array([lo, hi], dtype=float))
        return np.asarray(ranges, dtype=float)
