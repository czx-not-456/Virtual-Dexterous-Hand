from __future__ import annotations

import math
import numpy as np

from src.common import load_yaml
from src.processing.normalization import normalized_to_angle
from .kinematics import HandKinematics


class HumanHandModel:
    def __init__(self, config_path: str = "configs/human_hand.yaml") -> None:
        cfg = load_yaml(config_path)
        self.joint_order = list(cfg["joint_ranges_deg"].keys())
        self.ranges = np.array([
            [math.radians(float(cfg["joint_ranges_deg"][j][0])), math.radians(float(cfg["joint_ranges_deg"][j][1]))]
            for j in self.joint_order
        ])
        self.kin = HandKinematics(
            cfg["finger_lengths_m"], cfg["base_offsets_m"], cfg["base_yaw_deg"], self.joint_order
        )

    def angles_from_normalized(self, values: dict[str, float]) -> np.ndarray:
        return np.array([
            normalized_to_angle(values[j], self.ranges[i, 0], self.ranges[i, 1])
            for i, j in enumerate(self.joint_order)
        ], dtype=float)

    def normalized_from_angles(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        span = self.ranges[:, 1] - self.ranges[:, 0]
        return np.clip((q - self.ranges[:, 0]) / span, 0.0, 1.0)

    def fingertips(self, q: np.ndarray) -> dict[str, np.ndarray]:
        return self.kin.fingertips(q)
