"""Optional six-control command adapter from the teacher's CH-M6 mapping.

This is a command projection only. It does not change the MJCF actuators or
claim that the physical hand has six motors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.hand_model.robot_hand import RobotHandModel


class TeacherCHM6ControlAdapter:
    def __init__(self, robot: RobotHandModel, mapping_path: str | Path) -> None:
        self.robot = robot
        mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))["CH-M6"]
        if mapping.get("mapping_mode") != "shared_6d_then_qn_coupling":
            raise ValueError("unsupported teacher CH-M6 mapping mode")
        self.active_order = tuple(mapping["active_joint_order"])
        couplings = mapping["passive_joint_couplings"]
        if len(self.active_order) != 6 or len(couplings) != 5:
            raise ValueError("expected six controls and five passive couplings")
        joint_set = set(robot.joint_order)
        if len(set(self.active_order)) != 6 or not set(self.active_order) <= joint_set:
            raise ValueError("invalid active joint order")
        self.couplings = tuple((item["driver"], item["child"], float(item["multiplier"])) for item in couplings)
        if set(child for _, child, _ in self.couplings) != joint_set - set(self.active_order):
            raise ValueError("passive joint set does not match robot model")
        if any(driver not in self.active_order or not np.isfinite(gain) for driver, _, gain in self.couplings):
            raise ValueError("invalid passive coupling")
        self.index = {name: i for i, name in enumerate(robot.joint_order)}

    def expand_degrees(self, controls_deg: np.ndarray) -> np.ndarray:
        controls = np.asarray(controls_deg, dtype=float)
        if controls.shape != (6,) or not np.all(np.isfinite(controls)):
            raise ValueError("CH-M6 teacher control must be six finite degree values")
        result = np.zeros(len(self.robot.joint_order), dtype=float)
        for name, value in zip(self.active_order, controls):
            result[self.index[name]] = np.deg2rad(value)
        for driver, child, gain in self.couplings:
            result[self.index[child]] = result[self.index[driver]] * gain
        return np.clip(result, self.robot.ranges[:, 0], self.robot.ranges[:, 1])
