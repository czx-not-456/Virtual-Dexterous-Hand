from __future__ import annotations
import math
import numpy as np
from src.common import load_yaml
from .kinematics import SerialHandKinematics

class RobotHandModel:
    """CH-M6 native 11-joint software model."""
    def __init__(self, config_path: str = "configs/robot_hand.yaml") -> None:
        cfg = load_yaml(config_path)
        self.cfg = cfg
        self.source_model_path = str(cfg["source_model_path"])
        self.joint_order = list(cfg["joint_order"])
        self.ranges = np.asarray([[math.radians(float(cfg["joint_ranges_deg"][j][0])), math.radians(float(cfg["joint_ranges_deg"][j][1]))] for j in self.joint_order], dtype=float)
        arch = cfg["architecture"]
        self.nominal_dof = int(arch["nominal_dof"])
        self.effective_actuators = int(arch["effective_actuators"])
        self.actuator_order = list(arch["actuator_order"])
        self.finger_joints = {k: tuple(v) for k, v in cfg["finger_joints"].items()}
        self.human_mapping = dict(cfg["human_mapping"])
        self.synergies = dict(cfg.get("synergies", {}))
        self.simulation_names = dict(cfg["simulation_names"])
        self.kin = SerialHandKinematics(cfg["kinematic_chains"], self.joint_order)
        self._joint_index = {name: i for i, name in enumerate(self.joint_order)}

    def fingertips(self, q: np.ndarray) -> dict[str, np.ndarray]:
        return self.kin.fingertips(q)

    def joints_for_finger(self, finger: str) -> tuple[str, ...]:
        return self.finger_joints[finger]

    def map_human_normalized(self, normalized: np.ndarray, human_joint_order: list[str]) -> np.ndarray:
        values = {name: float(v) for name, v in zip(human_joint_order, normalized)}
        out = np.empty(len(self.joint_order), dtype=float)
        for i, joint in enumerate(self.joint_order):
            sources = self.human_mapping[joint]["sources"]
            total = sum(abs(float(w)) for w in sources.values())
            if total <= 1e-12:
                raise ValueError(f"CH-M6 mapping for {joint} has zero total weight")
            u = sum(float(w) * values[str(name)] for name, w in sources.items()) / total
            u = float(np.clip(u, 0.0, 1.0))
            out[i] = self.ranges[i, 0] + u * (self.ranges[i, 1] - self.ranges[i, 0])
        return out

    def actuator_targets_from_joint_targets(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.joint_order),):
            raise ValueError(f"expected {len(self.joint_order)} CH-M6 joint targets, got {q.shape}")
        return np.asarray([q[self._joint_index[name]] for name in self.actuator_order], dtype=float)

    def actuator_ctrl_ranges(self) -> np.ndarray:
        return np.asarray([self.ranges[self._joint_index[name]] for name in self.actuator_order])

    def finger_for_body(self, body_name: str) -> str | None:
        for finger, prefix in self.simulation_names["finger_body_prefixes"].items():
            if body_name.startswith(str(prefix)):
                return str(finger)
        return None

    def tip_site_name(self, finger: str) -> str:
        return str(self.simulation_names["fingertip_sites"][finger])

    def tip_geom_name(self, finger: str) -> str:
        return str(self.simulation_names["fingertip_geoms"][finger])