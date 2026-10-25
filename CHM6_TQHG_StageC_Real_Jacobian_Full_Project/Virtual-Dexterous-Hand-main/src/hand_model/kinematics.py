from __future__ import annotations
from dataclasses import dataclass
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")

def _quat_matrix_wxyz(quat: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=float)
    n = float(np.linalg.norm([w, x, y, z]))
    if n <= 1e-12:
        return np.eye(3)
    w, x, y, z = (np.asarray([w, x, y, z]) / n).tolist()
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=float)

def _axis_angle(axis: list[float], angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a /= max(float(np.linalg.norm(a)), 1e-12)
    x, y, z = a
    c, s, C = np.cos(angle), np.sin(angle), 1.0-np.cos(angle)
    return np.array([
        [c+x*x*C, x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, c+z*z*C],
    ], dtype=float)

@dataclass
class SerialHandKinematics:
    """Config-driven forward kinematics matching the CH-M6 MJCF tree."""
    chains: dict
    joint_order: list[str]

    @property
    def finger_lengths(self) -> dict[str, list[float]]:
        out = {}
        for finger, chain in self.chains.items():
            values = [float(np.linalg.norm(j["child_offset_m"])) for j in chain["joints"][:-1]]
            values.append(float(np.linalg.norm(chain["tip_local_m"])))
            out[finger] = values
        return out

    def finger_points(self, q: np.ndarray) -> dict[str, list[np.ndarray]]:
        qmap = {name: float(v) for name, v in zip(self.joint_order, np.asarray(q, dtype=float))}
        out = {}
        for finger in FINGERS:
            chain = self.chains[finger]
            p = np.asarray(chain["base_pos_m"], dtype=float)
            R = _quat_matrix_wxyz(chain.get("base_quat_wxyz", [1, 0, 0, 0]))
            points = [p.copy()]
            for joint in chain["joints"]:
                R = R @ _axis_angle(joint["axis"], qmap[joint["name"]])
                offset = np.asarray(joint.get("child_offset_m", [0, 0, 0]), dtype=float)
                if float(np.linalg.norm(offset)) > 0:
                    p = p + R @ offset
                    points.append(p.copy())
            p = p + R @ np.asarray(chain["tip_local_m"], dtype=float)
            points.append(p.copy())
            out[finger] = points
        return out

    def fingertips(self, q: np.ndarray) -> dict[str, np.ndarray]:
        return {finger: points[-1].copy() for finger, points in self.finger_points(q).items()}

    def keypoints(self, q: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for finger, points in self.finger_points(q).items():
            out[f"{finger}_base"] = points[0].copy()
            for i, point in enumerate(points[1:-1], 1):
                out[f"{finger}_j{i}"] = point.copy()
            out[f"{finger}_tip"] = points[-1].copy()
        return out
@dataclass
class HumanHandKinematics:
    """Original 15-channel human-hand feature model retained for glove input."""
    finger_lengths: dict[str, list[float]]
    base_offsets: dict[str, list[float]]
    base_yaw_deg: dict[str, float]
    joint_order: list[str]

    def finger_points(self, q: np.ndarray) -> dict[str, list[np.ndarray]]:
        import math
        qmap = {name: float(v) for name, v in zip(self.joint_order, np.asarray(q, dtype=float))}
        out = {}
        for finger in FINGERS:
            angles = [qmap[f"{finger}_mcp"], qmap[f"{finger}_pip"], qmap[f"{finger}_dip"]]
            yaw = math.radians(float(self.base_yaw_deg[finger]))
            direction = np.array([math.sin(yaw), math.cos(yaw), 0.0])
            p = np.asarray(self.base_offsets[finger], dtype=float)
            points = [p.copy()]
            cumulative = 0.0
            for length, angle in zip(self.finger_lengths[finger], angles):
                cumulative += angle
                p = p + direction * (float(length) * math.cos(cumulative)) + np.array([0.0, 0.0, -float(length) * math.sin(cumulative)])
                points.append(p.copy())
            out[finger] = points
        return out

    def fingertips(self, q: np.ndarray) -> dict[str, np.ndarray]:
        return {finger: points[-1].copy() for finger, points in self.finger_points(q).items()}