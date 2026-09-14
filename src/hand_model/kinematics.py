from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "little")


@dataclass
class HandKinematics:
    finger_lengths: dict[str, list[float]]
    base_offsets: dict[str, list[float]]
    base_yaw_deg: dict[str, float]
    joint_order: list[str]
    base_euler_deg: dict[str, list[float]] | None = None

    def _finger_angles(self, qmap: dict[str, float], finger: str) -> np.ndarray:
        return np.array([
            qmap[f"{finger}_mcp"],
            qmap[f"{finger}_pip"],
            qmap[f"{finger}_dip"],
        ], dtype=float)

    @staticmethod
    def _rotation_matrix_xyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        rx = math.radians(float(roll_deg))
        ry = math.radians(float(pitch_deg))
        rz = math.radians(float(yaw_deg))
        Rx = np.array([[1.0,0.0,0.0],[0.0,math.cos(rx),-math.sin(rx)],[0.0,math.sin(rx),math.cos(rx)]], dtype=float)
        Ry = np.array([[math.cos(ry),0.0,math.sin(ry)],[0.0,1.0,0.0],[-math.sin(ry),0.0,math.cos(ry)]], dtype=float)
        Rz = np.array([[math.cos(rz),-math.sin(rz),0.0],[math.sin(rz),math.cos(rz),0.0],[0.0,0.0,1.0]], dtype=float)
        return Rz @ Ry @ Rx

    def finger_points(self, q: np.ndarray) -> dict[str, list[np.ndarray]]:
        q = np.asarray(q, dtype=float)
        qmap = {name: float(q[i]) for i, name in enumerate(self.joint_order)}
        out: dict[str, list[np.ndarray]] = {}
        for finger in FINGERS:
            angles = self._finger_angles(qmap, finger)
            lengths = self.finger_lengths[finger]
            p = np.array(self.base_offsets[finger], dtype=float)
            points = [p.copy()]
            cumulative = 0.0

            if self.base_euler_deg and finger in self.base_euler_deg:
                roll_deg, pitch_deg, yaw_deg = self.base_euler_deg[finger]
                R_base = self._rotation_matrix_xyz(roll_deg, pitch_deg, yaw_deg)
                for L, a in zip(lengths, angles):
                    cumulative += float(a)
                    # 拇指的 MuJoCo hinge 为 +X；非拇指为 -X。
                    flexion_sign = 1.0 if finger == "thumb" else -1.0
                    local_seg = np.array([0.0, L * math.cos(cumulative), flexion_sign * L * math.sin(cumulative)], dtype=float)
                    p = p + R_base @ local_seg
                    points.append(p.copy())
            else:
                yaw = math.radians(float(self.base_yaw_deg[finger]))
                direction = np.array([math.sin(yaw), math.cos(yaw), 0.0])
                for L, a in zip(lengths, angles):
                    cumulative += float(a)
                    p = p + direction * (L * math.cos(cumulative)) + np.array([0.0, 0.0, -L * math.sin(cumulative)])
                    points.append(p.copy())
            out[finger] = points
        return out

    def fingertips(self, q: np.ndarray) -> dict[str, np.ndarray]:
        pts = self.finger_points(q)
        return {finger: pts[finger][-1].copy() for finger in FINGERS}

    def keypoints(self, q: np.ndarray) -> dict[str, np.ndarray]:
        pts = self.finger_points(q)
        out: dict[str, np.ndarray] = {}
        for finger, seq in pts.items():
            out[f"{finger}_base"] = seq[0].copy()
            out[f"{finger}_j1"] = seq[1].copy()
            out[f"{finger}_j2"] = seq[2].copy()
            out[f"{finger}_tip"] = seq[3].copy()
        return out
