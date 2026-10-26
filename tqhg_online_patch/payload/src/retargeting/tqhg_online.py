"""Online 15->6 task-gated mappings with the SAME 6->11 coupled output guard.

This deliberately bypasses the unrelated retargeting optimizer when mapping
baselines are compared, so improvements cannot be attributed to its settings.
The 6-control model and calibration residuals must be trained separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import CONTROL_LIMITS_DEG, CONTROL_ORDER
from src.retargeting.task_weighted_hybrid import gate_prediction, optimal_weight
from src.simulation.teacher_chm6_control import TeacherCHM6ControlAdapter


class TQHGOnlineMapper:
    METHODS = ("ridge6", "pca4", "fixed-blend", "single-gate", "tqhg")

    def __init__(self, robot: RobotHandModel, calibration_file: str | Path):
        self.robot = robot
        self.calibration_file = Path(calibration_file)
        cfg = json.loads(self.calibration_file.read_text(encoding="utf-8"))
        if cfg.get("schema") != "tqhg_online_calibration_v1":
            raise ValueError("TQHG model file has invalid schema; run scripts/train_tqhg_online.py")
        self.cfg = cfg
        p = cfg["model"]
        if p["control_order"] != list(CONTROL_ORDER):
            raise ValueError("6-control order mismatch")
        self.sensor_min = np.asarray(p["sensor_min_deg"], dtype=float)
        self.sensor_span = np.asarray(p["sensor_span_deg"], dtype=float)
        self.W_ridge = np.asarray(p["sensor_to_controls_weights"], dtype=float)
        self.b_ridge = np.asarray(p["sensor_to_controls_bias"], dtype=float)
        self.W_latent = np.asarray(p["sensor_to_latent_weights"], dtype=float)
        self.b_latent = np.asarray(p["sensor_to_latent_bias"], dtype=float)
        self.pca_basis = np.asarray(p["control_pca_basis"], dtype=float)
        self.mu = np.asarray(p["control_mean_normalized"], dtype=float)
        self.errors_ridge = np.asarray(cfg["calibration_errors_ridge"], dtype=float)
        self.errors_pca = np.asarray(cfg["calibration_errors_pca"], dtype=float)
        self.limits_deg = np.asarray(p["control_limits_deg"], dtype=float)
        self.teacher = TeacherCHM6ControlAdapter(robot, "数据手套数据/mod/joint_mapping.json")
        if (self.sensor_min.shape != (15,) or self.sensor_span.shape != (15,)
                or self.W_ridge.shape != (15, 6) or self.b_ridge.shape != (6,)
                or self.W_latent.shape[0] != 15 or self.pca_basis.shape[0] != 6
                or self.pca_basis.shape[1] != self.W_latent.shape[1]
                or self.errors_ridge.shape != self.errors_pca.shape
                or self.errors_ridge.ndim != 2 or self.errors_ridge.shape[1] != 6
                or self.errors_ridge.shape[0] < 12
                or not np.allclose(self.limits_deg, CONTROL_LIMITS_DEG)):
            raise ValueError("Inconsistent 15->6 model/calibration dimensions")
        values = [self.sensor_min, self.sensor_span, self.W_ridge, self.b_ridge,
                  self.W_latent, self.b_latent, self.pca_basis, self.mu,
                  self.errors_ridge, self.errors_pca]
        if not all(np.isfinite(v).all() for v in values) or np.any(self.sensor_span <= 0):
            raise ValueError("Nonfinite or invalid online model")

        # EXACT same software coupling as TEQS: 6 normalized controls to 11 rad.
        self.C = np.zeros((11, 6), dtype=float)
        for col, name in enumerate(self.teacher.active_order):
            self.C[self.teacher.index[name], col] = np.deg2rad(self.limits_deg[col])
        for driver, child, gain in self.teacher.couplings:
            col = self.teacher.active_order.index(driver)
            self.C[self.teacher.index[child], col] = self.C[self.teacher.index[driver], col] * gain
        self.lower, self.upper = np.zeros(6), np.ones(6)
        for j in range(6):
            rows = np.flatnonzero(np.abs(self.C[:, j]) > 1e-12)
            for row in rows:
                lo, hi = robot.ranges[row]
                a, b = sorted((lo / self.C[row, j], hi / self.C[row, j]))
                self.lower[j] = max(self.lower[j], a)
                self.upper[j] = min(self.upper[j], b)
        if np.any(self.lower > self.upper):
            raise ValueError("Software coupling conflicts with robot joint limits")
        self.last_gate = {"w_task": float("nan"), "w_null": float("nan"),
                          "visible_rank": 0, "box_clipped": False}

    def _predict_experts(self, sensors: np.ndarray):
        s = np.asarray(sensors, dtype=float)
        if s.shape != (15,) or not np.isfinite(s).all():
            raise ValueError("TQHG requires 15 finite glove channels")
        sn = np.clip((s - self.sensor_min) / self.sensor_span, 0, 1)
        ridge = np.clip(sn @ self.W_ridge + self.b_ridge, 0, 1)
        z = sn @ self.W_latent + self.b_latent
        pca = np.clip(self.mu + z @ self.pca_basis.T, 0, 1)
        return ridge, pca

    def _jacobian(self, simulator, task_mode: str) -> np.ndarray:
        if task_mode == "pinch":
            fingers = ("thumb", "index")
        elif task_mode in ("wrap", "sphere"):
            fingers = ("thumb", "index", "middle", "ring", "little")
        else:
            return np.zeros((1, 6))
        if simulator is None:
            # --sim none: gate is unobservable, calibrated task-space weight off.
            return np.zeros((1, 6))
        J = []
        indices = {name: i for i, name in enumerate(self.robot.joint_order)}
        for finger in fingers:
            names = self.robot.joints_for_finger(finger)
            cols = np.array([indices[name] for name in names], dtype=int)
            J.append(np.asarray(simulator.fingertip_jacobian(finger, names), dtype=float)
                     @ self.C[cols, :])
        return np.vstack(J)

    def predict(self, sensors_deg, *, method: str, simulator=None, task_mode="pinch") -> np.ndarray:
        if method not in self.METHODS:
            raise ValueError(f"Invalid mapping method: {method}")
        ridge, pca = self._predict_experts(sensors_deg)
        self.last_gate = {"w_task": float("nan"), "w_null": float("nan"),
                          "visible_rank": 0, "box_clipped": False}
        if method == "ridge6":
            return np.clip(ridge, self.lower, self.upper)
        if method == "pca4":
            return np.clip(pca, self.lower, self.upper)
        if method == "fixed-blend":
            return np.clip(0.5 * ridge + 0.5 * pca, self.lower, self.upper)
        J = self._jacobian(simulator, task_mode)
        result, info = gate_prediction(
            ridge, pca, self.errors_ridge, self.errors_pca, J,
            split=(method == "tqhg"),
        )
        self.last_gate = info
        return np.clip(result, self.lower, self.upper)

    def encode_joints(self, q_rad) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (11,) or not np.isfinite(q).all():
            raise ValueError("Invalid 11D joint state")
        return np.clip(np.linalg.lstsq(self.C, q, rcond=None)[0], self.lower, self.upper)

    def decode_controls(self, u) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.shape != (6,) or not np.isfinite(u).all():
            raise ValueError("Invalid 6D control")
        return self.C @ np.clip(u, self.lower, self.upper)

    def project_and_rate_limit(self, target, previous, max_speed_rad_s, dt_s) -> np.ndarray:
        u_des = self.encode_joints(target)
        if previous is None:
            return self.decode_controls(u_des)
        q_prev = np.asarray(previous, dtype=float)
        u_prev = self.encode_joints(q_prev)
        if np.max(np.abs(self.C @ u_prev - q_prev)) > 2e-5:
            raise ValueError("Previous command violated the six-control coupling")
        speeds = np.asarray(max_speed_rad_s, dtype=float)
        if speeds.shape != (11,) or not np.all(np.isfinite(speeds)) or np.any(speeds <= 0) or dt_s <= 0:
            raise ValueError("Invalid velocity limits or timestep")
        u_new = u_des.copy()
        for col in range(6):
            members = np.flatnonzero(np.abs(self.C[:, col]) > 1e-12)
            max_du = np.min(speeds[members] * dt_s / np.abs(self.C[members, col]))
            u_new[col] = np.clip(u_des[col], u_prev[col] - max_du, u_prev[col] + max_du)
        u_new = np.clip(u_new, self.lower, self.upper)
        return self.C @ u_new
