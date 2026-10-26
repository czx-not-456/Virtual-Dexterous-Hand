"""Optional 15/6 -> k -> 6 -> 11 CH-M6 constrained synergy mapper.

The six controls are *command semantics*, not evidence of six physical motors.
The five couplings are software assumptions from the provided joint_mapping.json.
All angle limits and couplings must be checked against the real hardware.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.hand_model.robot_hand import RobotHandModel
from src.simulation.teacher_chm6_control import TeacherCHM6ControlAdapter


CONTROL_ORDER = (
    "thumb_rotation", "thumb_bend", "index_base_bend", "middle_base_bend",
    "ring_base_bend", "pinky_base_bend",
)
CONTROL_LIMITS_DEG = np.array([100., 35., 90., 90., 90., 90.])


def fit_latent_model(sensors_deg: np.ndarray, target_controls_deg: np.ndarray, *,
                     k: int = 4, ridge: float = 0.05) -> dict:
    """Fit control PCA + paired sensor->shared-z ridge on *paired* samples.

    Fit only on training data. Test by pose/temporal holdout before deployment.
    """
    s = np.asarray(sensors_deg, dtype=float)
    u = np.asarray(target_controls_deg, dtype=float)
    if s.ndim != 2 or s.shape[1] != 15 or u.shape != (len(s), 6):
        raise ValueError("training arrays must be N x 15 sensors and N x 6 controls")
    if len(s) < max(8, k + 1) or not np.isfinite(s).all() or not np.isfinite(u).all():
        raise ValueError("training data is too small or non-finite")
    if not (1 <= k <= 6) or ridge < 0:
        raise ValueError("k must be 1..6 and ridge nonnegative")

    # Unitless, normalized controls: angle magnitudes do not dominate the PCA.
    un = np.clip(u / CONTROL_LIMITS_DEG, 0., 1.)
    control_mean = un.mean(axis=0)
    _, sing, v_t = np.linalg.svd(un - control_mean, full_matrices=False)
    basis = v_t[:k].T  # 6 x k, orthonormal columns
    z = (un - control_mean) @ basis

    # Train a second encoder to place 15 sensors into the SAME z coordinates.
    sensor_min, sensor_max = s.min(axis=0), s.max(axis=0)
    sensor_span = np.maximum(sensor_max - sensor_min, 1e-8)
    sn = np.clip((s - sensor_min) / sensor_span, 0., 1.)
    sensor_mean = sn.mean(axis=0)
    z_mean = z.mean(axis=0)
    xc = sn - sensor_mean
    weights = np.linalg.solve(xc.T @ xc + np.eye(15) * max(ridge, 1e-10),
                              xc.T @ (z - z_mean))
    bias = z_mean - sensor_mean @ weights
    # Full six-dimensional supervised reference is separate from the k-D encoder.
    # It is NOT an independent 11-joint ground truth, only a regression to the
    # six supplied teacher labels. Keep all six right singular vectors so task
    # adaptation can use the discarded directions without arbitrary completion.
    u_bias = un.mean(axis=0)
    full_weights = np.linalg.solve(
        xc.T @ xc + np.eye(15) * max(ridge, 1e-10),
        xc.T @ (un - u_bias),
    )
    full_bias = u_bias - sensor_mean @ full_weights
    explained = (sing ** 2) / max(float(np.sum(sing ** 2)), 1e-12)
    z_lo, z_hi = np.min(z, axis=0), np.max(z, axis=0)
    margin = np.maximum(0.15 * (z_hi - z_lo), 0.08)
    return {
        "version": 2,
        "k": k, "units": "degrees",
        "control_order": list(CONTROL_ORDER),
        "control_limits_deg": CONTROL_LIMITS_DEG.tolist(),
        "control_mean_normalized": control_mean.tolist(),
        "control_pca_basis": basis.tolist(),
        "control_pca_full_basis": v_t.T.tolist(),
        "sensor_to_controls_weights": full_weights.tolist(),
        "sensor_to_controls_bias": full_bias.tolist(),
        "sensor_min_deg": sensor_min.tolist(), "sensor_span_deg": sensor_span.tolist(),
        "sensor_to_latent_weights": weights.tolist(),
        "sensor_to_latent_bias": bias.tolist(),
        "explained_variance_ratio": explained.tolist(),
        "z_bounds": np.stack([z_lo - margin, z_hi + margin], axis=-1).tolist(),
        "training_rows": len(s),
        "warning": "Coupling ratios are software assumptions, not validated hardware transmissions.",
    }


class LatentSynergyMapper:
    def __init__(self, robot: RobotHandModel, model_path: str | Path,
                 coupling_path: str | Path = "数据手套数据/mod/joint_mapping.json") -> None:
        self.robot = robot
        self.model_path = Path(model_path)
        self.params = json.loads(self.model_path.read_text(encoding="utf-8"))
        p = self.params
        self.k = int(p["k"])
        if p["control_order"] != list(CONTROL_ORDER):
            raise ValueError("latent model control order differs from delivered base_mapping")
        self.control_limits_deg = np.asarray(p["control_limits_deg"], float)
        self.mu = np.asarray(p["control_mean_normalized"], float)
        self.basis = np.asarray(p["control_pca_basis"], float)
        self.s_min = np.asarray(p["sensor_min_deg"], float)
        self.s_span = np.asarray(p["sensor_span_deg"], float)
        self.sensor_weights = np.asarray(p["sensor_to_latent_weights"], float)
        self.sensor_bias = np.asarray(p["sensor_to_latent_bias"], float)
        self.z_bounds = np.asarray(p["z_bounds"], float)
        assert self.mu.shape == (6,) and self.basis.shape == (6, self.k)
        assert self.sensor_weights.shape == (15, self.k)
        self.teacher = TeacherCHM6ControlAdapter(robot, coupling_path)
        self.joint_index = {name: i for i, name in enumerate(robot.joint_order)}
        self.active_indices = np.array([self.joint_index[n] for n in self.teacher.active_order])
        self._last_output_z: np.ndarray | None = None

    def encode_controls(self, controls_deg: np.ndarray) -> np.ndarray:
        ctrl = np.asarray(controls_deg, float)
        if ctrl.shape != (6,) or not np.isfinite(ctrl).all():
            raise ValueError("expected six finite controls in degrees")
        return (np.clip(ctrl / self.control_limits_deg, 0., 1.) - self.mu) @ self.basis

    def encode_sensors(self, sensors_deg: np.ndarray) -> np.ndarray:
        s = np.asarray(sensors_deg, float)
        if s.shape != (15,) or not np.isfinite(s).all():
            raise ValueError("expected 15 finite sensor angles in degrees")
        return np.clip((s - self.s_min) / self.s_span, 0., 1.) @ self.sensor_weights + self.sensor_bias

    def decode_controls(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        if z.shape != (self.k,) or not np.isfinite(z).all():
            raise ValueError(f"expected {self.k} finite latent variables")
        un = np.clip(self.mu + z @ self.basis.T, 0., 1.)
        return un * self.control_limits_deg

    def decode_joints(self, z: np.ndarray) -> np.ndarray:
        return self.teacher.expand_degrees(self.decode_controls(z))

    def from_observation(self, sensors_deg: np.ndarray,
                         controls_deg: np.ndarray | None = None,
                         *, source: str = "sensor", sensor_weight: float = 0.8) -> np.ndarray:
        """Return 11 *radian* targets. In 'blend', both encoders share coordinates."""
        if source == "sensor":
            z = self.encode_sensors(sensors_deg)
        elif source == "control":
            if controls_deg is None:
                raise ValueError("controls required for 'control' source")
            z = self.encode_controls(controls_deg)
        elif source == "blend":
            if controls_deg is None or not (0. <= sensor_weight <= 1.):
                raise ValueError("blend needs controls and a weight in [0,1]")
            z = sensor_weight * self.encode_sensors(sensors_deg) + (
                1. - sensor_weight) * self.encode_controls(controls_deg)
        else:
            raise ValueError(f"unknown source {source!r}")
        return self.decode_joints(z)

    def encode_joints(self, q_rad: np.ndarray) -> np.ndarray:
        """Recover least-squares active controls from all 11 joints, then project onto z.

        Helps keep post-optimizer/contact-controller targets on the same k-D manifold.
        """
        q = np.asarray(q_rad, float)
        if q.shape != (11,) or not np.isfinite(q).all():
            raise ValueError("expected eleven finite joint angles in radians")
        active = q[self.active_indices].copy()
        for driver, child, gain in self.teacher.couplings:
            i = self.teacher.active_order.index(driver)
            active[i] = (active[i] + gain * q[self.joint_index[child]]) / (1. + gain * gain)
        z0 = self.encode_controls(np.rad2deg(active))
        # Clipped control dimensions make the 6->k inverse nonlinear.
        # Refine the projection so already-decoded joint targets remain fixed.
        if np.max(np.abs(self.decode_joints(z0) - q)) > 1e-7:
            from scipy.optimize import least_squares
            result = least_squares(lambda z: self.decode_joints(z) - q, z0,
                                   max_nfev=130, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            return result.x
        return z0

    def project_joints(self, q_rad: np.ndarray) -> np.ndarray:
        return self.decode_joints(self.encode_joints(q_rad))

    def project_and_rate_limit(self, target: np.ndarray, previous: np.ndarray | None,
                               max_speed_rad_s: np.ndarray, dt_s: float) -> np.ndarray:
        """Limit velocity along a line in *latent space*, preserving its manifold.

        Since joint clipping is piecewise linear, simply interpolating 11 joint
        angles can leave the k-dimensional manifold. Instead, binary-search a
        safe step between the previous and desired latent states.
        """
        target_z = self.encode_joints(target)
        if previous is None:
            self._last_output_z = target_z.copy()
            return self.decode_joints(target_z)
        previous = np.asarray(previous, float)
        max_step = np.asarray(max_speed_rad_s, float) * dt_s
        if max_step.shape != (11,) or np.any(max_step <= 0):
            raise ValueError("max joint speed must be positive for all 11 joints")
        # In normal streaming usage the previous command comes from this mapper.
        if (self._last_output_z is not None and
            np.max(np.abs(self.decode_joints(self._last_output_z) - previous)) < 1e-8):
            prev_z = self._last_output_z
        else:
            # For a separately provided initial command, recover a matching
            # latent state by nonlinear least squares (handles decoder clipping).
            from scipy.optimize import least_squares
            fit = least_squares(lambda z: self.decode_joints(z) - previous,
                                x0=self.encode_joints(previous), max_nfev=150,
                                ftol=1e-11, xtol=1e-11, gtol=1e-11)
            prev_z = fit.x
            if np.max(np.abs(self.decode_joints(prev_z) - previous)) > 1e-5:
                raise ValueError("previous command is not on the latent manifold")
        def candidate(fraction: float) -> np.ndarray:
            return self.decode_joints(prev_z + fraction * (target_z - prev_z))
        if np.all(np.abs(candidate(1.) - previous) <= max_step):
            fraction = 1.
        else:
            low, high = 0., 1.
            for _ in range(38):
                mid = (low + high) / 2.
                if np.all(np.abs(candidate(mid) - previous) <= max_step):
                    low = mid
                else:
                    high = mid
            fraction = low
        self._last_output_z = prev_z + fraction * (target_z - prev_z)
        return candidate(fraction)
