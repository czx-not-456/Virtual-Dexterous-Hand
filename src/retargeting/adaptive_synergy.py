"""Task-adaptive CH-M6 synergy mapping with bounded complementary residuals.

Research prototype, not hardware-calibrated: a full 6x6 PCA basis represents
command directions; task-space fingertip reachability chooses k primary columns,
while a bounded residual in the orthogonal complement can recover information.
All eleven joints remain coupled to *six software control commands*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.intent_recognition import Intent
from src.hand_model.robot_hand import RobotHandModel
from .latent_synergy import LatentSynergyMapper

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


@dataclass(frozen=True)
class AdaptiveDiagnostics:
    dimension: int
    projection_error_m: float
    residual_norm: float
    residual_cap: float
    feasible: bool


class TaskAdaptiveSynergyMapper:
    """Builds on the existing k-D model; requires a newly trained full-basis file.

    Decoder: u_norm = mu + B[:,:k] z + B[:,k:] delta.
    Constraints: robot joint ranges AND the six control ranges are enforced
    during optimization, not via unsafe after-the-fact independent joint clips.
    """

    FINGERS = {
        Intent.NEUTRAL: ('thumb', 'index', 'middle', 'ring', 'little'),
        Intent.PINCH: ('thumb', 'index'),
        Intent.WRAP: ('thumb', 'index', 'middle', 'ring', 'little'),
    }

    def __init__(self, base: LatentSynergyMapper, config: dict) -> None:
        self.base = base
        self.robot: RobotHandModel = base.robot
        self.config = dict(config)
        p = base.params
        if 'control_pca_full_basis' not in p or 'sensor_to_controls_weights' not in p:
            raise ValueError('Adaptive mode requires a retrained version-2 full-basis latent model')
        self.B = np.asarray(p['control_pca_full_basis'], float)
        self.W = np.asarray(p['sensor_to_controls_weights'], float)
        self.b = np.asarray(p['sensor_to_controls_bias'], float)
        self.mu = np.asarray(p['control_mean_normalized'], float)
        self.limits_deg = np.asarray(p['control_limits_deg'], float)
        self.sensor_min = np.asarray(p['sensor_min_deg'], float)
        self.sensor_span = np.asarray(p['sensor_span_deg'], float)
        if (self.B.shape != (6, 6) or self.W.shape != (15, 6)
            or self.b.shape != (6,) or not np.isfinite(self.B).all()
            or not np.allclose(self.B.T @ self.B, np.eye(6), atol=1e-5)):
            raise ValueError('Invalid orthonormal full control basis or ridge encoder')
        self.k_min = int(config.get('k_min', 2))
        self.k_max = int(config.get('k_max', 5))
        if not 1 <= self.k_min <= self.k_max <= 6:
            raise ValueError('expected 1 <= k_min <= k_max <= 6')
        self.projection_abs_m = float(config.get('projection_abs_m', 0.0015))
        self.projection_relative = float(config.get('projection_relative', 0.12))
        self.fd_step = float(config.get('jacobian_step', 1e-4))
        self.command_scale = {key: float(value) for key, value in
                              config.get('command_scale_m', {}).items()}
        self.residual_caps = {key: float(value) for key, value in
                              config.get('residual_caps', {}).items()}
        self.lambda_residual = float(config.get('lambda_residual', 0.025))
        self.lambda_reference = float(config.get('lambda_reference', 0.15))
        self.maxiter = int(config.get('maxiter', 80))
        if (min(self.projection_abs_m, self.projection_relative, self.fd_step,
                self.lambda_residual, self.lambda_reference) < 0
            or self.fd_step <= 0 or self.maxiter <= 0
            or any(v < 0 for v in self.residual_caps.values())):
            raise ValueError('adaptive thresholds, costs and steps must be nonnegative')

        # Matrix from six normalized control coordinates to eleven joint RADIANS.
        self.C = np.zeros((11, 6), float)
        t = base.teacher
        for col, joint in enumerate(t.active_order):
            self.C[t.index[joint], col] = np.deg2rad(self.limits_deg[col])
        for driver, child, gain in t.couplings:
            col = t.active_order.index(driver)
            self.C[t.index[child], col] = self.C[t.index[driver], col] * gain
        self.lower = np.zeros(6)
        self.upper = np.ones(6)
        for col in range(6):
            indices = np.flatnonzero(np.abs(self.C[:, col]) > 1e-12)
            for idx in indices:
                factor = self.C[idx, col]
                lo, hi = self.robot.ranges[idx]
                bounds = sorted((lo / factor, hi / factor))
                self.lower[col] = max(self.lower[col], bounds[0])
                self.upper[col] = min(self.upper[col], bounds[1])
        if np.any(self.lower > self.upper) or np.any(self.mu < self.lower - 1e-6) or np.any(self.mu > self.upper + 1e-6):
            raise ValueError('Control/joint constraints are inconsistent with trained mean')
        self.last_diag: AdaptiveDiagnostics | None = None

    def _check_control(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, float)
        if u.shape != (6,) or not np.isfinite(u).all():
            raise ValueError('expected six finite normalized controls')
        return np.clip(u, self.lower, self.upper)

    def controls_from_observation(self, sensors_deg: np.ndarray | None,
                                  controls_deg: np.ndarray | None,
                                  source: str = 'sensor', sensor_weight: float = 0.8) -> np.ndarray:
        if source not in ('sensor', 'control', 'blend'):
            raise ValueError('source must be sensor, control or blend')
        if source in ('sensor', 'blend'):
            s = np.asarray(sensors_deg, float)
            if s.shape != (15,) or not np.isfinite(s).all():
                raise ValueError('sensor/blend source requires 15 finite angles')
            s = np.clip((s - self.sensor_min) / self.sensor_span, 0, 1)
            pred = s @ self.W + self.b
        if source in ('control', 'blend'):
            ctrl = np.asarray(controls_deg, float)
            if ctrl.shape != (6,) or not np.isfinite(ctrl).all():
                raise ValueError('control/blend source requires six finite angles')
            ctrl = ctrl / self.limits_deg
        if source == 'sensor':
            raw = pred
        elif source == 'control':
            raw = ctrl
        else:
            if not (0 <= sensor_weight <= 1):
                raise ValueError('sensor_weight must be in [0,1]')
            raw = sensor_weight * pred + (1 - sensor_weight) * ctrl
        return self._check_control(raw)

    def decode_controls(self, normalized: np.ndarray) -> np.ndarray:
        u = self._check_control(normalized)
        return self.C @ u

    def encode_joints(self, q: np.ndarray) -> np.ndarray:
        """Least-squares 11->6 constrained coupled commands for projection/limiting."""
        q = np.asarray(q, float)
        if q.shape != (11,) or not np.isfinite(q).all():
            raise ValueError('expected eleven finite joint angles')
        return self._check_control(np.linalg.lstsq(self.C, q, rcond=None)[0])

    def project_joints(self, q: np.ndarray) -> np.ndarray:
        return self.decode_controls(self.encode_joints(q))

    def _project_to_active_subspace(self, q_target: np.ndarray) -> np.ndarray:
        """Project servo/filter output onto current primary+bounded-residual set.

        The final physical rate limiter can temporarily retain a preceding
        frame's coefficient if k/cap has changed; avoiding a discontinuous
        command takes priority over instantaneous latent-bound satisfaction.
        """
        ut = self.encode_joints(q_target)
        if self.last_diag is None:
            return ut
        k, cap = self.last_diag.dimension, self.last_diag.residual_cap
        B, mu = self.B, self.mu
        x0 = B.T @ (ut - mu)
        x0[k:] = np.clip(x0[k:], -cap, cap)
        def feasible(x):
            u = mu + B @ x
            return np.all(u >= self.lower - 1e-9) and np.all(u <= self.upper + 1e-9)
        if feasible(x0):
            return mu + B @ x0
        if minimize is None:  # pragma: no cover
            return mu.copy()
        # A convex constrained least-squares problem (6 variables).
        A = self.C @ B
        target_q = self.C @ ut
        def fun(x):
            error = self.C @ mu + A @ x - target_q
            return .5 * float(error @ error)
        def jac(x):
            return A.T @ (self.C @ mu + A @ x - target_q)
        bounds = [(None, None)] * k + [(-cap, cap)] * (6 - k)
        result = minimize(fun, np.zeros(6), jac=jac, method='SLSQP', bounds=bounds,
                          constraints=[{'type': 'ineq', 'fun': lambda x: mu + B @ x - self.lower,
                                        'jac': lambda x: B},
                                       {'type': 'ineq', 'fun': lambda x: self.upper - mu - B @ x,
                                        'jac': lambda x: -B}],
                          options={'maxiter': 80, 'ftol': 1e-11, 'disp': False})
        if feasible(result.x) and np.isfinite(result.x).all():
            return mu + B @ result.x
        return mu.copy()  # guaranteed feasible, if numerics cannot solve

    def project_and_rate_limit(self, target: np.ndarray, previous: np.ndarray | None,
                               max_speed_rad_s: np.ndarray, dt_s: float) -> np.ndarray:
        """Linear interpolation in *six coupled controls*, preserving all limits.

        previous should be the preceding commanded target (not lagged physical
        joint observations, which need not lie on the coupled software manifold).
        """
        u = self._project_to_active_subspace(target)
        if previous is None:
            return self.decode_controls(u)
        prev = np.asarray(previous, float)
        v = np.asarray(max_speed_rad_s, float)
        if prev.shape != (11,) or v.shape != (11,) or np.any(v <= 0) or dt_s <= 0:
            raise ValueError('invalid previous targets, velocities or timestep')
        up = self.encode_joints(prev)
        qp = self.C @ up
        if np.max(np.abs(qp - prev)) > 1e-6:
            raise ValueError('previous command must be on the coupled six-control manifold')
        delta = self.C @ (u - up)
        mask = np.abs(delta) > 1e-14
        alpha = min(1.0, float(np.min(v[mask] * dt_s / np.abs(delta[mask])))) if np.any(mask) else 1.0
        return self.decode_controls(up + alpha * (u - up))

    def _tip_vector(self, u: np.ndarray, fingers: tuple[str, ...]) -> np.ndarray:
        tips = self.robot.fingertips(self.decode_controls(u))
        return np.concatenate([tips[f] for f in fingers])

    def _jacobian(self, u: np.ndarray, fingers: tuple[str, ...]) -> np.ndarray:
        J = np.zeros((3 * len(fingers), 6))
        for i in range(6):
            up, um = u.copy(), u.copy()
            up[i] = min(self.upper[i], up[i] + self.fd_step)
            um[i] = max(self.lower[i], um[i] - self.fd_step)
            step = up[i] - um[i]
            if step > 1e-12:
                J[:, i] = (self._tip_vector(up, fingers) - self._tip_vector(um, fingers)) / step
        return J

    def select_dimension(self, q_prev: np.ndarray, u_target: np.ndarray,
                         intent: Intent) -> tuple[int, float]:
        """Smallest k whose extra task-space error relative to full-6D is small.

        Task reachability uses J(u_prev)*(u_target-u_prev), locally linearized.
        A task-dependent, metrically scaled control term avoids silently dropping
        independent commands that happen to produce little fingertip movement.
        """
        if intent not in self.FINGERS:
            raise ValueError(f'unknown intent {intent}')
        up = self.encode_joints(q_prev)
        target = self._check_control(u_target)
        J = self._jacobian(up, self.FINGERS[intent])
        scale = self.command_scale.get(intent.value, 0.006)
        A = np.vstack([J, scale * np.eye(6)])
        desired = A @ (target - up)
        if np.linalg.norm(desired) <= 1e-10:
            return self.k_min, 0.0
        # Relative to the best full six-dimensional reachable displacement.
        full_projection = (A @ np.linalg.lstsq(A, desired, rcond=None)[0])
        threshold = max(self.projection_abs_m, self.projection_relative * np.linalg.norm(full_projection))
        for k in range(self.k_min, self.k_max + 1):
            AB = A @ self.B[:, :k]
            projected = AB @ np.linalg.lstsq(AB, full_projection, rcond=None)[0]
            error = float(np.linalg.norm(full_projection - projected))
            if error <= threshold:
                return k, error
        k = self.k_max
        AB = A @ self.B[:, :k]
        projected = AB @ np.linalg.lstsq(AB, full_projection, rcond=None)[0]
        return k, float(np.linalg.norm(full_projection - projected))

    def optimize(self, objective, q_prev: np.ndarray, intent: Intent,
                 target: np.ndarray, *, ftol: float = 1e-6):
        """Solve at most six coefficient variables with explicit linear limits.

        Returns joint targets, scipy result or None, and diagnostic statistics.
        Residual coefficients are bounded but penalized: enabling them is NOT
        equivalent to claiming the robot has eleven independent actuators.
        """
        target = self._check_control(target)
        k, projection_error = self.select_dimension(q_prev, target, intent)
        cap = self.residual_caps.get(intent.value, 0.12)
        cap = max(0., cap)
        B, mu = self.B, self.mu
        bounds = [(None, None)] * k + [(-cap, cap)] * (6 - k)
        u_from_x = lambda x: mu + B @ x
        def fun(x):
            u = u_from_x(x)
            q = self.C @ u
            residual = x[k:]
            return (float(objective(q))
                    + self.lambda_reference * float(np.mean((u - target) ** 2))
                    + self.lambda_residual * float(np.dot(residual, residual)))
        # Coefficient origin is always feasible (trained mean is interior).
        x0 = np.zeros(6)
        x_target = B.T @ (target - mu)
        x_target[k:] = np.clip(x_target[k:], -cap, cap)
        trial = u_from_x(x_target)
        if np.all(trial >= self.lower - 1e-12) and np.all(trial <= self.upper + 1e-12):
            x0 = x_target
        else:
            # Feasible convex interpolation from the training mean.
            diff = trial - mu
            alpha = 1.
            for i in range(6):
                if diff[i] > 0:
                    alpha = min(alpha, (self.upper[i] - mu[i]) / diff[i])
                elif diff[i] < 0:
                    alpha = min(alpha, (self.lower[i] - mu[i]) / diff[i])
            x0 = x_target * max(0., min(1., alpha)) * 0.999999
        result = None
        x = x0
        if minimize is not None:
            result = minimize(
                fun, x0, method='SLSQP', bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': lambda a: u_from_x(a) - self.lower,
                              'jac': lambda a: B},
                             {'type': 'ineq', 'fun': lambda a: self.upper - u_from_x(a),
                              'jac': lambda a: -B}],
                options={'maxiter': self.maxiter, 'ftol': ftol, 'disp': False},
            )
            if (np.isfinite(result.x).all()
                and np.all(u_from_x(result.x) >= self.lower - 1e-7)
                and np.all(u_from_x(result.x) <= self.upper + 1e-7)
                and fun(result.x) <= fun(x0) + 1e-9):
                x = result.x
        # Explicit safety check, not post-hoc joint clipping.
        u = u_from_x(x)
        feasible = bool(np.all(u >= self.lower - 2e-7) and np.all(u <= self.upper + 2e-7))
        q = self.C @ self._check_control(u)
        diag = AdaptiveDiagnostics(k, projection_error,
                                   float(np.linalg.norm(x[k:])), cap, feasible)
        self.last_diag = diag
        return q, result, diag
