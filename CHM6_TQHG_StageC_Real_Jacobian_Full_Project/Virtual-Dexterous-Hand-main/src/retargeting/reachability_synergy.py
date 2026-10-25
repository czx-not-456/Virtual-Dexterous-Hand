"""Reachability-gated, sparse-residual task-conditioned retargeting (research prototype).

Unlike Scheme B, stage one optimizes exactly k variables. A sparse second
stage is *event triggered* by a task-space reachability certificate and only
adds complementary directions if necessary. A local hand Jacobian shapes the
basis online. This implementation does not claim a global optimality proof,
measured real-hand accuracy, or verified hardware transmission ratios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from src.features.intent_recognition import Intent
from .adaptive_synergy import TaskAdaptiveSynergyMapper, minimize


@dataclass(frozen=True)
class ReachabilityDiagnostics:
    dimension: int                 # primary optimized variables (k)
    projection_error_m: float      # local task-feature projection gap
    residual_norm: float
    residual_cap: float
    feasible: bool                 # software limits and coupling only
    optimized_variables: int       # REAL NLP dimension, not a declared k
    active_residuals: int
    stage: int                     # 1 = primary, 2 = residual rescue
    task_linear_error_m: float
    task_full_approx_error_m: float # unattainable portion of user task target
    nonlinear_objective: float
    fallback_full_reference: bool = False
    command_subspace_rank: int = 6


class ReachabilityGatedSynergyMapper(TaskAdaptiveSynergyMapper):
    """Local task-oriented synergy basis + true low-rank optimization + rescue.

    Constraint certificate is only local linearized reachability (Jacobian),
    not a certificate of task success or absence of all collisions.
    """

    def __init__(self, base, config: dict) -> None:
        super().__init__(base, config)
        self.task_prior_weight_m = float(config.get('task_prior_weight_m', 0.012))
        self.task_direction_weight_m = float(config.get('task_direction_weight_m', 0.012))
        self.task_command_weight_m = float(config.get('task_command_weight_m', 0.004))
        self.rescue_threshold_m = float(config.get('rescue_threshold_m', 0.0015))
        self.rescue_min_improvement_m = float(config.get('rescue_min_improvement_m', 0.00002))
        self.lambda_sparse = float(config.get('lambda_sparse', 0.004))
        self.fallback_full_reference = bool(config.get('fallback_full_reference', False))
        self.fallback_min_gain = float(config.get('fallback_min_gain', 1e-8))
        self.max_residual_axes = int(config.get('max_residual_axes', 2))
        self.delta_smoothing = float(config.get('delta_smoothing', 1e-4))
        if any(v < 0 for v in (self.task_prior_weight_m, self.task_direction_weight_m,
                              self.task_command_weight_m, self.rescue_threshold_m,
                              self.rescue_min_improvement_m, self.lambda_sparse)):
            raise ValueError('all reachability thresholds/penalties must be nonnegative')
        if self.delta_smoothing <= 0 or not 0 <= self.max_residual_axes <= 6:
            raise ValueError('invalid sparse residual settings')
        self.active_basis: np.ndarray | None = None  # (6, k + opened complement)
        self.active_anchor: np.ndarray | None = None  # feasible six-dimensional command
        self.active_residual_start: int = 0
        self.active_cap: float = 0.
        self.last_diag: ReachabilityDiagnostics | None = None

    def _features(self, up: np.ndarray, reference: np.ndarray,
                  intent: Intent, task_targets_m: dict | None):
        fingers = self.FINGERS[intent]
        J = self._jacobian(up, fingers)
        current_tips = self._tip_vector(up, fingers)
        tip_goal = (self._tip_vector(reference, fingers)
                    if task_targets_m is None else
                    np.concatenate([np.asarray(task_targets_m[f], float).reshape(3)
                                    for f in fingers]))
        if not np.isfinite(tip_goal).all():
            raise ValueError('task targets must be finite')
        # Real fingertip displacements get priority; a small control-reference
        # term protects independent actions in nearly invisible task directions.
        weight = self.task_command_weight_m
        A = np.vstack((J, weight * np.eye(6)))
        rhs = np.concatenate((tip_goal - current_tips, weight * (reference - up)))
        return J, A, rhs

    def _basis(self, J: np.ndarray, direction: np.ndarray) -> np.ndarray:
        # Analytic, *not* a learned task-specific PCA. The 6-vector covariance
        # prior regularizes unobservable task directions with current data.
        variance = np.asarray(self.base.params.get('explained_variance_ratio', [1]*6), float)
        if variance.shape != (6,):
            variance = np.ones(6)
        variance = np.maximum(variance, 0.)
        variance = variance / max(float(np.max(variance)), 1e-12)
        H = J.T @ J + self.task_prior_weight_m ** 2 * (
            (self.B * variance[None, :]) @ self.B.T)
        norm = float(direction @ direction)
        if norm > 1e-12:
            H += self.task_direction_weight_m ** 2 * np.outer(direction, direction) / norm
        _, V = np.linalg.eigh(H)
        return V[:, ::-1].copy()

    @staticmethod
    def _linear_error(A: np.ndarray, desired: np.ndarray,
                      displacement: np.ndarray) -> float:
        return float(np.linalg.norm(A @ displacement - desired))

    def _feasible(self, u: np.ndarray, tol: float = 1e-7) -> bool:
        return bool(np.isfinite(u).all() and
                    np.all(u >= self.lower - tol) and np.all(u <= self.upper + tol))

    def _stage_solve(self, objective: Callable, up: np.ndarray, ref: np.ndarray,
                     D: np.ndarray, k: int, cap: float, x_start: np.ndarray,
                     ftol: float):
        """Optimize exactly D.shape[1] variables, with linear joint constraints."""
        n = D.shape[1]
        if x_start.shape != (n,):
            raise ValueError('invalid stage initial variables')
        def u_of(x):
            return up + D @ x
        def loss(x):
            u = u_of(x)
            extra = x[k:]
            sparse = np.sqrt(extra**2 + self.delta_smoothing**2) - self.delta_smoothing
            return (float(objective(self.C @ u))
                    + self.lambda_reference * float(np.mean((u - ref)**2))
                    + self.lambda_residual * float(extra @ extra)
                    + self.lambda_sparse * float(np.sum(sparse)))
        bounds = [(None, None)] * k + [(-cap, cap)] * (n - k)
        candidate = x_start.copy()
        # Zero is guaranteed feasible: the anchor is itself a valid command.
        if not self._feasible(u_of(candidate)):
            candidate = np.zeros(n)
        solver_result = None
        if minimize is not None:
            solver_result = minimize(
                loss, candidate, method='SLSQP', bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: u_of(x) - self.lower,
                     'jac': lambda x: D},
                    {'type': 'ineq', 'fun': lambda x: self.upper - u_of(x),
                     'jac': lambda x: -D},
                ],
                options={'maxiter': self.maxiter, 'ftol': ftol, 'disp': False})
            if (np.asarray(solver_result.x).shape == (n,) and
                self._feasible(u_of(solver_result.x)) and
                loss(solver_result.x) <= loss(candidate) + 1e-10):
                candidate = np.asarray(solver_result.x, float)
        return candidate, solver_result, float(loss(candidate))

    def _choose_dimension(self, A: np.ndarray, desired: np.ndarray,
                          V: np.ndarray, eps: float) -> tuple[int, float]:
        """Extensible local rank selector, called by the original Scheme C solver."""
        k, certificate = self.k_max, float('inf')
        for d in range(self.k_min, self.k_max + 1):
            Ak = A @ V[:, :d]
            projected = Ak @ np.linalg.lstsq(Ak, desired, rcond=None)[0]
            certificate = float(np.linalg.norm(projected - desired))
            k = d
            if certificate <= eps:
                break
        return k, certificate

    def _contact_rescue_needed(self) -> bool:
        return False  # Scheme C does not use observed contact feedback.

    def _allowed_residual_axes(self, k: int, intent: Intent) -> int:
        return min(self.max_residual_axes, 6 - k)

    def optimize(self, objective: Callable, q_prev: np.ndarray, intent: Intent,
                 target: np.ndarray, *, ftol: float = 1e-6,
                 task_targets_m: dict | None = None):
        if intent not in self.FINGERS:
            raise ValueError(f'unknown task intent {intent!r}')
        ref = self._check_control(target)
        up = self.encode_joints(q_prev)  # feasible warm anchor
        J, A, rhs = self._features(up, ref, intent, task_targets_m)
        # Reachable 6D linear target, limited to admissible physical command box.
        full_delta = np.linalg.lstsq(A, rhs, rcond=None)[0]
        goal = np.clip(up + full_delta, self.lower, self.upper)
        reachable_desired = A @ (goal - up)
        unreachable = float(np.linalg.norm(rhs - reachable_desired))
        V = self._basis(J, goal - up)
        eps = max(self.projection_abs_m,
                  self.projection_relative * float(np.linalg.norm(reachable_desired)))
        k, certificate = self._choose_dimension(A, reachable_desired, V, eps)
        primary = V[:, :k]
        x_primary = np.linalg.lstsq(primary, goal - up, rcond=None)[0]
        if not self._feasible(up + primary @ x_primary):
            # Safe interpolation toward feasible anchor; avoids setting x to
            # an infeasible projection before constrained optimization.
            alpha = 1.
            du = primary @ x_primary
            for i in range(6):
                if du[i] > 0:
                    alpha = min(alpha, (self.upper[i] - up[i]) / du[i])
                elif du[i] < 0:
                    alpha = min(alpha, (self.lower[i] - up[i]) / du[i])
            x_primary *= np.clip(alpha, 0, 1) * .9999
        x_best, result, loss_best = self._stage_solve(
            objective, up, ref, primary, k, 0., x_primary, ftol)
        D_best = primary
        x_stage = x_best
        active_indices: list[int] = []
        task_err = self._linear_error(A, reachable_desired, D_best @ x_best)
        stage = 1
        cap = max(0., self.residual_caps.get(intent.value, 0.12))
        # Only truly needed residual coordinates are promoted: 1, then 2, ...
        if cap > 0 and (task_err > max(eps, self.rescue_threshold_m)
                            or self._contact_rescue_needed()):
            for _ in range(self._allowed_residual_axes(k, intent)):
                remainder = [i for i in range(k, 6) if i not in active_indices]
                if not remainder:
                    break
                residual = reachable_desired - A @ (D_best @ x_best)
                # Prefer complement direction with largest predicted task error
                # reduction; a small epsilon prevents poorly scaled choices.
                scores = [abs(float((A @ V[:, i]) @ residual)) /
                          max(float(np.linalg.norm(A @ V[:, i])), 1e-9)
                          for i in remainder]
                idx = remainder[int(np.argmax(scores))]
                new_indices = active_indices + [idx]
                D_try = np.column_stack((primary, V[:, new_indices]))
                x_try, result_try, loss_try = self._stage_solve(
                    objective, up, ref, D_try, k, cap,
                    np.concatenate((x_best, np.zeros(1))), ftol)
                e_try = self._linear_error(A, reachable_desired, D_try @ x_try)
                if (self._feasible(up + D_try @ x_try) and
                    loss_try <= loss_best + 1e-9 and
                    e_try < task_err - self.rescue_min_improvement_m):
                    active_indices, D_best, x_best = new_indices, D_try, x_try
                    task_err, loss_best, result = e_try, loss_try, result_try
                    stage = 2
                else:
                    break
                if task_err <= max(eps, self.rescue_threshold_m):
                    break
        u = up + D_best @ x_best
        if not self._feasible(u):  # Defensive: no independent joint clipping.
            u = up.copy()
            D_best = primary
            x_best = np.zeros(k)
            active_indices = []
            stage = 1
        # Optional full-reference performance shield: guarantees only that the
        # chosen command is no worse under THIS heuristic nonlinear objective
        # than the feasible direct coupled 6D reference (not real grasp success).
        # Reference is evaluated once and NOT numerically optimized.
        fallback_used = False
        if (self.fallback_full_reference and
            float(objective(self.C @ ref)) + self.fallback_min_gain <
                float(objective(self.C @ u))):
            u = ref.copy()
            D_best = np.eye(6)
            x_best = ref - up
            active_indices = []
            stage = 3
            fallback_used = True
        self.active_basis = D_best.copy()
        self.active_anchor = up.copy()
        self.active_residual_start = 6 if fallback_used else k
        self.active_cap = cap
        diag = ReachabilityDiagnostics(
            dimension=k, projection_error_m=certificate,
            residual_norm=(0. if fallback_used else float(np.linalg.norm(x_best[k:]))),
            residual_cap=cap, feasible=self._feasible(u),
            optimized_variables=(k if fallback_used else int(D_best.shape[1])),
            active_residuals=len(active_indices), stage=stage,
            task_linear_error_m=float(self._linear_error(A, reachable_desired, u - up)),
            task_full_approx_error_m=unreachable,
            nonlinear_objective=float(objective(self.C @ u)),
            fallback_full_reference=fallback_used,
            command_subspace_rank=int(D_best.shape[1]),
        )
        self.last_diag = diag
        return self.C @ u, result, diag

    def _project_to_active_subspace(self, q_target: np.ndarray) -> np.ndarray:
        if self.active_basis is None or self.active_anchor is None:
            return self.encode_joints(q_target)
        target = self.encode_joints(q_target)
        D = self.active_basis
        anchor = self.active_anchor
        k = self.active_residual_start
        cap = self.active_cap
        weighted = self.C @ D
        y = self.C @ (target - anchor)
        x_guess = np.linalg.lstsq(weighted, y, rcond=None)[0]
        x_guess[k:] = np.clip(x_guess[k:], -cap, cap)
        def u_of(x):
            return anchor + D @ x
        if self._feasible(u_of(x_guess)):
            return u_of(x_guess)
        if minimize is None:  # ensure safe command without scipy
            return anchor.copy()
        def fun(x):
            r = weighted @ x - y
            return .5 * float(r @ r)
        sol = minimize(fun, np.zeros(D.shape[1]), method='SLSQP',
                       jac=lambda x: weighted.T @ (weighted @ x - y),
                       bounds=[(None, None)] * k + [(-cap, cap)] * (D.shape[1] - k),
                       constraints=[
                           {'type': 'ineq', 'fun': lambda x: u_of(x) - self.lower,
                            'jac': lambda x: D},
                           {'type': 'ineq', 'fun': lambda x: self.upper - u_of(x),
                            'jac': lambda x: -D}],
                       options={'maxiter': 60, 'ftol': 1e-11, 'disp': False})
        return u_of(sol.x) if self._feasible(u_of(sol.x)) else anchor.copy()
