"""Scheme D: reliability-weighted task synergies with feedback-gated sparse rescue.

IMPORTANT SCOPE: local risk bounds are mathematically valid only for a locally
linearized hand model AND if the supplied covariance is an upper bound on
zero-mean random input errors. The bundled pose-derived proxy DOES NOT verify
those assumptions, so all bound values are *conditional*, not hardware safety
certificates. Nonlinear verification below checks the actual FK model once.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np

from src.features.intent_recognition import Intent
from .reachability_synergy import ReachabilityGatedSynergyMapper, ReachabilityDiagnostics
from .uncertainty_model import EncoderUncertainty


@dataclass(frozen=True)
class CredibleContactDiagnostics(ReachabilityDiagnostics):
    # Conditional linearized second-moment bound on task-space deviation.
    conditional_risk_bound_m: float = 0.0
    risk_budget_m: float = 0.0
    risk_feasible: bool = False
    contact_phase: str = 'UNOBSERVED'
    sensor_uncertainty_trace: float = 0.0
    nonlinear_tip_error_m: float = 0.0
    nonlinear_reference_error_m: float = 0.0
    nonlinear_guard_used: bool = False
    uncertainty_hold_used: bool = False
    risk_policy_enforced: bool = False


class CredibleContactSynergyMapper(ReachabilityGatedSynergyMapper):
    """A task-error/risk minimum-rank selector plus contact-conditioned rescue.

    The 6D software command space is a hard physical design assumption.
    An ``observed`` flag distinguishes absent contact telemetry from no contact.
    """

    def __init__(self, base, config: dict) -> None:
        super().__init__(base, config)
        self.risk_policy = str(config.get('risk_policy', 'diagnostic'))
        if self.risk_policy not in ('diagnostic', 'enforce'):
            raise ValueError('risk_policy must be diagnostic or enforce')
        self.risk_alpha = float(config.get('risk_alpha', 0.10))
        self.risk_budget_m = float(config.get('risk_budget_m', 0.025))
        self.risk_contact_budget_m = float(config.get('risk_contact_budget_m', 0.020))
        self.risk_basis_weight = float(config.get('risk_basis_weight', 3.0))
        self.guard_regret_m = float(config.get('guard_regret_m', 0.001))
        self.hard_hold_risk_m = float(config.get('hard_hold_risk_m', 0.))
        self.contact_dwell = int(config.get('contact_dwell_frames', 2))
        self.contact_loss_dwell = int(config.get('contact_loss_frames', 2))
        self.contact_rescue_frames = int(config.get('contact_rescue_frames', 4))
        self.residual_in_secure_contact = bool(config.get('residual_in_secure_contact', False))
        self.uncertainty_enabled = bool(config.get('uncertainty_enabled', True))
        self.contact_feedback_enabled = bool(config.get('contact_feedback_enabled', True))
        self.nonlinear_guard_enabled = bool(config.get('nonlinear_guard_enabled', True))
        self.uncertainty = EncoderUncertainty(
            self, config.get('uncertainty_file'), float(config.get('control_sigma_deg', 2.)))
        if (not 0 < self.risk_alpha < 1 or min(self.risk_budget_m,
              self.risk_contact_budget_m, self.risk_basis_weight,
              self.guard_regret_m, self.hard_hold_risk_m) < 0 or
              min(self.contact_dwell, self.contact_loss_dwell) < 1 or
              self.contact_rescue_frames < 0):
            raise ValueError('invalid Scheme D uncertainty/contact settings')
        self.sigma_u = self.uncertainty.control_cov.copy()
        self.contact_phase = 'UNOBSERVED'
        self._contact_seen = False
        self._stable_count = 0
        self._lost_count = 0
        self._rescue_remaining = 0
        self._conditional_bound = float('nan')
        self._confidence_feasible = False
        self._observed_tip_flags: dict[str, bool] = {}

    def controls_from_observation(self, sensors_deg, controls_deg, source='sensor',
                                  sensor_weight: float = 0.8):
        self.sigma_u = (self.uncertainty.covariance(
            sensors_deg, controls_deg, source=source, blend_weight=sensor_weight)
            if self.uncertainty_enabled else np.zeros((6, 6)))
        return super().controls_from_observation(
            sensors_deg, controls_deg, source=source, sensor_weight=sensor_weight)

    def update_contact(self, metrics: Mapping | None, *, task_active: bool,
                       observed: bool = True) -> str:
        """Use actual previous-frame telemetry, with secure/loss hysteresis.

        No telemetry -> UNOBSERVED, NEVER inferred as 'zero physical contact'.
        This method must not be fed synthetic or future contact labels.
        """
        if not self.contact_feedback_enabled or not observed or metrics is None:
            self.contact_phase = 'UNOBSERVED'
            self._contact_seen = False
            self._observed_tip_flags = {}
            self._stable_count = self._lost_count = self._rescue_remaining = 0
            return self.contact_phase
        if not task_active:
            self.contact_phase = 'FREE'
            self._contact_seen = False
            self._observed_tip_flags = {}
            self._stable_count = self._lost_count = self._rescue_remaining = 0
            return self.contact_phase
        if self.contact_phase in ('UNOBSERVED', 'FREE'):
            self.contact_phase = 'APPROACH'
        flags = {f: bool(metrics.get(f'{f}_tip_contact', False)) for f in
                 ('thumb', 'index', 'middle', 'ring', 'little')}
        self._observed_tip_flags = flags
        proxy_stable = bool(metrics.get('stable_contact_proxy', False))
        observed_contact = any(flags.values()) or bool(metrics.get('finger_contact', False))
        self._contact_seen |= observed_contact
        if proxy_stable and observed_contact:
            self._stable_count += 1
            self._lost_count = 0
            if self._stable_count >= self.contact_dwell:
                self.contact_phase = 'SECURE'
                self._rescue_remaining = 0
        else:
            self._stable_count = 0
            if self.contact_phase in ('SECURE', 'LOSS_RISK'):
                self._lost_count += 1
                if self._lost_count == self.contact_loss_dwell:
                    self.contact_phase = 'LOSS_RISK'
                    self._rescue_remaining = self.contact_rescue_frames
            elif self._rescue_remaining > 0:
                self.contact_phase = 'LOSS_RISK'
            else:
                self.contact_phase = 'APPROACH'
        return self.contact_phase

    def _features(self, up, reference, intent, task_targets_m):
        J, A, rhs = super()._features(up, reference, intent, task_targets_m)
        if self.contact_phase not in ('APPROACH', 'LOSS_RISK'):
            return J, A, rhs
        fingers = self.FINGERS[intent]
        weights = np.ones(len(fingers))
        # Do not assume a contact that was never measured. Once contact exists,
        # prioritize uncontacted fingers during a grasp attempt.
        if self._contact_seen:
            weights = np.asarray([.65 if self._observed_tip_flags.get(f, False)
                                  else 1.5 for f in fingers], dtype=float)
        elif self.contact_phase == 'LOSS_RISK':
            weights[:] = 1.4
        scale = np.repeat(np.sqrt(weights), 3)
        J = scale[:, None] * J
        A = A.copy()
        A[:len(scale)] *= scale[:, None]
        rhs = rhs.copy()
        rhs[:len(scale)] *= scale
        return J, A, rhs

    def _basis(self, J, direction):
        # The C scheme's eigenspace carries the model-informed task and PCA
        # priors. Penalize low-SNR directions via a symmetric uncertainty
        # whitening, then sort by conditioned task utility. NOT a new PCA fit.
        variance = np.maximum(np.diag(self.sigma_u), 0)
        weight = self.risk_basis_weight if self.risk_policy == 'enforce' else 0.
        scale = 1. / np.sqrt(1. + weight *
                             variance / max(float(np.median(variance)), 1e-7))
        H = J.T @ J + self.task_prior_weight_m**2 * (
            (self.B * np.maximum(np.asarray(
                self.base.params.get('explained_variance_ratio', [1]*6), float),
                1e-12)[None, :]) @ self.B.T)
        norm = float(direction @ direction)
        if norm > 1e-12:
            H += self.task_direction_weight_m**2 * np.outer(direction, direction) / norm
        H = (scale[:, None] * H) * scale[None, :]
        _, V = np.linalg.eigh(H)
        return V[:, ::-1].copy()

    @staticmethod
    def conditional_bound(A, V, k, covariance, desired, alpha):
        """||projection residual|| + sqrt(tr(M Sigma M.T) / alpha).

        For *zero-mean* error with true covariance Sigma, Chebyshev/Markov
        gives >= 1-alpha probability in the LINEAR model. Supplied covariance
        is empirical and the nonlinear FK is not covered by that guarantee.
        """
        Vk = V[:, :k]
        Ak = A @ Vk
        projected = Ak @ np.linalg.lstsq(Ak, desired, rcond=None)[0]
        err = float(np.linalg.norm(projected - desired))
        P = Vk @ Vk.T
        M = A @ P
        variance = max(float(np.trace(M @ covariance @ M.T)), 0.)
        return err, err + np.sqrt(variance / alpha)

    def _choose_dimension(self, A, desired, V, eps):
        contact_task = self.contact_phase in ('APPROACH', 'LOSS_RISK', 'SECURE')
        budget = (self.risk_contact_budget_m if contact_task else self.risk_budget_m)
        if self.risk_policy == 'diagnostic':
            # Bundled pose-group covariance cannot justify intervention.
            # Still record what its conditional bound would predict.
            k, error = super()._choose_dimension(A, desired, V, eps)
            _, bound = self.conditional_bound(
                A, V, k, self.sigma_u, desired, self.risk_alpha)
            self._conditional_bound = bound
            self._confidence_feasible = bound <= budget
            return k, error
        best = None
        for k in range(self.k_min, self.k_max + 1):
            error, bound = self.conditional_bound(
                A, V, k, self.sigma_u, desired, self.risk_alpha)
            if best is None or bound < best[2]:
                best = (k, error, bound)
            if error <= eps and bound <= budget:
                self._conditional_bound = bound
                self._confidence_feasible = True
                return k, error
        # No candidate meets the *conditional* budget: choose the minimum-risk
        # dimension and explicitly mark that no confidence criterion was met.
        k, err, bound = best
        self._conditional_bound = bound
        self._confidence_feasible = False
        return k, err

    def _contact_rescue_needed(self):
        return self.contact_phase == 'LOSS_RISK' and self._rescue_remaining > 0

    def _allowed_residual_axes(self, k, intent):
        if self.contact_phase == 'SECURE' and not self.residual_in_secure_contact:
            return 0
        return super()._allowed_residual_axes(k, intent)

    def _nonlinear_tip_error(self, q, task_targets_m, intent):
        fingers = self.FINGERS[intent]
        targets = (self.robot.fingertips(self.C @ self._last_reference)
                   if task_targets_m is None else task_targets_m)
        actual = self.robot.fingertips(q)
        return float(np.sqrt(np.mean([np.sum((actual[f] - targets[f])**2)
                                      for f in fingers])))

    def _project_to_active_subspace(self, q_target):
        if self.active_basis is not None and self.active_basis.shape[1] == 0:
            return self.active_anchor.copy()
        return super()._project_to_active_subspace(q_target)

    def optimize(self, objective, q_prev, intent, target, *, ftol=1e-6,
                 task_targets_m=None):
        self._last_reference = self._check_control(target)
        q_candidate, result, diag = super().optimize(
            objective, q_prev, intent, target, ftol=ftol,
            task_targets_m=task_targets_m)
        q_reference = self.decode_controls(self._last_reference)
        err_ref = self._nonlinear_tip_error(q_reference, task_targets_m, intent)
        err_candidate = self._nonlinear_tip_error(q_candidate, task_targets_m, intent)
        guard = False
        hold = False
        # Exact FK verification relative to a feasible, *unoptimized* reference
        # (not an accuracy guarantee relative to the human hand itself).
        if (self.nonlinear_guard_enabled and
            err_candidate > err_ref + self.guard_regret_m):
            q_candidate, err_candidate = q_reference.copy(), err_ref
            guard = True
        if (self.risk_policy == 'enforce' and self.hard_hold_risk_m > 0 and
            self._conditional_bound > self.hard_hold_risk_m and
            self.contact_phase != 'SECURE'):
            q_candidate = self.decode_controls(self.encode_joints(q_prev))
            err_candidate = self._nonlinear_tip_error(q_candidate, task_targets_m, intent)
            hold = True
        if hold or guard:
            self.active_anchor = self.encode_joints(q_prev)
            self.active_basis = (np.zeros((6, 0)) if hold else np.eye(6))
            self.active_residual_start = (0 if hold else 6)
            self.active_cap = 0.
        if self._rescue_remaining > 0:
            self._rescue_remaining -= 1
            if self._rescue_remaining == 0 and self.contact_phase == 'LOSS_RISK':
                self.contact_phase = 'APPROACH'
        output_diag = CredibleContactDiagnostics(
            **{k: getattr(diag, k) for k in ReachabilityDiagnostics.__dataclass_fields__},
            conditional_risk_bound_m=float(self._conditional_bound),
            risk_budget_m=(self.risk_contact_budget_m if self.contact_phase in
                           ('APPROACH', 'SECURE', 'LOSS_RISK') else self.risk_budget_m),
            risk_feasible=self._confidence_feasible,
            contact_phase=self.contact_phase,
            sensor_uncertainty_trace=float(np.trace(self.sigma_u)),
            nonlinear_tip_error_m=err_candidate,
            nonlinear_reference_error_m=err_ref,
            nonlinear_guard_used=guard,
            uncertainty_hold_used=hold,
            risk_policy_enforced=self.risk_policy == 'enforce')
        if hold or guard:
            output_diag = replace(
                output_diag, stage=(5 if hold else 4),
                command_subspace_rank=(0 if hold else 6),
                feasible=self._feasible(self.encode_joints(q_candidate)),
                nonlinear_objective=float(objective(q_candidate)))
        self.last_diag = output_diag
        return q_candidate, result, output_diag
