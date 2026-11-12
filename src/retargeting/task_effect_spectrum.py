"""Scheme E: task-effect quotient spectrum with an auditable local certificate.

Distinctive design: select directions by *current desired task effect*, rather
than by PCA pose variance or largest singular value.  For the dictionary of
weighted task singular vectors the mode-energy identity gives the exact
best k-mode approximation of the local linear task displacement.  Nonlinear
FK and FINAL rate-limited commands are verified against a feasible reference.

Mathematical scope: the spectrum theorem applies to the local LINEAR model
and mode subsets from this particular dictionary; it is not a global minimum-
DoF result. A probabilistic/ellipsoidal statement needs separately calibrated
sensor noise. We do not claim independent mechanical actuation of 11 joints.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import numpy as np
try:
    from scipy.optimize import minimize as _minimize
except ImportError:  # pragma: no cover
    _minimize = None

from .credible_contact_synergy import CredibleContactSynergyMapper, CredibleContactDiagnostics


@dataclass(frozen=True)
class TaskEffectDiagnostics(CredibleContactDiagnostics):
    selected_mode_indices: tuple[int, ...] = ()
    spectral_omission_m: float = float('nan')
    spectral_ellipsoid_risk_m: float = float('nan')
    selected_energy_fraction: float = 0.0
    nonlinear_feasible_reference_m: float = float('nan')
    nonlinear_candidate_m: float = float('nan')
    final_guard_used: bool = False
    final_tip_error_m: float = float('nan')
    final_reference_error_m: float = float('nan')
    baseline_regret_m: float = float('nan')
    rate_box_valid: bool = False
    # Diagnostic fields consumed by src.main; never claim an unverified object-space value.
    final_guard_reason: int = 0
    final_object_task_error_m: float = float('nan')
    final_object_baseline_m: float = float('nan')
    # Explicitly distinguish selector request, accepted modes and emitted rank.
    # Older `spectrum_modes` was wired to command_subspace_rank and therefore
    # overstated the number of selected modes whenever the algorithm fell back.
    requested_mode_count: int = 0
    accepted_mode_count: int = 0
    output_control_rank: int = 0
    fallback_reason: str = ''
    final_guard_enabled: bool = True
    final_guard_regret_m: float = float('nan')
    candidate_attempts: int = 0
    reject_spectral_count: int = 0
    reject_fidelity_count: int = 0
    reject_nonlinear_count: int = 0
    candidate_min_fk_error_m: float = float('nan')
    candidate_min_regret_m: float = float('nan')
    candidate_max_fidelity_excess_deg: float = float('nan')
    selection_status: str = 'not_evaluated'


def effect_spectrum(A: np.ndarray, desired: np.ndarray, metric: np.ndarray):
    """Return task-gain ordered spectral dictionary and exact omission energies.

    Let W=metric SPD, A W^{-1/2}=U Sigma V^T and d_i=W^{-1/2}v_i.
    The task images A d_i are mutually orthogonal. For any subset S, the
    minimum linear residual squared equals
        ||b||^2 - sum_{i in S} (u_i^T b)^2.
    Therefore the k modes with the greatest task gains are optimal among all
    k-element subsets OF THIS FIXED DICTIONARY. Not among arbitrary subspaces.
    """
    A = np.asarray(A, float)
    desired = np.asarray(desired, float)
    W = np.asarray(metric, float)
    if A.ndim != 2 or desired.shape != (A.shape[0],) or W.shape != (A.shape[1], A.shape[1]):
        raise ValueError('incompatible spectral dimensions')
    if not np.isfinite(A).all() or not np.isfinite(desired).all() or not np.isfinite(W).all():
        raise ValueError('nonfinite spectral argument')
    ev, Q = np.linalg.eigh((W + W.T) / 2)
    if np.min(ev) <= 0:
        raise ValueError('metric must be positive definite')
    Winvhalf = (Q / np.sqrt(ev)[None, :]) @ Q.T
    U, singular, Vt = np.linalg.svd(A @ Winvhalf, full_matrices=False)
    # For m>=n, returns exactly n control directions. Training configurations
    # add a small command preservation term so m>=6.
    if len(singular) != A.shape[1]:
        raise ValueError('task matrix must have at least as many rows as controls')
    if np.min(singular) <= 1e-10:
        raise ValueError('task map must have full column rank; add a control-guide term')
    D = Winvhalf @ Vt.T
    coefficients = U.T @ desired
    gains = coefficients**2
    order = np.argsort(-gains, kind='stable')
    total = float(desired @ desired)
    omission = np.sqrt(np.maximum(0., total - np.cumsum(gains[order])))
    return D[:, order], singular[order], coefficients[order], gains[order], omission, order


def ray_to_control_box(anchor: np.ndarray, delta: np.ndarray,
                       lower: np.ndarray, upper: np.ndarray):
    """Maximum feasible line segment on anchor -> anchor+delta inside box."""
    anchor = np.asarray(anchor, float)
    delta = np.asarray(delta, float)
    if not (anchor.shape == delta.shape == lower.shape == upper.shape):
        raise ValueError('invalid box shape')
    if np.any(anchor < lower - 1e-8) or np.any(anchor > upper + 1e-8):
        raise ValueError('anchor must already be feasible')
    alpha = 1.0
    for i in range(len(delta)):
        if delta[i] > 1e-12:
            alpha = min(alpha, max(0.0, (upper[i] - anchor[i]) / delta[i]))
        elif delta[i] < -1e-12:
            alpha = min(alpha, max(0.0, (lower[i] - anchor[i]) / delta[i]))
    return np.clip(anchor + np.clip(alpha, 0., 1.) * delta, lower, upper), float(np.clip(alpha, 0., 1.))


class TaskEffectSpectrumMapper(CredibleContactSynergyMapper):
    """A task-mode spectral selector with post-processing feasibility shield.

    No generic high-dimensional nonlinear solver in the normal path. Selecting
    the active mode count uses at most six 6x6 SVD coordinates, with feasible
    commands derived by an analytic radial line search. Existing joint-
    coupling/saturation and observed-contact interfaces are retained.
    """

    def __init__(self, base, config: dict):
        # Same adapter, uncertainty module and prior-frame contact interface as D.
        super().__init__(base, config)
        if np.any(np.count_nonzero(np.abs(self.C)>1e-12,axis=1)>1):
            raise ValueError('joint-wise independent rate box requires at most one command per joint')
        self.k_min = int(config.get('spectral_k_min', 1))
        self.k_max = int(config.get('spectral_k_max', 6))
        self.metric_prior = float(config.get('metric_prior', 0.12))
        self.metric_noise = float(config.get('metric_noise', 0.0))
        self.metric_floor = float(config.get('metric_floor', 0.05))
        self.max_ref_regret_m = float(config.get('max_reference_regret_m', 0.0001))
        self.encoder_fidelity_slack_deg = float(config.get('encoder_fidelity_slack_deg', 1.5))
        self.linear_ref_weight_m = float(config.get('linear_ref_weight_m', 0.012))
        self.refine_mode_qp = bool(config.get('refine_mode_qp', False))
        self.force_feasible_baseline = bool(config.get('force_feasible_baseline', False))
        self.fixed_spectral_modes = config.get('fixed_spectral_modes')
        self.final_nonlinear_guard_enabled = bool(config.get('final_nonlinear_guard_enabled', True))
        if self.fixed_spectral_modes is not None:
            self.fixed_spectral_modes = int(self.fixed_spectral_modes)
            if not 1 <= self.fixed_spectral_modes <= 6:
                raise ValueError('fixed_spectral_modes must be in [1, 6]')
        self.ellipsoid_radius = float(config.get('ellipsoid_radius', 2.0))
        self.require_calibrated_risk = bool(config.get('require_calibrated_risk', True))
        self.noise_independently_calibrated = bool(config.get('noise_independently_calibrated', False))
        if not 1 <= self.k_min <= self.k_max <= 6:
            raise ValueError('invalid spectral dimensions')
        if min(self.metric_prior, self.metric_noise, self.metric_floor,
               self.max_ref_regret_m, self.ellipsoid_radius,
               self.encoder_fidelity_slack_deg, self.linear_ref_weight_m) < 0:
            raise ValueError('spectral penalties and radii must be nonnegative')
        if (self.risk_policy == 'enforce' and self.require_calibrated_risk and
                not self.noise_independently_calibrated):
            raise ValueError('risk enforcement requires independently calibrated covariance')
        self._anchor_cmd = None
        self._max_step_control = None
        self._target_tips = None
        self._intent = None
        self._baseline_raw = None
        self._reference_u = None
        self._diagnostics = None
        self._object_task_targets = None

    def set_object_task_goal(self, fingertip_provider, task_targets):
        """Receive optional MuJoCo contact goals for later diagnostics.

        World-frame MuJoCo targets cannot be compared blindly with the local
        kinematic model, so do not claim an object-space error certificate.
        Object-space diagnostic fields remain NaN until frames are aligned.
        """
        self._object_task_targets = (None if task_targets is None else
                                     {name: np.asarray(value, dtype=float).copy()
                                      for name, value in task_targets.items()})

    def set_command_anchor(self, previous_command, max_speed_rad_s, dt_s):
        """Called before solve. Uses previous *commanded* q, not lagging qactual."""
        if previous_command is None:
            self._anchor_cmd = None
        else:
            previous_command = np.asarray(previous_command, float)
            self._anchor_cmd = self.encode_joints(previous_command)
            if not np.allclose(self.decode_controls(self._anchor_cmd), previous_command, atol=1e-6):
                raise ValueError('the previous command must be coupled')
        v = np.asarray(max_speed_rad_s, float)
        if v.shape != (11,) or np.any(v <= 0) or dt_s <= 0:
            raise ValueError('invalid speed limits')
        # CH-M6 fixed C: each joint depends on exactly one software control.
        limits = np.full(6, np.inf)
        for i in range(6):
            idx = np.flatnonzero(np.abs(self.C[:, i]) > 1e-12)
            limits[i] = np.min(v[idx] * dt_s / np.abs(self.C[idx, i]))
        self._max_step_control = limits

    def _speed_bounds(self, anchor):
        if self._anchor_cmd is None or self._max_step_control is None:
            return self.lower.copy(), self.upper.copy()
        return (np.maximum(self.lower, self._anchor_cmd - self._max_step_control),
                np.minimum(self.upper, self._anchor_cmd + self._max_step_control))

    def _metric(self):
        # Penalize PCA directions in inverse proportion to their pose frequency,
        # but never use PCA variance itself to select task modes.
        v = np.asarray(self.base.params.get('explained_variance_ratio', [1.]*6), float)
        if v.shape != (6,):
            v = np.ones(6)
        v = v / max(float(np.max(v)), 1e-12)
        penalties = np.clip(1 / (v + self.metric_floor) - 1., 0., 8.)
        W = np.eye(6) + self.metric_prior * ((self.B * penalties[None, :]) @ self.B.T)
        # Proxy noise must not silently alter control. Only enabled explicitly.
        if self.metric_noise > 0:
            if not self.noise_independently_calibrated:
                raise ValueError('noise metric requires independently calibrated covariance')
            scale = max(float(np.trace(self.sigma_u))/6, 1e-8)
            W += self.metric_noise * self.sigma_u / scale
        return (W + W.T)/2

    def _minimum_risk_subset(self, A, rhs, D, gains, metric, covariance, threshold, budget):
        """Exact minimum-cardinality robust mode subset (at most 2**6-1).

        Within THIS fixed W-orthogonal spectral dictionary, enumerate all
        subsets of sizes kmin..kmax. An accepted subset has an exact local
        linear truncation residual + deterministic ellipsoid perturbation
        upper bound below the budget. No guarantee outside the ellipsoid and
        no guarantee for the nonlinear dynamics without a separate guard.
        """
        total=float(rhs@rhs)
        for k in range(self.k_min,self.k_max+1):
            possibilities=[]
            for indices in combinations(range(6),k):
                missing=max(0.,total-float(np.sum(gains[list(indices)])))
                omit=np.sqrt(missing)
                if omit>threshold+1e-12: continue
                risk=self._spectral_risk(A,D[:,indices],metric,covariance)
                if omit+risk<=budget+1e-12:
                    possibilities.append((omit+risk,omit,indices,risk))
            if possibilities:
                _,omit,indices,risk=min(possibilities,key=lambda z:(z[0],z[1]))
                return tuple(indices),float(omit),float(risk)
        return None

    def _refine_in_modes(self, anchor, baseline, Dk, A, rhs, lo, hi, x0):
        """Small convex local QP; exactly k numerical unknowns.

        This refinement handles input/velocity bounds without clipping each
        joint independently (which would destroy the selected task subspace).
        The mathematical spectral omission formula remains an unconstrained
        lower bound, and nonlinear FK is ALWAYS checked afterward.
        """
        if _minimize is None or not self.refine_mode_qp:
            return ray_to_control_box(anchor, Dk @ x0, lo, hi)[0]
        M=A@Dk
        f=self.linear_ref_weight_m
        P=np.vstack((M, f*Dk))
        v=np.concatenate((rhs, f*(baseline-anchor)))
        H=P.T@P+1e-9*np.eye(Dk.shape[1])
        g=P.T@v
        def loss(x): return .5*float(x@H@x)-float(g@x)
        def jac(x): return H@x-g
        zero=np.zeros(Dk.shape[1])
        out=_minimize(loss,zero,jac=jac,method='SLSQP',
           constraints=[{'type':'ineq','fun':lambda x: anchor+Dk@x-lo,
                         'jac':lambda x:Dk},
                        {'type':'ineq','fun':lambda x: hi-anchor-Dk@x,
                         'jac':lambda x:-Dk}],
           options={'maxiter':12,'ftol':1e-10,'disp':False})
        trial=anchor+Dk@np.asarray(out.x,float)
        if np.isfinite(trial).all() and np.all(trial>=lo-1e-7) and np.all(trial<=hi+1e-7):
            return np.clip(trial,lo,hi)
        return ray_to_control_box(anchor,Dk@x0,lo,hi)[0]

    def _fidelity_ok(self, candidate, baseline, reference):
        """A component-wise relative error budget in source CONTROL DEGREES.

        Absolute errors versus labels cannot be bounded without ground truth.
        This enforces adherence to the sensor encoder versus a feasible 6D
        reference, preventing task-space optimization from overwriting the
        calibrated finger controls indiscriminately.
        """
        extra = self.encoder_fidelity_slack_deg
        return bool(np.all(np.abs(candidate - reference)*self.limits_deg <=
                   np.abs(baseline-reference)*self.limits_deg + extra + 1e-9))

    def _spectral_risk(self, A, selected_basis, metric, covariance):
        """Ellipsoid worst-case task perturbation rho ||A P Sigma^1/2||2.

        P is the W-orthogonal projector on the selected mode span.
        Valid only if the ACTUAL input error belongs to the stated ellipsoid.
        """
        P = selected_basis @ selected_basis.T @ metric
        e, Q = np.linalg.eigh((covariance + covariance.T)/2)
        sqrt_cov = (Q * np.sqrt(np.maximum(e, 0.))[None, :]) @ Q.T
        return float(self.ellipsoid_radius * np.linalg.norm(A @ P @ sqrt_cov, 2))

    def optimize(self, objective, q_prev, intent, target, *, ftol=1e-6, task_targets_m=None):
        if intent not in self.FINGERS:
            raise ValueError('unsupported intent')
        reference = self._check_control(target)
        anchor = (self._anchor_cmd.copy() if self._anchor_cmd is not None
                  else self.encode_joints(q_prev))
        lo, hi = self._speed_bounds(anchor)
        self._last_reference = reference.copy()
        self._reference_u = reference.copy()
        self._intent = intent
        self._target_tips = task_targets_m
        J, A, desired = self._features(anchor, reference, intent, task_targets_m)
        # The human and CH-M6 task spaces are not necessarily mutually
        # reachable. Select modes against a LOCAL FEASIBLE FULL-6D TASK
        # DISPLACEMENT, not the large, inherently unattainable human residual.
        # This distinction is essential to a meaningful spectral certificate.
        # The input encoder's feasible reference is a calibration anchor;
        # use a BOX projection (component-wise) rather than uniform global
        # interpolation so no finger loses movement due to another finger.
        baseline = np.clip(reference, lo, hi)
        attainable_rhs = A @ (baseline - anchor)
        W = self._metric()
        D, sing, coeff, gains, omissions, original_indices = effect_spectrum(
            A, attainable_rhs, W)
        self._baseline_raw = baseline.copy()
        base_tip = self._nonlinear_tip_error(self.decode_controls(baseline), task_targets_m, intent)
        threshold = max(self.projection_abs_m,
                        self.projection_relative * float(np.linalg.norm(attainable_rhs)))
        selected = self.k_max
        chosen_risk = 0.
        self._conditional_subset_unavailable = False
        # The first decision uses an EXACT projection residual for the LOCAL
        # linear model, not a heuristic PCA retained variance percentage.
        if self.risk_policy == 'enforce':
            robust=self._minimum_risk_subset(
                A,attainable_rhs,D,gains,W,self.sigma_u,
                threshold,self.risk_budget_m)
            if robust is not None:
                subset,_,chosen_risk=robust
                # Reorder the remaining directions only for a later rescue;
                # the first |S| columns are the certified rank-minimum subset.
                remaining=[i for i in range(6) if i not in subset]
                order=list(subset)+remaining
                D, sing, coeff, gains, original_indices = (
                    D[:,order],sing[order],coeff[order],gains[order],
                    original_indices[order])
                selected=len(subset)
                omissions=np.sqrt(np.maximum(0.,float(attainable_rhs@attainable_rhs)-
                                           np.cumsum(gains)))
            else:
                # No conditional certificate: diagnostic failure is explicit.
                # Do not silently assert a probabilistic guarantee.
                self._conditional_subset_unavailable = True
                chosen_risk=self._spectral_risk(
                    A,D[:,:selected],W,self.sigma_u)
        else:
            if self.fixed_spectral_modes is not None:
                # A true fixed-k ablation: no dynamic selector and no later
                # rank rescue. It shares the exact E solver/shields.
                selected = self.fixed_spectral_modes
            else:
                for k in range(self.k_min, self.k_max+1):
                    if omissions[k-1] <= threshold:
                        selected=k
                        break
            chosen_risk=self._spectral_risk(
                A,D[:,:selected],W,self.sigma_u)

        # Accept only candidates satisfying ALL declared certificates.  The
        # previous implementation could mark an improving candidate as
        # accepted before checking its spectral omission budget.  That mixed
        # "better than baseline" with "certified under the selected k".
        best = baseline.copy()
        best_err = base_tip
        best_k = 0
        accepted_k = None
        candidate_attempts = 0
        reject_spectral_count = 0
        reject_fidelity_count = 0
        reject_nonlinear_count = 0
        candidate_errors: list[float] = []
        candidate_regrets: list[float] = []
        fidelity_excesses: list[float] = []
        candidate_dimensions = ([selected] if self.fixed_spectral_modes is not None
                                else range(selected, self.k_max + 1))
        for k in candidate_dimensions:
            candidate_attempts += 1
            x = np.divide(coeff[:k], sing[:k], out=np.zeros(k),
                          where=np.abs(sing[:k]) > 1e-10)
            proposal = self._refine_in_modes(
                anchor, baseline, D[:, :k], A, attainable_rhs, lo, hi, x)
            actual_err = self._nonlinear_tip_error(
                self.decode_controls(proposal), task_targets_m, intent)
            regret = float(actual_err - base_tip)
            candidate_errors.append(float(actual_err))
            candidate_regrets.append(regret)

            # Constraint and certificate checks are reported independently.
            spectral_ok = bool(omissions[k - 1] <= threshold + 1e-12 or k == self.k_max)
            trusted = self._fidelity_ok(proposal, baseline, reference)
            if reference.size:
                fidelity_excess = float(np.max(
                    np.abs(proposal - reference) * self.limits_deg
                    - (np.abs(baseline - reference) * self.limits_deg
                       + self.encoder_fidelity_slack_deg)))
            else:
                fidelity_excess = float('nan')
            fidelity_excesses.append(fidelity_excess)
            nonlinear_ok = bool(np.isfinite(actual_err) and
                                actual_err <= base_tip + self.max_ref_regret_m + 1e-12)
            rate_box_ok = bool(np.all(proposal >= lo - 1e-8) and
                               np.all(proposal <= hi + 1e-8) and
                               self._feasible(proposal))
            if not spectral_ok:
                reject_spectral_count += 1
            if not trusted or not rate_box_ok:
                reject_fidelity_count += 1
            if not nonlinear_ok:
                reject_nonlinear_count += 1

            # Lowest-dimensional candidate passing spectral, encoder-fidelity,
            # nonlinear-FK and command-box constraints wins. Otherwise use the
            # feasible full-reference command; never mislabel a partial success.
            if spectral_ok and trusted and nonlinear_ok and rate_box_ok:
                accepted_k = k
                best = proposal.copy()
                best_err = float(actual_err)
                best_k = k
                break

        if self.force_feasible_baseline or (self.risk_policy == 'enforce' and
                self._conditional_subset_unavailable):
            best, best_err, best_k = baseline.copy(), base_tip, 0
        fallback = best_k == 0
        if self.force_feasible_baseline:
            fallback_reason = 'forced_feasible_baseline_ablation'
        elif self.risk_policy == 'enforce' and self._conditional_subset_unavailable:
            fallback_reason = 'risk_subset_unavailable'
        elif fallback:
            reasons = []
            if reject_spectral_count:
                reasons.append('spectral_budget')
            if reject_fidelity_count:
                reasons.append('encoder_fidelity_or_rate_box')
            if reject_nonlinear_count:
                reasons.append('nonlinear_fk_regret')
            fallback_reason = 'no_candidate_passed:' + ('+'.join(reasons) if reasons else 'unknown')
        else:
            fallback_reason = ''
        candidate_min_fk_error = min(candidate_errors) if candidate_errors else float('nan')
        candidate_min_regret = min(candidate_regrets) if candidate_regrets else float('nan')
        candidate_max_fidelity_excess = max(fidelity_excesses) if fidelity_excesses else float('nan')
        if fallback:
            selection_status = 'fallback_reference'
        elif accepted_k == selected:
            selection_status = 'accepted_requested_k'
        else:
            selection_status = 'accepted_escalated_k'
        self.active_anchor = anchor.copy()
        self.active_basis = D[:, :best_k].copy() if best_k else np.zeros((6, 0))
        self.active_residual_start = best_k
        self.active_cap = 0.
        self._conditional_bound = float(omissions[max(selected-1, 0)] + chosen_risk)
        self._confidence_feasible = bool(self._conditional_bound <= self.risk_budget_m)
        self._baseline_tip_error = base_tip
        self._candidate_tip_error = best_err
        n = 6 if fallback else best_k
        diag = TaskEffectDiagnostics(
            dimension=selected, projection_error_m=float(omissions[selected-1]),
            residual_norm=0., residual_cap=0., feasible=self._feasible(best),
            # Default algorithm uses analytical spectral coefficients and a
            # feasible line step: zero nonlinear-program decision variables.
            optimized_variables=(n if self.refine_mode_qp and _minimize is not None else 0),
            active_residuals=max(0, best_k-selected),
            stage=(3 if fallback else 1 if best_k == selected else 2),
            task_linear_error_m=float(np.linalg.norm(A @ (best-anchor)-attainable_rhs)),
            task_full_approx_error_m=float(np.linalg.norm(desired-attainable_rhs)),
            nonlinear_objective=float(objective(self.C @ best)),
            fallback_full_reference=fallback, command_subspace_rank=n,
            conditional_risk_bound_m=self._conditional_bound,
            risk_budget_m=self.risk_budget_m, risk_feasible=self._confidence_feasible,
            contact_phase=self.contact_phase,
            sensor_uncertainty_trace=float(np.trace(self.sigma_u)),
            nonlinear_tip_error_m=best_err, nonlinear_reference_error_m=base_tip,
            nonlinear_guard_used=fallback, uncertainty_hold_used=False,
            risk_policy_enforced=(self.risk_policy=='enforce'),
            selected_mode_indices=tuple(int(i) for i in original_indices[:best_k]),
            spectral_omission_m=float(omissions[selected-1]),
            spectral_ellipsoid_risk_m=float(chosen_risk),
            selected_energy_fraction=(float(sum(gains[:best_k])) / max(float(attainable_rhs@attainable_rhs),1e-12)),
            nonlinear_feasible_reference_m=base_tip, nonlinear_candidate_m=best_err,
            baseline_regret_m=best_err-base_tip,
            requested_mode_count=int(selected),
            accepted_mode_count=int(best_k),
            output_control_rank=int(n),
            fallback_reason=fallback_reason,
            final_guard_enabled=self.final_nonlinear_guard_enabled,
            candidate_attempts=int(candidate_attempts),
            reject_spectral_count=int(reject_spectral_count),
            reject_fidelity_count=int(reject_fidelity_count),
            reject_nonlinear_count=int(reject_nonlinear_count),
            candidate_min_fk_error_m=float(candidate_min_fk_error),
            candidate_min_regret_m=float(candidate_min_regret),
            candidate_max_fidelity_excess_deg=float(candidate_max_fidelity_excess),
            selection_status=selection_status,
        )
        self.last_diag = diag
        self._diagnostics = diag
        return self.C @ best, None, diag

    def project_and_rate_limit(self, target, previous, max_speed_rad_s, dt_s):
        """Final-output task shield AFTER external servo, LPF and rate limiting.

        Certified quantity is the model FK task error *relative to a feasible
        baseline under exactly the same per-frame speed/box constraints*.
        This does not certify hardware tracking, physical contacts or collision.
        """
        target = np.asarray(target, float)
        if previous is None:
            # No last command => no rate bound at first frame.
            prevu = self._anchor_cmd
        else:
            prevu = self.encode_joints(previous)
            if not np.allclose(self.C@prevu, previous, atol=1e-6):
                raise ValueError('previous command must be on coupled manifold')
        if prevu is None:
            lo, hi = self.lower, self.upper
            anchor = self.encode_joints(target)
        else:
            self.set_command_anchor(self.C@prevu, max_speed_rad_s, dt_s)
            lo, hi = self._speed_bounds(prevu)
            anchor = prevu
        candidate, _ = ray_to_control_box(anchor, self.encode_joints(target)-anchor, lo, hi)
        baseline = np.clip(self._reference_u, lo, hi)
        candidate_err = self._nonlinear_tip_error(
            self.C@candidate, self._target_tips, self._intent)
        baseline_err = self._nonlinear_tip_error(
            self.C@baseline, self._target_tips, self._intent)
        guard_regret = float(candidate_err - baseline_err)
        guarded = (self.final_nonlinear_guard_enabled and
                   (candidate_err > baseline_err + self.max_ref_regret_m or
                    not self._fidelity_ok(candidate, baseline, self._reference_u)))
        if guarded:
            candidate, candidate_err = baseline, baseline_err
        legal = (self._feasible(candidate) and
                 np.all(candidate >= lo-1e-8) and np.all(candidate <= hi+1e-8))
        if not legal:
            raise RuntimeError('final output violated coupled/rate bounds')
        if self.last_diag is not None:
            self.last_diag = replace(self.last_diag,
                final_guard_used=bool(guarded), final_tip_error_m=candidate_err,
                final_reference_error_m=baseline_err, rate_box_valid=bool(legal),
                final_guard_reason=int(bool(guarded)),
                final_object_task_error_m=float('nan'),
                final_object_baseline_m=float('nan'),
                final_guard_enabled=self.final_nonlinear_guard_enabled,
                final_guard_regret_m=guard_regret)
        return self.C @ candidate
