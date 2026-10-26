"""TQHG-v2: condition-aware robust 4D expert, task/NULL gating.

Research prototype. Sensor faults must be measured or explicitly indicated;
stationary fingers alone are NOT treated as stuck sensors. Do not mistake a
calibrated proxy teacher or contact-hold success for hardware grasp stability.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.retargeting.tqhg_online import TQHGOnlineMapper
from src.retargeting.task_weighted_hybrid import task_projector


def masked_features(sensor_normalized, valid_mask, medians):
    x = np.asarray(sensor_normalized, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    center = np.asarray(medians, dtype=float)
    if x.shape[-1] != 15 or mask.shape != x.shape or center.shape != (15,):
        raise ValueError('expected matched 15-channel observations and validity masks')
    return np.concatenate((np.where(mask, np.clip(x, 0., 1.), center),
                           mask.astype(float)), axis=-1)


def solve_ridge(features, targets, regularizer=1.0):
    X, Y = np.asarray(features, float), np.asarray(targets, float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError('invalid ridge training matrices')
    xm, ym = X.mean(0), Y.mean(0)
    xc = X-xm
    W = np.linalg.solve(xc.T@xc + float(regularizer)*np.eye(X.shape[1]),
                        xc.T@(Y-ym))
    return W, ym-xm@W


def empirical_weight(error_a, error_b, projector, *, regularizer=0.03,
                     extra_quadratic=0., extra_linear=0.):
    """Convex w for expert B minimizing ||P(e_a+w(e_b-e_a))||^2+extra.

    Extra quadratic/linear terms can represent local null-posture and command
    continuity loss. We intentionally allow w=0; nonzero is not a goal.
    """
    a, b, H = np.asarray(error_a, float), np.asarray(error_b, float), np.asarray(projector, float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 6 or H.shape != (6, 6):
        raise ValueError('bad calibration residuals/projector')
    d = b-a
    curvature = float(np.mean(np.einsum('ni,ij,nj->n', d, H, d)))
    linear = float(np.mean(np.einsum('ni,ij,nj->n', a, H, d)))
    lam = max(1.e-9, float(regularizer)*max(curvature, 1.e-6))
    denominator = curvature + lam + float(extra_quadratic)
    numerator = -linear + 0.5*lam - float(extra_linear)
    return float(np.clip(numerator/max(denominator, 1.e-12), 0., 1.))


def conditioned_weight(clean_a, clean_b, fault_a, fault_b, P, severity, **kwargs):
    """Interpolate held-out second moments, not fitted weights."""
    ca, cb, fa, fb = [np.asarray(x, float) for x in (clean_a,clean_b,fault_a,fault_b)]
    if ca.shape != cb.shape or fa.shape != fb.shape:
        raise ValueError('mismatched residuals')
    d0, d1 = cb-ca, fb-fa
    def moments(a,d):
        return (float(np.mean(np.einsum('ni,ij,nj->n', d,P,d))),
                float(np.mean(np.einsum('ni,ij,nj->n', a,P,d))))
    h0,g0 = moments(ca,d0)
    h1,g1 = moments(fa,d1)
    r = float(np.clip(severity,0.,1.))
    h = (1-r)*h0+r*h1
    g = (1-r)*g0+r*g1
    reg = max(1.e-9, float(kwargs.get('regularizer',.03))*max(h,1.e-6))
    eq = float(kwargs.get('extra_quadratic',0.))
    el = float(kwargs.get('extra_linear',0.))
    return float(np.clip((-g+0.5*reg-el)/(h+reg+eq), 0.,1.))


class SensorQuality:
    """Conservative quality detector; no unsupported stuck-sensor diagnosis."""
    def __init__(self, low, span, medians):
        self.low = np.asarray(low, float)
        self.span = np.asarray(span, float)
        self.medians = np.asarray(medians, float)
        self.prev = None

    def reset(self):
        self.prev = None

    def observe(self, sensors_deg, explicit_valid=None):
        s = np.asarray(sensors_deg, float)
        if s.shape != (15,):
            raise ValueError('expected 15 glove sensor channels')
        finite = np.isfinite(s)
        mask = finite.copy()
        if explicit_valid is not None:
            v = np.asarray(explicit_valid, dtype=bool)
            if v.shape != (15,):
                raise ValueError('sensor validity mask must have 15 entries')
            mask &= v
        sn = np.divide(np.where(finite,s,self.low)-self.low, self.span)
        # Substantial excursions beyond TRAINING calibration range are not
        # silently interpreted as normal measurements.
        extreme = (sn < -.20) | (sn > 1.20)
        mask &= ~extreme
        sn = np.clip(sn,0.,1.)
        # An implausible one-frame jump decreases confidence, but does not
        # automatically erase a valid channel (legitimate gestures can be fast).
        jump_penalty = 0.
        if self.prev is not None:
            jump_penalty = float(np.mean(np.clip((np.abs(sn-self.prev)-.35)/.65,0.,1.) * mask))
        self.prev = sn.copy()
        reliability = float(np.clip(np.mean(mask)-.25*jump_penalty,0.,1.))
        return sn, mask, reliability, int(np.count_nonzero(~mask))


class TQHGv2OnlineMapper(TQHGOnlineMapper):
    METHODS = TQHGOnlineMapper.METHODS + ('tqhg-v2',)

    def __init__(self, robot, calibration_file):
        path = Path(calibration_file)
        cfg = json.loads(path.read_text(encoding='utf-8'))
        if cfg.get('schema') != 'tqhg_online_calibration_v2':
            raise ValueError('v2 model required: run scripts/train_tqhg_v2.py')
        super().__init__(robot, cfg['base_v1'])
        self.calibration_file = path
        self.v2_cfg = cfg
        robust = cfg['robust_expert']
        self.robust_W = np.asarray(robust['feature_to_latent_weights'],float)
        self.robust_b = np.asarray(robust['feature_to_latent_bias'],float)
        self.medians = np.asarray(robust['sensor_medians_normalized'],float)
        self.robust_basis = np.asarray(self.pca_basis,float)
        if self.robust_W.shape != (30,4) or self.robust_b.shape != (4,) or self.medians.shape != (15,):
            raise ValueError('robust expert must be mask-conditioned 30->4 latent')
        self.quality = SensorQuality(self.sensor_min,self.sensor_span,self.medians)
        self.history_u = None
        self.previous_w = None
        self._last_task = None
        self._diagnostic_ready = True
        self.reset()

    def reset(self):
        self.history_u = None
        self.previous_w = None
        self._last_task = None
        self.quality.reset()

    def predict(self, sensors_deg, *, method='tqhg-v2', simulator=None,
                task_mode='pinch', sensor_valid_mask=None):
        if method != 'tqhg-v2':
            return super().predict(sensors_deg, method=method, simulator=simulator, task_mode=task_mode)
        # A new task should not inherit old gate smoothness or a previous
        # task's null-space posture memory.
        if task_mode != self._last_task:
            self.reset()
            self._last_task = task_mode
        sn, mask, reliability, n_bad = self.quality.observe(sensors_deg, sensor_valid_mask)
        x = masked_features(sn,mask,self.medians)
        ridge = np.clip(x[:15] @ self.W_ridge + self.b_ridge,0.,1.)
        latent = x @ self.robust_W + self.robust_b
        robust = np.clip(self.mu + latent @ self.robust_basis.T,0.,1.)
        P = task_projector(self._jacobian(simulator,task_mode))
        Q = np.eye(6)-P
        cal = self.v2_cfg['gate_calibration']
        A0,B0,A1,B1 = [np.asarray(cal[k],float) for k in
                        ('normal_errors_ridge','normal_errors_robust',
                         'fault_errors_ridge','fault_errors_robust')]
        # Missing explicit sensor channels indicate measured degradation.
        severity = float(np.clip((1.-reliability)*2.0,0.,1.))
        cfg = self.v2_cfg['hyperparameters']
        reg=float(cfg['gate_regularizer'])
        wt = (conditioned_weight(A0,B0,A1,B1,P,severity,regularizer=reg)
              if np.trace(P) >= 0.5 else 0.0)
        d_null=Q@(robust-ridge)
        # Auxiliary objective: move null posture toward the projection of the
        # Ridge command onto the learned 4D synergy manifold, and limit changes
        # vs the PREVIOUS *mapper* output. This is not a measured grip quality.
        pca_ref = self.mu + self.robust_basis @ (self.robust_basis.T @ (ridge-self.mu))
        posture_lam=float(cfg['null_posture_weight'])
        smooth_lam=float(cfg['null_smooth_weight']) if self.history_u is not None else 0.
        extra_q=(posture_lam+smooth_lam)*float(d_null@d_null)
        extra_l=posture_lam*float((Q@(ridge-pca_ref))@d_null)
        if self.history_u is not None:
            extra_l += smooth_lam*float((Q@(ridge-self.history_u))@d_null)
        wn=conditioned_weight(A0,B0,A1,B1,Q,severity,regularizer=reg,
                              extra_quadratic=extra_q,extra_linear=extra_l)
        smooth=float(cfg['gate_smoothing'])
        if self.previous_w is not None:
            wt = (1.-smooth)*self.previous_w[0]+smooth*wt
            wn = (1.-smooth)*self.previous_w[1]+smooth*wn
        self.previous_w=(wt,wn)
        raw=ridge+wt*(P@(robust-ridge))+wn*d_null
        bounded=np.clip(raw,self.lower,self.upper)
        prior_safe = None if self.history_u is None else self.history_u.copy()
        d=robust-ridge
        self.last_gate={
            'w_task':float(wt),'w_null':float(wn),
            'visible_rank':int(round(np.trace(P))),
            'box_clipped':bool(np.max(np.abs(raw-bounded))>1e-10),
            'sensor_reliability':reliability,'invalid_channels':n_bad,
            'gate_severity':severity,
            'expert_delta_norm':float(np.linalg.norm(d)),
            'task_delta_norm':float(np.linalg.norm(wt*(P@d))),
            'null_delta_norm':float(np.linalg.norm(wn*d_null)),
            'post_clip_task_delta_norm':float(np.linalg.norm(P@(bounded-raw))),
            'gate_fallback':int(n_bad >= 10),
        }
        # Fail safe: when 10+ channels are explicitly invalid, don't trust
        # either learned expert; issue the previous safe mapper command or
        # a conservative neutral control. Final motor guard remains active.
        if n_bad >= 10:
            # Hold previous SAFE mapped command when possible; when beginning
            # with 10+ absent sensors use a bounded calibration mean. Never
            # synthesize a new grasp motion from nearly all missing sensors.
            bounded = (prior_safe if prior_safe is not None else
                       np.clip(self.mu,self.lower,self.upper))
        self.history_u=bounded.copy()
        return bounded
