"""Stage C: fair robust6/robust4 task-subspace ablation for CH-M6.

Controls are normalized 6D. Actual task Jacobian is supplied by MuJoCo.
With no simulator, P=0 and alpha cannot be interpreted as task control.
All expert/risk parameters are fitted on historical TRAIN/CAL only.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from src.retargeting.tqhg_v2 import TQHGv2OnlineMapper, masked_features
from src.retargeting.task_weighted_hybrid import task_projector

STAGEC_METHODS = ('robust6', 'robust4', 'stagec-alpha', 'stagec-beta', 'stagec-dual')


def _conditioned_projector_weight(errors, projector, missing_count, regularizer=.03):
    """Interpolate error second moments between CAL-only fault strata.

    Every fault row is paired (same mask and teacher target for both experts).
    No held-out 24-frame labels enter this calculation. A clipped zero weight
    is a valid and informative optimum.
    """
    H = np.asarray(projector, float)
    if H.shape != (6, 6) or not np.isfinite(H).all():
        raise ValueError('projector must be finite 6x6')
    counts = (0, 1, 3, 5)
    severity = float(np.clip(missing_count, 0, 5))
    if severity <= 1:
        left, right = 0, 1
    elif severity <= 3:
        left, right = 1, 3
    else:
        left, right = 3, 5
    mix = (severity-left)/(right-left)
    def second_moments(key):
        group = errors[str(key)]
        a, b = np.asarray(group['a'],float),np.asarray(group['b'],float)
        if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 6 or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('invalid Stage C calibration residuals')
        d = b-a
        h = np.mean(np.einsum('ni,ij,nj->n',d,H,d))
        g = np.mean(np.einsum('ni,ij,nj->n',a,H,d))
        return float(h),float(g)
    h0,g0 = second_moments(left)
    h1,g1 = second_moments(right)
    h,g = (1-mix)*h0+mix*h1, (1-mix)*g0+mix*g1
    reg = max(1e-9, float(regularizer)*max(h,1e-6))
    return float(np.clip((-g+0.5*reg)/(h+reg),0.,1.))


class StageCOnlineMapper(TQHGv2OnlineMapper):
    """Robust expert pair + alpha-only / beta-only / dual ablation.

    Explicitly missing sensors are *masked* for BOTH experts. A stationary
    measurement is not inferred to be missing. Fault injection is orchestrated
    upstream, while intent recognition still sees the clean replay stream.
    """
    def __init__(self, robot, v2_file, robust6_file, gate_file):
        super().__init__(robot, v2_file)
        r = json.loads(Path(robust6_file).read_text(encoding='utf-8'))
        g = json.loads(Path(gate_file).read_text(encoding='utf-8'))
        if r.get('schema') != 'stageB_robust_ridge6_v1' or g.get('schema') != 'stageC_task_gate_v1':
            raise ValueError('Stage C requires Stage B robust6 and CAL-only Stage C gate models')
        self.robust6_W = np.asarray(r['feature_to_controls_weights'],float)
        self.robust6_b = np.asarray(r['feature_to_controls_bias'],float)
        self.stagec_gates = g
        if self.robust6_W.shape != (30,6) or self.robust6_b.shape != (6,) or not np.isfinite(self.robust6_W).all():
            raise ValueError('Bad 30->6 robust6 model')
        if not np.allclose(r['sensor_medians_normalized'],self.medians):
            raise ValueError('robust6 and robust4 must use matching imputation statistics')
        if (r['provenance']['train_count']!=124 or r['provenance']['calibration_count']!=66 or
            g['provenance']['calibration_count'] != 66 or
            g['provenance']['seed'] != r['provenance']['seed']):
            raise ValueError('Incompatible TRAIN/CAL provenance')
        self.methods = STAGEC_METHODS

    def predict(self, sensors_deg, *, method='stagec-dual', simulator=None,
                task_mode='pinch', sensor_valid_mask=None):
        if method not in STAGEC_METHODS:
            raise ValueError('Unsupported Stage C method '+str(method))
        if task_mode != self._last_task:
            self.reset()
            self._last_task = task_mode
        sn, valid, reliability, n_bad = self.quality.observe(sensors_deg, sensor_valid_mask)
        x = masked_features(sn,valid,self.medians)
        a = np.clip(x @ self.robust6_W + self.robust6_b, self.lower,self.upper)
        b = np.clip(self.mu + (x @ self.robust_W+self.robust_b)@self.robust_basis.T,
                    self.lower,self.upper)
        J = self._jacobian(simulator,task_mode)
        P=task_projector(J); N=np.eye(6)-P
        d=b-a
        cfg=self.stagec_gates['hyperparameters']
        if method in ('robust6','robust4'):
            wt,wn=0.,0.
        else:
            wt=(_conditioned_projector_weight(self.stagec_gates['errors'],P,n_bad,cfg['gate_regularizer'])
                if simulator is not None and np.trace(P)>.5 and method!='stagec-beta' else 0.)
            wn=(_conditioned_projector_weight(self.stagec_gates['errors'],N,n_bad,cfg['gate_regularizer'])
                if method!='stagec-alpha' else 0.)
            # Optional auxiliary null-posture goal remains a soft criterion. A
            # nonzero beta is NEVER forced without calibration evidence.
            if wn>0 and float(cfg['null_posture_weight'])>0:
                pca_ref=self.mu+self.robust_basis@(self.robust_basis.T@(a-self.mu))
                null_d=N@d
                gain=float(cfg['null_posture_weight'])
                # convex regularization toward a learned 4D reference posture
                denom=(null_d@null_d)*(1.+gain)+1e-12
                posture_w=float(np.clip(-((N@(a-pca_ref))@null_d)/denom,0.,1.))
                wn=(1-gain)*wn+gain*posture_w
            smooth=float(cfg['gate_smoothing'])
            if self.previous_w is not None:
                wt=(1-smooth)*self.previous_w[0]+smooth*wt
                wn=(1-smooth)*self.previous_w[1]+smooth*wn
        self.previous_w=(wt,wn)
        raw=(b if method=='robust4' else a if method=='robust6' else
             a+wt*(P@d)+wn*(N@d))
        bounded=np.clip(raw,self.lower,self.upper)
        if n_bad >= 10:
            bounded=(self.history_u.copy() if self.history_u is not None else
                     np.clip(self.mu,self.lower,self.upper))
        leakage=float(np.linalg.norm(J@N)/(max(np.linalg.norm(J),1e-12)))
        self.last_gate={
            'w_task':float(wt),'w_null':float(wn),'visible_rank':int(round(np.trace(P))),
            'box_clipped':bool(np.max(np.abs(raw-bounded))>1e-10),
            'sensor_reliability':float(reliability),'invalid_channels':int(n_bad),
            'gate_severity':float(n_bad/15.),
            'expert_delta_norm':float(np.linalg.norm(d)),
            'task_delta_norm':float(np.linalg.norm(wt*(P@d))),
            'null_delta_norm':float(np.linalg.norm(wn*(N@d))),
            'post_clip_task_delta_norm':float(np.linalg.norm(P@(bounded-raw))),
            'gate_fallback':int(n_bad>=10),
            'stagec_projection_leakage':leakage,
            'stagec_expert_a_norm':float(np.linalg.norm(a)),
            'stagec_expert_b_norm':float(np.linalg.norm(b)),
            'stagec_task_output_shift':float(np.linalg.norm(J@(bounded-a))),
            'stagec_null_output_shift':float(np.linalg.norm(N@(bounded-a))),
        }
        self.history_u=bounded.copy()
        return bounded


def fault_mask_for_frame(frame, *, channels, start, stop):
    """Explicit ONE-BASED channel IDs; end frame is exclusive."""
    mask=np.ones(15,bool)
    if start <= int(frame) < stop:
        for channel in channels:
            if int(channel)<1 or int(channel)>15:
                raise ValueError('fault channel ID must be 1..15')
            mask[int(channel)-1]=False
    return mask


def parse_fault_channels(value):
    if not value.strip():return ()
    try: chans=tuple(int(s.strip()) for s in value.split(','))
    except ValueError as exc:raise ValueError('Comma-separated channel IDs required (1..15)') from exc
    if len(set(chans))!=len(chans) or any(x<1 or x>15 for x in chans):
        raise ValueError('Fault channels must be unique integers 1..15')
    return chans
