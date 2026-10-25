"""Research prototype: task-observable / complementary-space confidence gating.

Blends two *different* 15-sensor -> 6-control experts in a task-conditioned
orthogonal decomposition. The gate is fitted on a SEPARATE calibration set.
It is not a new dimension-reduction theorem and does not guarantee hardware
safety; the final 6D command must be checked against the robot's bounds.
"""
from __future__ import annotations
import numpy as np


def task_projector(J: np.ndarray, relative_tol: float = 0.04) -> np.ndarray:
    """Euclidean projector onto task-visible control directions (row space J)."""
    J = np.asarray(J, float)
    if J.ndim != 2 or J.shape[1] != 6 or not np.isfinite(J).all():
        raise ValueError('J must be finite [task_dim,6]')
    _, s, vt = np.linalg.svd(J, full_matrices=False)
    if s.size == 0 or s[0] < 1e-12:
        return np.zeros((6,6))
    r = int(np.count_nonzero(s >= relative_tol * s[0]))
    B = vt[:r].T
    return B @ B.T


def optimal_weight(errors_a: np.ndarray, errors_b: np.ndarray,
                   metric: np.ndarray, regularizer: float = 0.05) -> float:
    """Calibration-optimal scalar mixing weight (for expert B).

    Uses bias-aware empirical second moments, including cross correlation.
    The ridge regularizer stabilizes nearly identical experts, pulling weight
    towards 0.5. This is an empirical optimum on CALIBRATION, not a test bound.
    """
    a, b, H = map(lambda x: np.asarray(x, float), (errors_a,errors_b,metric))
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 6 or H.shape != (6,6):
        raise ValueError('invalid residual/metric shapes')
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(H).all()):
        raise ValueError('nonfinite input')
    H = 0.5*(H+H.T)
    if np.linalg.eigvalsh(H).min() < -1e-9:
        raise ValueError('metric must be positive semidefinite')
    delta = b-a
    denom = float(np.mean(np.einsum('ni,ij,nj->n', delta,H,delta)))
    numer = -float(np.mean(np.einsum('ni,ij,nj->n', a,H,delta)))
    # Prior towards a neutral convex mixture stabilizes a small calibration set.
    reg = float(regularizer) * max(denom, 1e-6)
    if reg < 0: raise ValueError('regularizer must be nonnegative')
    return float(np.clip((numer + 0.5*reg)/(denom+reg), 0.,1.))


def gate_prediction(expert_a: np.ndarray, expert_b: np.ndarray,
                    calibration_errors_a: np.ndarray,
                    calibration_errors_b: np.ndarray,
                    J: np.ndarray, *, regularizer: float = 0.05,
                    split: bool = True) -> tuple[np.ndarray,dict]:
    """Returns safe *normalized control box* candidate [0,1]^6.

    When split=True use independent task-visible and task-null weights.
    Note that orthogonal projectors can make the unbounded output leave the box,
    so final clipping is mandatory and invalidates exact subspace membership.
    A downstream 6->11 coupled rate/limit guard is still necessary.
    """
    a,b = np.asarray(expert_a,float),np.asarray(expert_b,float)
    if a.shape != (6,) or b.shape != (6,): raise ValueError('expected six controls')
    P=task_projector(J)
    Q=np.eye(6)-P
    ea=np.asarray(calibration_errors_a,float)
    eb=np.asarray(calibration_errors_b,float)
    wt=optimal_weight(ea,eb,P,regularizer)
    wn=optimal_weight(ea,eb,Q,regularizer) if split else wt
    raw=a+(wt*P+wn*Q)@(b-a)
    out=np.clip(raw,0.,1.)
    return out, {'w_task':wt, 'w_null':wn,
                 'visible_rank':int(round(np.trace(P))),
                 'box_clipped':bool(np.max(np.abs(raw-out))>1e-10)}
