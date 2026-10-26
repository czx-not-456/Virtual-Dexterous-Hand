"""Scheme C algorithmic tests. NOT experimental evidence of better grasps."""
import numpy as np
import pytest

from src.common import load_yaml
from src.features.intent_recognition import Intent
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper
from src.retargeting.reachability_synergy import ReachabilityGatedSynergyMapper


def mapper(**config):
    cfg = load_yaml('configs/reachability_synergy.yaml')
    cfg['fallback_full_reference'] = False  # isolate the two-stage solver
    cfg.update(config)
    return ReachabilityGatedSynergyMapper(
        LatentSynergyMapper(RobotHandModel(), 'configs/latent_chm6_adaptive.json'), cfg)


def test_true_variable_count_lower_than_six_without_rescue():
    m = mapper(k_min=2, k_max=2, max_residual_axes=0)
    up = m.mu.copy()
    qprev = m.decode_controls(up)
    target = m._check_control(up + 0.03 * m.B[:, 0])
    q, result, diag = m.optimize(lambda q: np.mean((q - m.decode_controls(target))**2),
                                 qprev, Intent.NEUTRAL, target)
    assert diag.dimension == 2
    assert diag.optimized_variables == 2  # not a nominal k on a six-variable optimizer
    assert diag.active_residuals == 0
    assert diag.stage == 1 and diag.feasible
    assert np.allclose(q, m.C @ m.encode_joints(q), atol=1e-8)


def test_event_triggered_residual_can_actually_open():
    # Choose an unreachable-by-primary command in the local complement.
    m = mapper(k_min=2, k_max=2, max_residual_axes=2,
               rescue_threshold_m=0., projection_abs_m=0., projection_relative=0.,
               lambda_reference=20.0, lambda_residual=0., lambda_sparse=0.,
               rescue_min_improvement_m=0., residual_caps={'NEUTRAL': 0.30})
    up = m.mu.copy()
    qprev = m.decode_controls(up)
    J = m._jacobian(up, m.FINGERS[Intent.NEUTRAL])
    V = m._basis(J, np.zeros(6))
    displacement = 0.18 * V[:, -1]
    target = m._check_control(up + displacement)
    # Even with a clipped command, the full target is outside the 2D primary span.
    q, result, diag = m.optimize(lambda q: np.mean((q - m.C @ target)**2),
                                 qprev, Intent.NEUTRAL, target)
    fixed = mapper(k_min=2, k_max=2, max_residual_axes=0,
                   rescue_threshold_m=0., projection_abs_m=0.,
                   projection_relative=0., lambda_reference=20.0,
                   lambda_residual=0., lambda_sparse=0.)
    q_fixed, _, diag_fixed = fixed.optimize(
        lambda q: np.mean((q - fixed.C @ target)**2),
        qprev, Intent.NEUTRAL, target)
    assert diag.optimized_variables > diag.dimension
    assert diag.optimized_variables <= 4
    assert diag.stage == 2 and diag.feasible
    assert m._feasible(m.encode_joints(q))
    assert np.linalg.norm(m.encode_joints(q) - target) < np.linalg.norm(
        fixed.encode_joints(q_fixed) - target)


def test_task_specific_basis_responds_to_intent():
    m = mapper()
    up = m.mu
    J_pinch = m._jacobian(up, m.FINGERS[Intent.PINCH])
    J_wrap = m._jacobian(up, m.FINGERS[Intent.WRAP])
    assert J_pinch.shape == (6, 6)
    assert J_wrap.shape == (15, 6)
    Bp = m._basis(J_pinch, np.zeros(6))
    Bw = m._basis(J_wrap, np.zeros(6))
    assert np.allclose(Bp.T @ Bp, np.eye(6), atol=1e-9)
    assert np.allclose(Bw.T @ Bw, np.eye(6), atol=1e-9)
    assert not np.allclose(abs(Bp), abs(Bw), atol=1e-5)


def test_postprocessing_preserves_control_limits_coupling_and_rate():
    m = mapper(k_min=2, k_max=2, max_residual_axes=1)
    prev = m.decode_controls(m.mu)
    target = m._check_control(m.mu + 0.2 * m.B[:, 2])
    q, _, diag = m.optimize(lambda q: np.mean((q - m.C @ target)**2),
                            prev, Intent.NEUTRAL, target)
    output = m.project_and_rate_limit(q + 0.03, prev, np.full(11, 0.2), 1/60)
    assert m._feasible(m.encode_joints(output))
    assert np.allclose(m.C @ m.encode_joints(output), output, atol=1e-8)
    assert np.max(abs(output - prev)) <= 0.2 / 60 + 1e-9


def test_full_retarget_task_targets_are_accepted():
    m = mapper()
    prev = m.decode_controls(m.mu)
    fingers = m.FINGERS[Intent.PINCH]
    tips = m.robot.fingertips(prev)
    wanted = {f: tips[f].copy() + np.array([0.001, 0.0, 0.0]) for f in fingers}
    q, _, diag = m.optimize(lambda q: 0., prev, Intent.PINCH,
                            m.mu, task_targets_m=wanted)
    assert diag.feasible
    assert np.isfinite(diag.task_full_approx_error_m)
    assert 2 <= diag.optimized_variables <= 6


def test_bad_numeric_inputs_are_rejected():
    with pytest.raises(ValueError):
        mapper(max_residual_axes=-1)
    m = mapper()
    q = m.decode_controls(m.mu)
    with pytest.raises(ValueError):
        m.optimize(lambda q: 0., q, Intent.PINCH, m.mu,
                   task_targets_m={'thumb': [float('nan'), 0, 0], 'index': [0, 0, 0]})


def test_optional_full_reference_shield_is_labeled_not_concealed():
    m = mapper(k_min=2, k_max=2, max_residual_axes=0,
               fallback_full_reference=True, fallback_min_gain=0.)
    qprev = m.decode_controls(m.mu)
    goal = m._check_control(m.mu + 0.2 * m.B[:, 4])
    qref = m.decode_controls(goal)
    q, result, diag = m.optimize(
        lambda q: float(np.mean((q - qref)**2)), qprev, Intent.NEUTRAL, goal)
    assert diag.fallback_full_reference
    assert diag.stage == 3
    assert diag.optimized_variables == 2  # true number sent to optimizer
    assert diag.command_subspace_rank == 6  # output is NOT claimed to be 2D
    assert np.allclose(q, qref)
