"""Scheme-B unit / invariance tests. No contact or hardware validation implied."""
from pathlib import Path

import numpy as np
import pytest

from src.common import load_yaml
from src.features.intent_recognition import Intent
from src.glove.captured_samples import load_captured_samples
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.adaptive_synergy import TaskAdaptiveSynergyMapper
from src.retargeting.latent_synergy import LatentSynergyMapper, fit_latent_model
from src.retargeting.optimizer import IntentDrivenRetargeter

MODEL = 'configs/latent_chm6_adaptive.json'


def make_mapper(**overrides):
    cfg = load_yaml('configs/adaptive_synergy.yaml')
    cfg.update(overrides)
    return TaskAdaptiveSynergyMapper(LatentSynergyMapper(RobotHandModel(), MODEL), cfg)


def assert_coupled(m, q):
    assert q.shape == (11,)
    assert np.all(q >= m.robot.ranges[:, 0] - 1e-9)
    assert np.all(q <= m.robot.ranges[:, 1] + 1e-9)
    for driver, child, gain in m.base.teacher.couplings:
        idx = m.base.teacher.index
        assert q[idx[child]] == pytest.approx(gain * q[idx[driver]], abs=1e-8)


def test_full_basis_and_supervised_reference_are_separate_from_k_reconstruction():
    m = make_mapper()
    samples = load_captured_samples(Path('数据手套数据/records/chm6_calibrations/20260917_102742_G12LC040704_left/captured_samples.csv'))
    s, controls = samples[0].sensors_deg, samples[0].targets_deg
    for mode in ('sensor', 'control', 'blend'):
        u = m.controls_from_observation(s, controls, mode)
        assert u.shape == (6,)
        assert_coupled(m, m.decode_controls(u))
    assert np.allclose(m.B.T @ m.B, np.eye(6), atol=1e-8)


def test_dimensional_gate_responds_to_different_command_directions():
    m = make_mapper(k_min=2, k_max=5, projection_abs_m=1e-6, projection_relative=0.02)
    q = m.decode_controls(m.mu)
    p0 = m._check_control(m.mu + 0.12 * m.B[:, 0])
    p4 = m._check_control(m.mu + 0.12 * m.B[:, 4])
    low, _ = m.select_dimension(q, p0, Intent.NEUTRAL)
    high, _ = m.select_dimension(q, p4, Intent.NEUTRAL)
    assert low == 2
    assert high >= 4


def test_bounded_complement_restores_lost_independent_direction():
    # An input movement in the fifth/sixth PCA direction is removed by k=2.
    # Allowing a bounded complement should reduce reference error.
    cfg = dict(k_min=2, k_max=2, lambda_reference=10., lambda_residual=1e-4,
               residual_caps={'NEUTRAL': .30}, maxiter=120)
    relaxed = make_mapper(**cfg)
    fixed = make_mapper(**{**cfg, 'residual_caps': {'NEUTRAL': 0.}})
    target = relaxed._check_control(relaxed.mu + 0.14 * relaxed.B[:, 3])
    qprev = relaxed.decode_controls(relaxed.mu)
    cost = lambda q: 0.
    q1, _, diag1 = relaxed.optimize(cost, qprev, Intent.NEUTRAL, target)
    q0, _, diag0 = fixed.optimize(cost, qprev, Intent.NEUTRAL, target)
    err1 = np.linalg.norm(relaxed.encode_joints(q1) - target)
    err0 = np.linalg.norm(fixed.encode_joints(q0) - target)
    assert diag1.residual_norm > 0.02
    assert err1 < err0 * 0.5
    assert diag1.feasible and diag0.feasible
    assert_coupled(relaxed, q1)
    assert_coupled(fixed, q0)


def test_slsqp_optimized_and_output_limited_remain_feasible():
    m = make_mapper()
    h = HumanHandModel()
    rh = m.robot
    optimizer = IntentDrivenRetargeter(h, rh, load_yaml('configs/algorithm.yaml')['optimizer'],
                                        latent_mapper=m.base, adaptive_mapper=m,
                                        adaptive_source='control')
    sensors = np.linspace(5, 60, 15)
    qprev = m.decode_controls(m.mu)
    qh = h.angles_from_normalized({name: .55 for name in h.joint_order})
    controls = np.array([40., 18., 35., 80., 12., 70.])
    result = optimizer.solve(qh, h.fingertips(qh), Intent.PINCH,
                             q_prev=qprev, q_base=qprev, sensors_deg=sensors,
                             controls_deg=controls)
    assert np.isfinite(result.objective)
    assert 2 <= result.components['adaptive_k'] <= 5
    assert result.components['adaptive_feasible'] == 1.
    assert_coupled(m, result.q_target)
    # A deliberately decoupled target may be projected back to 6D, then
    # rate-limited. Output target is coupled and within every joint speed cap.
    target = result.q_target + np.linspace(-.2, .1, 11)
    vmax = np.full(11, .15)
    out = m.project_and_rate_limit(target, qprev, vmax, dt_s=.02)
    assert_coupled(m, out)
    assert np.max(np.abs(out - qprev)) <= .003 + 1e-9


def test_no_mixed_devices_and_invalid_input_rejected():
    m = make_mapper()
    with pytest.raises(ValueError):
        m.controls_from_observation(np.zeros(14), None, 'sensor')
    with pytest.raises(ValueError):
        m.controls_from_observation(None, None, 'control')
    with pytest.raises(ValueError):
        m.project_and_rate_limit(np.zeros(11), np.zeros(11), np.zeros(11), 1/60)


def test_new_fit_exports_full_basis_without_changing_legacy_k_fields():
    rng = np.random.default_rng(10)
    p = fit_latent_model(rng.uniform(0, 90, (64, 15)),
                         rng.uniform(0, 30, (64, 6)), k=4)
    assert p['version'] == 2
    assert np.asarray(p['control_pca_full_basis']).shape == (6, 6)
    assert np.asarray(p['sensor_to_controls_weights']).shape == (15, 6)
    assert np.asarray(p['control_pca_basis']).shape == (6, 4)


def test_task_intent_changes_selected_subspace_for_non_target_finger():
    m = make_mapper()
    prev = m.decode_controls(m.mu)
    target = m.mu.copy()
    target[3] = min(m.upper[3], target[3] + .25)  # middle finger, not pinch target
    pinch_k, _ = m.select_dimension(prev, target, Intent.PINCH)
    wrap_k, _ = m.select_dimension(prev, target, Intent.WRAP)
    assert pinch_k < wrap_k


def test_downstream_projection_binds_residual_after_servo():
    m = make_mapper(k_min=2, k_max=2, residual_caps={'NEUTRAL': 0.03})
    q = m.decode_controls(m.mu)
    goal = m.mu.copy()
    goal[4] += .3
    _, _, diag = m.optimize(lambda q: 0.0, q, Intent.NEUTRAL, goal)
    assert diag.dimension == 2
    # Simulate a downstream controller requesting more independent motion than
    # the active residual cap. The projector must remain in the cap set.
    desired = m.decode_controls(m._check_control(goal))
    projected = m.project_and_rate_limit(desired, None, np.full(11, 100.0), .01)
    coeff = m.B.T @ (m.encode_joints(projected) - m.mu)
    assert np.max(np.abs(coeff[2:])) <= .03 + 1e-8
    assert_coupled(m, projected)
