"""Scheme D algorithm-property tests, not physical manipulation evidence."""
from __future__ import annotations

import numpy as np
import pytest
from src.common import load_yaml
from src.features.intent_recognition import Intent
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper
from src.retargeting.credible_contact_synergy import CredibleContactSynergyMapper
from src.retargeting.uncertainty_model import EncoderUncertainty, symmetric_psd


def mapper(**overrides):
    cfg = load_yaml('configs/credible_contact_synergy.yaml')
    cfg.update(overrides)
    return CredibleContactSynergyMapper(
        LatentSynergyMapper(RobotHandModel(), 'configs/latent_chm6_adaptive.json'), cfg)


def test_calibrated_proxy_is_psd_and_units_are_normalized():
    m = mapper()
    x = (m.sensor_min + 0.5*m.sensor_span)
    pred = m.controls_from_observation(x, np.zeros(6), source='sensor')
    assert pred.shape == (6,) and np.all((pred >= m.lower) & (pred <= m.upper))
    assert np.isfinite(m.sigma_u).all()
    assert np.linalg.eigvalsh(m.sigma_u).min() >= -1e-10
    assert np.trace(m.sigma_u) > 0
    with pytest.raises(ValueError):
        symmetric_psd(np.array([[1., 2.], [2., -1.]]), 2)
    with pytest.raises(ValueError):
        m.controls_from_observation(np.full(15, np.nan), np.zeros(6))


def test_conditional_error_bound_increases_with_noise():
    m = mapper()
    A = np.eye(6)
    V = np.eye(6)
    desired = np.array([0., 0., .1, 0., 0., 0.])
    small = m.conditional_bound(A,V,2,np.eye(6)*1e-8,desired,.1)
    large = m.conditional_bound(A,V,2,np.eye(6)*1e-4,desired,.1)
    assert large[0] == pytest.approx(small[0])
    assert large[1] > small[1]
    # Deterministic projector error goes to zero if the missing component is used.
    assert m.conditional_bound(A,V,3,np.zeros((6,6)),desired,.1)[0] < 1e-10


def test_rank_true_and_coupling_rate_limits_hold():
    m = mapper(k_min=2, k_max=3, max_residual_axes=0)
    sensors = m.sensor_min + m.sensor_span * 0.4
    target = m.controls_from_observation(sensors, None, source='sensor')
    prev = m.decode_controls(m.mu)
    result, _, d = m.optimize(lambda q: np.mean((q-m.C@target)**2),
                              prev, Intent.NEUTRAL, target)
    assert d.optimized_variables in (2,3)
    assert d.command_subspace_rank in (2,3,6)
    assert d.contact_phase == 'UNOBSERVED'
    out = m.project_and_rate_limit(result, prev, np.full(11,.4),1/60)
    assert np.max(np.abs(out-prev)) < .4/60 + 1e-9
    assert np.allclose(out, m.C @ m.encode_joints(out), atol=1e-9)
    assert m._feasible(m.encode_joints(out))


def test_contact_hysteresis_and_unobserved_is_not_no_contact():
    m = mapper(contact_dwell_frames=2, contact_loss_frames=2, contact_rescue_frames=3)
    stable = {'thumb_tip_contact':True,'index_tip_contact':True,
              'finger_contact':True, 'stable_contact_proxy':True}
    loss = {'thumb_tip_contact':False,'index_tip_contact':False,
            'finger_contact':False, 'stable_contact_proxy':False}
    assert m.update_contact({}, task_active=True, observed=False) == 'UNOBSERVED'
    assert m.update_contact(stable, task_active=True) == 'APPROACH'
    assert m.update_contact(stable, task_active=True) == 'SECURE'
    assert m._allowed_residual_axes(3, Intent.PINCH) == 0
    assert m.update_contact(loss, task_active=True) == 'SECURE'
    assert m.update_contact(loss, task_active=True) == 'LOSS_RISK'
    assert m._contact_rescue_needed()
    remaining = m._rescue_remaining
    m.update_contact(loss, task_active=True)
    assert m._rescue_remaining == remaining  # not continuously restarted
    assert m.update_contact(loss, task_active=False) == 'FREE'
    assert m.update_contact(None, task_active=True, observed=False) == 'UNOBSERVED'


def test_fk_guard_and_uncertainty_hold_are_diagnosed():
    m = mapper(k_min=2,k_max=2,max_residual_axes=0,
               nonlinear_guard_enabled=True,guard_regret_m=0.,
               lambda_reference=0.0,fallback_full_reference=False)
    prev = m.decode_controls(m.mu)
    target = m._check_control(m.mu + .18*m.B[:,3])
    qref = m.decode_controls(target)
    # Zero task objective means the solver has no reason to leave its feasible
    # initial projected point except reference regularization (disabled).
    q, _, d = m.optimize(lambda q: 0., prev, Intent.NEUTRAL, target)
    assert d.nonlinear_tip_error_m <= d.nonlinear_reference_error_m+1e-8
    if d.nonlinear_guard_used:
        assert np.allclose(q,qref)
        assert d.command_subspace_rank==6
    held = mapper(risk_policy="enforce",hard_hold_risk_m=1e-10,max_residual_axes=0)
    start = held.decode_controls(held.mu)
    goal = held._check_control(held.mu + .05*held.B[:,0])
    qhold, _, dh = held.optimize(lambda q: 0.,start,Intent.NEUTRAL,goal)
    assert dh.uncertainty_hold_used and dh.command_subspace_rank == 0
    out = held.project_and_rate_limit(qhold, start, np.ones(11), 1/60)
    assert np.allclose(out,start,atol=1e-8)


def test_observed_contact_changes_task_weighting_without_fake_telemetry():
    m = mapper()
    up = m.mu.copy()
    ref = m._check_control(up+.05)
    J0,A0,r0 = m._features(up,ref,Intent.PINCH,None)
    m.update_contact({'thumb_tip_contact':True,'index_tip_contact':False,
                      'finger_contact':True,'stable_contact_proxy':False},task_active=True)
    J1,A1,r1 = m._features(up,ref,Intent.PINCH,None)
    assert m.contact_phase=='APPROACH'
    assert np.linalg.norm(J1[:3]) < np.linalg.norm(J0[:3])
    assert np.linalg.norm(J1[3:]) > np.linalg.norm(J0[3:])
    m.update_contact(None,task_active=True,observed=False)
    J2,A2,r2 = m._features(up,ref,Intent.PINCH,None)
    assert np.allclose(J2,J0) and np.allclose(A2,A0) and np.allclose(r2,r0)


def test_idealized_linear_bound_covers_synthetic_noise():
    """Check the *conditional* mathematics, not real-world calibrated coverage."""
    rng = np.random.default_rng(1024)
    J = rng.normal(size=(6, 6)) * .015
    covariance = np.diag(np.array([.02, .01, .015, .012, .009, .016])**2)
    basis = np.eye(6)
    desired = np.array([.003, -.001, .004, .006, -.002, .003])
    k = 3
    alpha = .1
    residual, bound = CredibleContactSynergyMapper.conditional_bound(
        J, basis, k, covariance, desired, alpha)
    Vk = basis[:, :k]
    Ak = J @ Vk
    projection = Ak @ np.linalg.lstsq(Ak, desired, rcond=None)[0]
    error_vector = desired - projection
    task_noise = (J @ (Vk @ Vk.T) @ rng.multivariate_normal(
        np.zeros(6), covariance, size=5000).T).T
    observed = np.linalg.norm(error_vector - task_noise, axis=1)
    assert float(np.mean(observed <= bound)) >= 1 - alpha
    assert bound >= residual
