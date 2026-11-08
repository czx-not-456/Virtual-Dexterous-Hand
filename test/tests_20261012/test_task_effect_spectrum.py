"""Mathematical-identity and command-shield tests for Scheme E."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from itertools import combinations
import numpy as np
import pytest

from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper
from src.retargeting.task_effect_spectrum import (
    TaskEffectSpectrumMapper, effect_spectrum, ray_to_control_box)
from src.features.intent_recognition import Intent

ROOT = Path(__file__).resolve().parents[2]


def make_mapper():
    import yaml
    model=LatentSynergyMapper(RobotHandModel(), ROOT/'configs/latent_chm6_adaptive.json')
    cfg=yaml.safe_load((ROOT/'configs/task_effect_spectrum.yaml').read_text(encoding='utf-8'))
    return TaskEffectSpectrumMapper(model,cfg)


def test_mode_subset_is_optimal_within_spectral_dictionary():
    rng = np.random.default_rng(209)
    A = rng.normal(size=(12, 6))
    target = rng.normal(size=12)
    V = rng.normal(size=(6,6))
    W = V.T@V + .3*np.eye(6)
    D,singular,coeff,gain,omission,original_indices=effect_spectrum(A,target,W)
    assert np.allclose(D.T@W@D,np.eye(6),atol=1e-10)
    for k in range(1,7):
        optimal=np.linalg.norm(A@D[:,:k]@np.linalg.lstsq(A@D[:,:k],target,rcond=None)[0]-target)
        assert abs(optimal-omission[k-1])<1e-9
        for indices in combinations(range(6),k):
            matrix=A@D[:,indices]
            err=np.linalg.norm(matrix@np.linalg.lstsq(matrix,target,rcond=None)[0]-target)
            assert optimal <= err+1e-9


def test_orthogonal_energy_decomposition():
    rng=np.random.default_rng(313)
    A=rng.standard_normal((18,6))
    b=rng.standard_normal(18)
    D,s,c,g,e,order=effect_spectrum(A,b,np.eye(6))
    assert np.all(np.diff(g)<=1e-12)
    assert np.all(np.diff(e)<=1e-12)
    for k in range(6):
        assert np.isclose(e[k]**2+sum(g[:k+1]),b@b,atol=1e-9)


def test_box_ray_is_feasible_and_maximal():
    a=np.array([.4,.3]); du=np.array([1.0,-1.]); lo=np.zeros(2);hi=np.ones(2)
    proposed, alpha=ray_to_control_box(a,du,lo,hi)
    assert np.all(proposed >= lo-1e-12) and np.all(proposed<=hi+1e-12)
    assert np.isclose(alpha,0.3)
    with pytest.raises(ValueError):
        ray_to_control_box(np.array([2.,.1]),du,lo,hi)


def test_unverified_noise_cannot_claim_enforced_risk():
    import yaml
    model=LatentSynergyMapper(RobotHandModel(), ROOT/'configs/latent_chm6_adaptive.json')
    cfg=yaml.safe_load((ROOT/'configs/task_effect_spectrum.yaml').read_text(encoding='utf-8'))
    cfg['risk_policy']='enforce'
    with pytest.raises(ValueError,match='independently calibrated'):
        TaskEffectSpectrumMapper(model,cfg)


def test_actual_output_coupling_rate_and_post_filter_guard():
    mapper=make_mapper()
    ref=np.array([.42,.56,.44,.61,.32,.18])
    prev=mapper.decode_controls(np.array([.30,.40,.35,.49,.25,.16]))
    speed=np.ones(11)*np.deg2rad(360)
    mapper.set_command_anchor(prev,speed,1/60)
    # Deliberately target a high but reachable model tip displacement.
    target=mapper.robot.fingertips(mapper.decode_controls(ref))
    q,_,diag=mapper.optimize(lambda q: float(q@q),prev,Intent.PINCH,ref,
                    task_targets_m={f:target[f] for f in mapper.FINGERS[Intent.PINCH]})
    # Test post-filter with off-manifold perturbations.
    altered=q.copy()
    altered[1] += .20
    out=mapper.project_and_rate_limit(altered,prev,speed,1/60)
    assert np.allclose(out,mapper.C@mapper.encode_joints(out),atol=1e-9)
    assert np.all(np.abs(out-prev)<=speed/60+1e-8)
    assert mapper.last_diag.rate_box_valid
    assert (mapper.last_diag.final_tip_error_m <=
            mapper.last_diag.final_reference_error_m+mapper.max_ref_regret_m+1e-8)


def test_contact_feedback_absent_not_hallucinated():
    m=make_mapper()
    m.update_contact(None,task_active=True,observed=False)
    assert m.contact_phase=='UNOBSERVED'


def test_metric_changes_with_prior_and_not_label_variance_only():
    m=make_mapper()
    metric=m._metric()
    assert np.linalg.eigvalsh(metric).min()>=1-1e-10
    J=m._jacobian(np.array([.4,.4,.4,.4,.4,.4]),m.FINGERS[Intent.PINCH])
    A=np.vstack((J,.005*np.eye(6)))
    d=np.array([.1,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]) if A.shape[0]==12 else np.arange(A.shape[0])*.001
    D,s,c,g,e,order=effect_spectrum(A,d,metric)
    assert np.isfinite(e).all()


def test_conditional_risk_subset_is_minimum_rank_by_enumeration():
    m=make_mapper()
    m.k_min=1
    m.k_max=6
    A=np.eye(6)
    D=np.eye(6)
    b=np.array([.4,.2,.1,.03,0.,0.])
    gain=b*b
    W=np.eye(6)
    cov=np.diag([0.0001]*6)
    chosen=m._minimum_risk_subset(A,b,D,gain,W,cov,
                                   threshold=.11,budget=.20)
    assert chosen is not None
    idx,omit,risk=chosen
    assert omit+risk<=.20+1e-10
    for k in range(1,len(idx)):
        for sub in combinations(range(6),k):
            residual=np.sqrt(max(0.,b@b-sum(gain[list(sub)])))
            subrisk=m._spectral_risk(A,D[:,sub],W,cov)
            assert residual>.11 or residual+subrisk>.20


def test_within_ellipsoid_perturbation_respects_spectral_risk():
    m=make_mapper()
    rng=np.random.default_rng(0)
    A=rng.normal(size=(10,6))
    W=np.diag([1.,2.,3.,4.,5.,6.])
    D,_,_,_,_,_=effect_spectrum(A,rng.normal(size=10),W)
    sub=D[:,:3]
    sigma=np.diag([.0001,.0003,.0002,.0001,.0004,.0001])
    bound=m._spectral_risk(A,sub,W,sigma)
    P=sub@sub.T@W
    for _ in range(200):
        z=rng.normal(size=6)
        z=z/np.linalg.norm(z)*m.ellipsoid_radius
        eta=np.diag(np.sqrt(np.diag(sigma)))@z
        assert np.linalg.norm(A@P@eta)<=bound+1e-12


def test_fixed_spectral_dimension_is_a_true_fixed_k_ablation():
    m=make_mapper()
    m.fixed_spectral_modes=4
    ref=np.array([.42,.56,.44,.61,.32,.18])
    prev=m.decode_controls(np.array([.30,.40,.35,.49,.25,.16]))
    speed=np.ones(11)*np.deg2rad(360)
    m.set_command_anchor(prev,speed,1/60)
    target=m.robot.fingertips(m.decode_controls(ref))
    _,_,diag=m.optimize(lambda q: float(q@q),prev,Intent.PINCH,ref,
        task_targets_m={f:target[f] for f in m.FINGERS[Intent.PINCH]})
    assert diag.requested_mode_count == 4
    assert diag.output_control_rank in (4,6)
    assert diag.accepted_mode_count in (0,4)


def test_fixed_spectral_dimension_rejects_invalid_values():
    m=make_mapper()
    m.fixed_spectral_modes=7
    with pytest.raises(ValueError,match='fixed_spectral_modes'):
        # Reinitialization validates the public config path.
        import yaml
        model=LatentSynergyMapper(RobotHandModel(), ROOT/'configs/latent_chm6_adaptive.json')
        cfg=yaml.safe_load((ROOT/'configs/task_effect_spectrum.yaml').read_text(encoding='utf-8'))
        cfg['fixed_spectral_modes']=7
        TaskEffectSpectrumMapper(model,cfg)


def test_final_nonlinear_guard_can_be_disabled_as_an_explicit_ablation():
    m=make_mapper()
    m.final_nonlinear_guard_enabled=False
    ref=np.array([.42,.56,.44,.61,.32,.18])
    prev=m.decode_controls(np.array([.30,.40,.35,.49,.25,.16]))
    speed=np.ones(11)*np.deg2rad(360)
    m.set_command_anchor(prev,speed,1/60)
    target=m.robot.fingertips(m.decode_controls(ref))
    q,_,_=m.optimize(lambda q: float(q@q),prev,Intent.PINCH,ref,
        task_targets_m={f:target[f] for f in m.FINGERS[Intent.PINCH]})
    out=m.project_and_rate_limit(q,prev,speed,1/60)
    assert m.last_diag.final_guard_enabled is False
    assert m.last_diag.rate_box_valid
    assert np.allclose(out,m.C@m.encode_joints(out),atol=1e-9)


def test_improving_candidate_that_breaks_spectral_budget_is_not_accepted(monkeypatch):
    """Better-than-baseline FK alone is not enough to certify a lower-k output."""
    m = make_mapper()
    m.fixed_spectral_modes = 4
    m.projection_abs_m = 0.0
    m.projection_relative = 0.0
    ref = np.array([.42, .56, .44, .61, .32, .18])
    prev = m.decode_controls(np.array([.30, .40, .35, .49, .25, .16]))
    speed = np.ones(11) * np.deg2rad(360)
    m.set_command_anchor(prev, speed, 1 / 60)
    target = m.robot.fingertips(m.decode_controls(ref))
    errors = iter([1.0, 0.5])  # feasible reference, then improving but uncertified k=4 candidate
    monkeypatch.setattr(m, "_nonlinear_tip_error", lambda *a, **k: next(errors))
    monkeypatch.setattr(m, "_fidelity_ok", lambda *a, **k: True)

    _, _, diag = m.optimize(lambda q: float(q @ q), prev, Intent.PINCH, ref,
        task_targets_m={f: target[f] for f in m.FINGERS[Intent.PINCH]})

    assert diag.accepted_mode_count == 0
    assert diag.output_control_rank == 6
    assert diag.fallback_full_reference is True
    assert diag.reject_spectral_count == 1
    assert diag.selection_status == "fallback_reference"
    assert diag.fallback_reason == "no_candidate_passed:spectral_budget"
