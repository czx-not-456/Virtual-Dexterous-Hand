"""Stage C task-subspace and fault-injection invariants; no MuJoCo required."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.tqhg_stagec import (
    StageCOnlineMapper, parse_fault_channels, fault_mask_for_frame,
    _conditioned_projector_weight, STAGEC_METHODS)
from src.retargeting.task_weighted_hybrid import task_projector


class PinchJacStub:
    def fingertip_jacobian(self, finger, joint_names):
        J=np.zeros((3,len(joint_names)))
        J[0,:]=np.arange(1,len(joint_names)+1)
        J[1,:]=np.linspace(1,0,len(joint_names))
        return J


@pytest.fixture
def mapper():
    return StageCOnlineMapper(RobotHandModel(),
        'configs/tqhg_online_v2.json',
        'configs/stageB_robust_ridge6.json',
        'configs/stageC_task_gate.json')


def test_fault_schedule_end_exclusive_and_1_based():
    channels=parse_fault_channels('1,4,15')
    assert channels == (1,4,15)
    for frame in (0,119,240,320):
        assert fault_mask_for_frame(frame, channels=channels, start=120, stop=240).all()
    mask=fault_mask_for_frame(120, channels=channels, start=120, stop=240)
    assert np.where(~mask)[0].tolist()==[0,3,14]
    for invalid in ('0','16','1,1','1,hi'):
        with pytest.raises(ValueError):parse_fault_channels(invalid)


def test_paired_calibration_uses_training_matched_experts(mapper):
    c=mapper.stagec_gates
    assert c['schema']=='stageC_task_gate_v1'
    assert c['provenance']['train_count']==124
    assert c['provenance']['calibration_count']==66
    for n in ('0','1','3','5'):
        a,b=np.asarray(c['errors'][n]['a']),np.asarray(c['errors'][n]['b'])
        assert a.shape==b.shape and a.shape[1]==6
        assert len(a)==(66 if n=='0' else 792)


def test_task_and_null_projectors_orthogonal(mapper):
    J=mapper._jacobian(PinchJacStub(),'pinch')
    P=task_projector(J);N=np.eye(6)-P
    assert 1<=np.trace(P)<=5
    assert np.allclose(P@P,P,atol=1e-10)
    assert np.allclose(N@N,N,atol=1e-10)
    assert np.linalg.norm(J@N) < 1e-9


def test_five_methods_use_same_models_finite_outputs(mapper):
    s=mapper.sensor_min+.46*mapper.sensor_span
    mask=np.ones(15,bool); mask[[1,5,8]]=False
    for method in STAGEC_METHODS:
        mapper.reset()
        u=mapper.predict(s,method=method,simulator=PinchJacStub(),sensor_valid_mask=mask)
        assert u.shape==(6,) and np.isfinite(u).all()
        assert np.all(u>=mapper.lower-1e-9) and np.all(u<=mapper.upper+1e-9)
        assert mapper.last_gate['invalid_channels']==3
        assert mapper.last_gate['stagec_projection_leakage']<1e-8
        assert np.allclose(mapper.decode_controls(u),mapper.C@u)
        if method=='stagec-alpha': assert mapper.last_gate['w_null']==0.
        if method=='stagec-beta': assert mapper.last_gate['w_task']==0.
        if method in ('robust6','robust4'):
            assert mapper.last_gate['w_task']==mapper.last_gate['w_null']==0.


def test_healthy_stationary_not_marked_fault(mapper):
    s=mapper.sensor_min+.45*mapper.sensor_span
    for _ in range(75):mapper.predict(s,method='stagec-dual',simulator=PinchJacStub())
    assert mapper.last_gate['invalid_channels']==0
    assert mapper.last_gate['sensor_reliability']==pytest.approx(1.)


def test_missing_channel_recovery_changes_mask(mapper):
    s=mapper.sensor_min+.46*mapper.sensor_span
    for frame in range(115,245):
        mask=fault_mask_for_frame(frame,channels=(1,4,7),start=120,stop=240)
        mapper.predict(s,method='stagec-dual',simulator=PinchJacStub(),sensor_valid_mask=mask)
        assert mapper.last_gate['invalid_channels']==(3 if 120<=frame<240 else 0)


def test_no_mujoco_alpha_is_not_fabricated(mapper):
    s=mapper.sensor_min+.4*mapper.sensor_span
    mapper.predict(s,method='stagec-dual',simulator=None)
    assert mapper.last_gate['visible_rank']==0
    assert mapper.last_gate['w_task']==0


def test_missing_ten_channels_safe_fallback(mapper):
    s=mapper.sensor_min+.35*mapper.sensor_span
    previous=mapper.predict(s,method='stagec-dual',simulator=PinchJacStub()).copy()
    mask=np.ones(15,bool); mask[:10]=False
    outcome=mapper.predict(s,method='stagec-dual',simulator=PinchJacStub(),sensor_valid_mask=mask)
    assert mapper.last_gate['gate_fallback']==1
    assert np.allclose(previous,outcome)


def test_alpha_positive_only_when_calibration_supports():
    d={}
    for key in ('0','1','3','5'):
        d[key]={'a':np.full((24,6),.15).tolist(), 'b':np.zeros((24,6)).tolist()}
    P=np.eye(6)
    assert _conditioned_projector_weight(d,P,3)>0.95
    for key in ('0','1','3','5'):
        d[key]={'a':np.zeros((24,6)).tolist(), 'b':np.full((24,6),.15).tolist()}
    assert _conditioned_projector_weight(d,P,3)<.05
