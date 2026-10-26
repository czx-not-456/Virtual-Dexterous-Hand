"""TQHG-v2 safety/compatibility and gate tests. No MuJoCo installation needed."""
from pathlib import Path
import json
import numpy as np
import pytest
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.tqhg_v2 import (
    TQHGv2OnlineMapper, SensorQuality, empirical_weight, conditioned_weight,
    masked_features,
)

CAL=Path('configs/tqhg_online_v2.json')

@pytest.fixture
def m():
    assert CAL.exists(), 'Run python scripts/train_tqhg_v2.py'
    return TQHGv2OnlineMapper(RobotHandModel(),CAL)

class PinchStub:
    def fingertip_jacobian(self, finger, joint_names):
        arr=np.zeros((3,len(joint_names)))
        arr[0,:]=np.arange(1,len(joint_names)+1)
        arr[1,:]=np.linspace(1,0,len(joint_names))
        return arr


def test_old_schema_preserved(m):
    assert m.cfg['schema']=='tqhg_online_calibration_v1'
    assert m.v2_cfg['schema']=='tqhg_online_calibration_v2'
    assert m.v2_cfg['provenance']['train_count']==124
    assert m.v2_cfg['provenance']['calibration_count']==66


def test_clean_command_is_six_dimensional_coupled_and_bounded(m):
    sensors=m.sensor_min+.40*m.sensor_span
    u=m.predict(sensors,method='tqhg-v2',simulator=PinchStub())
    assert u.shape==(6,)
    assert np.isfinite(u).all()
    assert np.all(u>=m.lower-1e-9) and np.all(u<=m.upper+1e-9)
    q=m.decode_controls(u)
    assert np.allclose(q,m.C@u,atol=1e-10)
    assert np.all(q>=m.robot.ranges[:,0]-1e-9)
    assert np.all(q<=m.robot.ranges[:,1]+1e-9)
    assert m.last_gate['visible_rank'] in (1,2,3,4,5,6)
    assert m.last_gate['invalid_channels']==0
    assert m.last_gate['sensor_reliability']==pytest.approx(1.)


def test_explicit_mask_changes_reliability(m):
    s=m.sensor_min+.4*m.sensor_span
    mask=np.ones(15,bool); mask[[0,4,8]]=False
    u=m.predict(s,method='tqhg-v2',simulator=PinchStub(),sensor_valid_mask=mask)
    assert np.isfinite(u).all()
    assert m.last_gate['invalid_channels']==3
    assert m.last_gate['sensor_reliability']==pytest.approx(.8)
    assert m.last_gate['gate_severity']>0


def test_unknown_missing_data_no_crash(m):
    s=m.sensor_min+.4*m.sensor_span
    s[1]=np.nan;s[2]=float('inf')
    u=m.predict(s,method='tqhg-v2',simulator=PinchStub())
    assert np.isfinite(u).all()
    assert m.last_gate['invalid_channels']==2


def test_10_missing_triggers_bounded_fallback(m):
    s=m.sensor_min+.5*m.sensor_span
    before=m.predict(s,method='tqhg-v2',simulator=PinchStub())
    s[:11]=np.nan
    after=m.predict(s,method='tqhg-v2',simulator=PinchStub())
    assert m.last_gate['gate_fallback']==1
    assert np.allclose(before,after)


def test_stationary_finger_not_mistaken_for_stuck(m):
    s=m.sensor_min+.35*m.sensor_span
    for _ in range(80):m.predict(s,method='tqhg-v2',simulator=PinchStub())
    assert m.last_gate['invalid_channels']==0
    assert m.last_gate['sensor_reliability']==pytest.approx(1.)


def test_favorable_expert_b_gets_nonzero_gate():
    n=40;H=np.eye(6)
    a=np.full((n,6),.15)
    b=np.zeros_like(a)
    w=conditioned_weight(a,b,a,b,H,1.)
    assert w>0.85
    w2=empirical_weight(a,b,H)
    assert w2>0.85


def test_worse_expert_gets_zero_gate():
    n=40;H=np.eye(6)
    a=np.zeros((n,6));b=np.full((n,6),.2)
    assert conditioned_weight(a,b,a,b,H,1.)<.05


def test_mask_features_has_expected_shape():
    x=np.full((4,15),.6)
    mask=np.ones((4,15),bool);mask[:,3]=False
    z=masked_features(x,mask,np.full(15,.4))
    assert z.shape==(4,30)
    assert np.allclose(z[:,3],.4)
    assert np.all(z[:,18]==0.)


def test_rate_guard_compatible(m):
    s=m.sensor_min+.38*m.sensor_span
    u=m.predict(s,method='tqhg-v2',simulator=PinchStub())
    start=m.decode_controls(m.lower+.03*(m.upper-m.lower))
    vmax=np.full(11,np.deg2rad(100.))
    q=m.project_and_rate_limit(m.decode_controls(u),start,vmax,1/60)
    assert np.max(np.abs(q-start)-vmax/60)<1.e-9
    assert np.allclose(m.decode_controls(m.encode_joints(q)),q,atol=1.e-7)


def test_failed_shape_rejected(m):
    with pytest.raises(ValueError):m.predict(np.zeros(6),method='tqhg-v2')
    with pytest.raises(ValueError):m.predict(np.zeros(15),method='tqhg-v2',sensor_valid_mask=np.ones(14))
