import numpy as np
from src.common import load_yaml
from src.features.feature_extractor import FeatureExtractor
from src.features.intent_recognition import Intent
from src.hand_model.human_hand import HumanHandModel
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.optimizer import IntentDrivenRetargeter

def _solver():
    h=HumanHandModel(); r=RobotHandModel(); return h,r,IntentDrivenRetargeter(h,r,load_yaml("configs/algorithm.yaml")["optimizer"])

def test_retargeting_respects_ch_m6_joint_limits():
    h,r,s=_solver(); qh=h.ranges[:,0]+.7*(h.ranges[:,1]-h.ranges[:,0]); f=FeatureExtractor(h).extract(qh); out=s.solve(qh,f.fingertip_positions,Intent.WRAP)
    assert out.q_target.shape==(11,); assert np.all(out.q_target>=r.ranges[:,0]-1e-9); assert np.all(out.q_target<=r.ranges[:,1]+1e-9)

def test_dynamic_pinch_is_finite_and_native_dimension():
    h,r,s=_solver(); qh=h.ranges.mean(axis=1); f=FeatureExtractor(h).extract(qh); out=s.solve(qh,f.fingertip_positions,Intent.PINCH)
    assert out.q_target.shape==(11,); assert np.isfinite(out.objective); assert all(np.isfinite(v) for v in out.components.values())

def test_unconfirmed_hardware_synergy_penalty_is_disabled():
    from src.retargeting.constraints import coupling_error
    _,r,_=_solver(); assert r.synergies=={}; assert coupling_error(r.ranges.mean(axis=1),r)==0.0