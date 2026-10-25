"""One-factor ablations of coupled servo; no MuJoCo or GPU needed."""
import numpy as np
import pytest

from src.hand_model.robot_hand import RobotHandModel
from src.common import load_yaml
from src.retargeting.latent_synergy import LatentSynergyMapper
from src.retargeting.task_effect_spectrum import TaskEffectSpectrumMapper
from src.retargeting.task_space_servo import ObjectAwareTaskServo


class Fake:
    def __init__(self, robot, q):
        self.q = np.asarray(q, float).copy()
    def joint_positions(self):
        return self.q.copy()
    def fingertip_position(self, finger):
        return np.zeros(3)
    def task_contact_target(self, finger, task_name, clearance_m, vertical_margin_m):
        return np.array([.02, .01, 0]) if finger == "thumb" else np.array([0, -.015, .008])
    def fingertip_jacobian(self, finger, joint_names):
        n=len(joint_names)
        arr=np.zeros((3,n)); arr[0,:]=np.linspace(.02,.01,n)
        arr[1,:]=np.linspace(-.005,.01,n)
        return arr


def setup():
    robot=RobotHandModel()
    base=LatentSynergyMapper(robot, "configs/latent_chm6_adaptive.json")
    mapper=TaskEffectSpectrumMapper(base, load_yaml("configs/task_effect_spectrum.yaml"))
    cfg=load_yaml("configs/algorithm.yaml")["task_space_servo"]
    q_nom=mapper.decode_controls(np.array([.82, .6, .3, .2, .2, .2]))
    q_act=mapper.decode_controls(np.array([.58, .3, .25, .2, .2, .2]))
    return robot, mapper, cfg, q_nom, Fake(robot, q_act)


def run(mode, contact):
    r,m,c,q,sim=setup()
    s=ObjectAwareTaskServo(r,c,coupled_ablation=mode)
    out,diag=s.apply_coupled(q,simulator=sim,task_name="pinch",intent_value="PINCH",contact=contact,mapper=m)
    u_out=m.encode_joints(out)
    assert np.all(np.isfinite(out))
    assert np.allclose(m.C@u_out,out,atol=1e-9)
    assert np.all(u_out>=m.lower-1e-9) and np.all(u_out<=m.upper+1e-9)
    assert diag.active
    return r,m,q,sim,out


def test_prior_only_changes_control_but_preserves_coupling():
    _,_,_,_,a=run("original",{})
    _,_,_,_,b=run("prior",{})
    assert np.linalg.norm(a-b)>1e-7


def test_thumb_contact_hold_matches_actual_six_control_pose():
    _,m,_,sim,a=run("original",{"thumb_tip_contact":True})
    r,m,q,sim,b=run("hold",{"thumb_tip_contact":True})
    thumb_idx=[r.joint_order.index(n) for n in r.joints_for_finger("thumb")]
    # No per-joint decoupling; C@u_actual preserves all CH-M6 software couplings.
    projected_actual=m.decode_controls(m.encode_joints(sim.q))
    assert np.allclose(b[thumb_idx],projected_actual[thumb_idx],atol=1e-8)
    assert np.linalg.norm(b[thumb_idx]-a[thumb_idx])>1e-4


def test_both_and_inactive_valid():
    r,m,q,sim,out=run("both",{"thumb_tip_contact":True})
    s=ObjectAwareTaskServo(r,load_yaml("configs/algorithm.yaml")["task_space_servo"],coupled_ablation="both")
    x,diag=s.apply_coupled(q,simulator=sim,task_name="pinch",intent_value="NEUTRAL",contact={},mapper=m)
    assert np.allclose(x,q) and not diag.active


def test_invalid_ablation_rejected():
    r,m,_,q,sim=setup()
    with pytest.raises(ValueError):
        ObjectAwareTaskServo(r,{},coupled_ablation="unknown")
