import numpy as np
from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.task_space_servo import ObjectAwareTaskServo
class Fake:
 def __init__(self,r): self.r=r; self.q=np.zeros(len(r.joint_order))
 def joint_positions(self): return self.q.copy()
 def fingertip_position(self,f): return np.zeros(3)
 def task_contact_target(self,finger,task_name,clearance_m,vertical_margin_m): return np.array([.01,0,0]) if finger=="thumb" else np.array([0,-.01,0])
 def fingertip_jacobian(self,finger,joints):
  n=len(joints); J=np.zeros((3,n)); J[0 if finger=="thumb" else 1,:]=np.linspace(.01,.006,n) * (1 if finger=="thumb" else -1); return J
def empty(): return {f"{f}_{k}":False for f in ("thumb","index","middle","ring","little") for k in ("contact","tip_contact")}
def test_inactive_servo_is_identity():
 r=RobotHandModel(); s=ObjectAwareTaskServo(r,load_yaml("configs/algorithm.yaml")["task_space_servo"]); q=np.zeros(11); out,d=s.apply(q,simulator=Fake(r),task_name="pinch",intent_value="NEUTRAL",contact=empty()); assert np.allclose(out,q); assert not d.active
def test_pinch_servo_supports_three_dof_thumb_and_two_dof_index():
 r=RobotHandModel(); s=ObjectAwareTaskServo(r,load_yaml("configs/algorithm.yaml")["task_space_servo"]); out,d=s.apply(np.zeros(11),simulator=Fake(r),task_name="pinch",intent_value="PINCH",contact=empty()); idx={n:i for i,n in enumerate(r.joint_order)}; assert d.corrected_digits==2; assert np.linalg.norm(out[[idx[n] for n in r.joints_for_finger("thumb")]])>0; assert np.linalg.norm(out[[idx[n] for n in r.joints_for_finger("index")]])>0; assert np.allclose(out[[idx[n] for n in r.joints_for_finger("middle")]],0)
def test_tip_contact_freezes_digit_at_actual_pose():
 r=RobotHandModel(); sim=Fake(r); ids=[r.joint_order.index(n) for n in r.joints_for_finger("thumb")]; sim.q[ids]=[.2,.1,.1]; c=empty(); c["thumb_tip_contact"]=True; out,_=ObjectAwareTaskServo(r,load_yaml("configs/algorithm.yaml")["task_space_servo"]).apply(np.zeros(11),simulator=sim,task_name="pinch",intent_value="PINCH",contact=c); assert np.allclose(out[ids],sim.q[ids])