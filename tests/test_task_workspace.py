import math
import numpy as np
from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel
def world(v): return np.diag([-1.,1.,-1.])@v+np.array([0.,0.,.240])
def test_pinch_object_is_in_ch_m6_thumb_index_workspace():
 c=load_yaml("configs/simulation.yaml")["mujoco"]; p=c["task_objects"]["pinch"]; r=RobotHandModel(); tips={k:world(v) for k,v in r.fingertips(r.ranges.mean(1)).items()}; obj=np.asarray(p["pos"]); assert tips["index"][0]<obj[0]<tips["thumb"][0]; assert abs(obj[1]-tips["index"][1])<.015
def test_wrap_cylinder_spans_ch_m6_mid_pose():
 c=load_yaml("configs/simulation.yaml")["mujoco"]; s=c["task_objects"]["wrap"]; rad,hh=map(float,s["size"]); x,y,z=map(float,s["pos"]); assert abs(z-hh-.070)<1e-9; assert y-rad<=.065; assert y+rad>=.13
def test_ch_m6_task_catalog_is_well_formed():
 tasks=load_yaml("configs/simulation.yaml")["mujoco"]["task_objects"]; assert set(tasks)=={"neutral","pinch","wrap"}; assert {v["active_intent"] for v in tasks.values()}=={"NEUTRAL","PINCH","WRAP"}; assert all(len(v["pos"])==3 and all(float(x)>0 for x in v["size"]) for v in tasks.values())
def test_ch_m6_pinch_posture_prior_is_near_object():
 a=load_yaml("configs/algorithm.yaml")["task_space_servo"]["pinch"]; c=load_yaml("configs/simulation.yaml")["mujoco"]; r=RobotHandModel(); q=np.zeros(11); idx={n:i for i,n in enumerate(r.joint_order)}
 for j,d in a["posture_prior_deg"].items(): q[idx[j]]=math.radians(float(d))
 tips={k:world(v) for k,v in r.fingertips(q).items()}; obj=np.asarray(c["task_objects"]["pinch"]["pos"]); assert np.linalg.norm(tips["thumb"]-obj)<.06; assert np.linalg.norm(tips["index"]-obj)<.06