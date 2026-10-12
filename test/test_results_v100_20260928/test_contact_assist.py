import numpy as np
from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.contact_assist import TaskContactAssist

def empty(): return {f"{f}_{k}":False for f in ("thumb","index","middle","ring","little") for k in ("contact","tip_contact")}
def test_pinch_assist_closes_only_ch_m6_thumb_and_index():
 r=RobotHandModel(); a=TaskContactAssist(r,load_yaml("configs/algorithm.yaml")["contact_assist"],1/60); out=a.apply(np.zeros(11),task_name="pinch",intent_value="PINCH",contact=empty()); i={n:k for k,n in enumerate(r.joint_order)}; assert out[i["th_mcp_joint"]]>0; assert out[i["ff_dip_joint"]]>0; assert out[i["mf_dip_joint"]]==0
def test_contact_stops_only_contacted_digit():
 r=RobotHandModel(); a=TaskContactAssist(r,load_yaml("configs/algorithm.yaml")["contact_assist"],1/60); c=empty(); c["index_tip_contact"]=True; out=a.apply(np.zeros(11),task_name="pinch",intent_value="PINCH",contact=c); i={n:k for k,n in enumerate(r.joint_order)}; assert out[i["ff_dip_joint"]]==0; assert out[i["th_mcp_joint"]]>0
def test_assist_releases_after_intent_ends():
 r=RobotHandModel(); a=TaskContactAssist(r,load_yaml("configs/algorithm.yaml")["contact_assist"],1/60); q=np.zeros(11); first=a.apply(q,task_name="pinch",intent_value="PINCH",contact=empty()); second=a.apply(q,task_name="pinch",intent_value="NEUTRAL",contact=empty()); assert np.linalg.norm(second)<np.linalg.norm(first)