import numpy as np
from src.common import load_yaml
from src.hand_model.robot_hand import RobotHandModel

def world_tip(local): return np.diag([-1.,1.,-1.])@local+np.array([0.,0.,.240])
def test_pinch_block_lies_between_mid_pose_thumb_and_index():
 r=RobotHandModel(); tips={k:world_tip(v) for k,v in r.fingertips(r.ranges.mean(axis=1)).items()}; c=np.asarray(load_yaml("configs/simulation.yaml")["mujoco"]["task_objects"]["pinch"]["pos"]); assert tips["index"][0]<c[0]<tips["thumb"][0]; assert abs(tips["index"][1]-c[1])<.015
def test_ch_m6_pinch_faces_are_opposed_on_x_axis():
 s=load_yaml("configs/simulation.yaml")["mujoco"]["task_objects"]["pinch"]; assert s["contact_faces"]["thumb"]=={"axis":0,"sign":1}; assert s["contact_faces"]["index"]=={"axis":0,"sign":-1}