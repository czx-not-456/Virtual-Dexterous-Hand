from src.common import load_yaml
def test_ch_m6_task_objects_rest_on_table():
 c=load_yaml("configs/simulation.yaml")["mujoco"]; assert abs(c["table"]["pos"][2]+c["table"]["size"][2]-.070)<1e-9
 p=c["task_objects"]["pinch"]; w=c["task_objects"]["wrap"]; assert abs(p["pos"][2]-p["size"][2]-.070)<1e-9; assert abs(w["pos"][2]-w["size"][1]-.070)<1e-9
def test_pinch_block_uses_ch_m6_x_opposition():
 p=load_yaml("configs/simulation.yaml")["mujoco"]["task_objects"]["pinch"]; assert 2*p["size"][0]==.016; assert p["contact_faces"]["thumb"]["axis"]==0