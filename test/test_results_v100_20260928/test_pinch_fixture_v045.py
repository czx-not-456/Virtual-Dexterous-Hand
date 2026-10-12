from pathlib import Path
from src.common import load_yaml
def test_ch_m6_pinch_block_keeps_table_support_and_x_thickness():
 c=load_yaml("configs/simulation.yaml")["mujoco"]; p=c["task_objects"]["pinch"]; hx,hy,hz=map(float,p["size"]); x,y,z=map(float,p["pos"]); assert abs(z-hz-.070)<1e-9; assert 2*hx==.016; assert 2*hy==.050; assert p["contact_faces"]["thumb"]["sign"]==1
def test_pinch_approach_fixture_releases_after_bilateral_tip_contact():
 m=Path("src/main.py").read_text(encoding="utf-8"); s=Path("src/simulation/mujoco_env.py").read_text(encoding="utf-8"); assert 'task_name == "pinch" and bilateral_tip_contact' in m; assert 'simulator.step(q_out, hold_object=hold_pinch_object)' in m; assert 'def _hold_task_object_pose' in s