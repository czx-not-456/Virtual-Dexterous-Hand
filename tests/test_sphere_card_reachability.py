from src.common import ROOT,load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.simulation.ch_m6_adapter import build_ch_m6_scene
def test_ch_m6_all_stl_assets_are_supplied_in_memory():
 c=load_yaml("configs/simulation.yaml")["mujoco"]; _,a=build_ch_m6_scene(ROOT/c["model_path"],RobotHandModel(),c); assert len(a)==12; assert all(k.startswith("meshes/") and v for k,v in a.items())
def test_only_required_project_tasks_are_configured(): assert set(load_yaml("configs/simulation.yaml")["mujoco"]["task_objects"])=={"neutral","pinch","wrap"}