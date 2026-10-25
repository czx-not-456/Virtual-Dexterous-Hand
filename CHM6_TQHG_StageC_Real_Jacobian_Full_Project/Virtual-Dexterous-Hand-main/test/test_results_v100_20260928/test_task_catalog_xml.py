import xml.etree.ElementTree as ET
from src.common import ROOT, load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.simulation.ch_m6_adapter import build_ch_m6_scene
from src.simulation.mujoco_env import MujocoSimulator

def test_neutral_pinch_wrap_specs_patch_runtime_scene():
 c=load_yaml("configs/simulation.yaml")["mujoco"]; xml,_=build_ch_m6_scene(ROOT/c["model_path"],RobotHandModel(),c); assert set(c["task_objects"])=={"neutral","pinch","wrap"}
 for name,spec in c["task_objects"].items():
  root=ET.fromstring(MujocoSimulator._configure_task_xml(xml,c["task_object_body"],spec)); body=next(b for b in root.iter("body") if b.get("name")==c["task_object_body"]); geom=next(g for g in body.findall("geom") if g.get("name")=="task_object_geom"); assert geom.get("type")==spec["type"],name