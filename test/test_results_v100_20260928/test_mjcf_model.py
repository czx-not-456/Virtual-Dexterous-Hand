from pathlib import Path
from xml.etree import ElementTree as ET
from src.common import ROOT, load_yaml
from src.hand_model.robot_hand import RobotHandModel
from src.simulation.ch_m6_adapter import build_ch_m6_scene

EXPECTED=["th_cmc_joint","th_mcp_joint","th_ip_joint","ff_mcp_joint","ff_dip_joint","mf_mcp_joint","mf_dip_joint","rf_mcp_joint","rf_dip_joint","lf_mcp_joint","lf_dip_joint"]
def _built():
    cfg=load_yaml("configs/simulation.yaml")["mujoco"]; p=ROOT/cfg["model_path"]; text,assets=build_ch_m6_scene(p,RobotHandModel(),cfg); return cfg,p,ET.fromstring(text),assets

def test_teacher_source_is_ch_m6_11_joint_model_without_project_edits():
    cfg,p,_,_= _built(); raw=ET.parse(p).getroot(); joints=[j.get("name") for j in raw.findall(".//joint") if j.get("name")]
    acts=[a.get("name") for a in raw.findall("./actuator/position")]
    assert joints==EXPECTED; assert acts==EXPECTED; assert raw.findall(".//site")==[]
    assert not any(b.get("name")==cfg["task_object_body"] for b in raw.iter("body"))

def test_runtime_adapter_adds_scene_sites_and_contact_regions():
    cfg,p,root,assets=_built(); assert p.is_file(); assert len(assets)==12
    sites={x.get("name") for x in root.findall(".//site")}; geoms={x.get("name") for x in root.findall(".//geom")}
    assert {"thumb_tip","index_tip","middle_tip","ring_tip","little_tip","object_center"}<=sites
    assert {"thumb_pad","index_pad","middle_pad","ring_pad","little_pad","task_object_geom","floor"}<=geoms
    assert any(b.get("name")==cfg["task_object_body"] for b in root.iter("body"))

def test_all_ch_m6_actuators_are_independent_position_controls():
    _,_,root,_=_built(); acts=root.findall("./actuator/position")
    assert len(acts)==11; assert all(a.get("joint")==a.get("name") for a in acts)
    assert root.find("tendon") is None