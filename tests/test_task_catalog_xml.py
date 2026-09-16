import xml.etree.ElementTree as ET

from src.common import ROOT, load_yaml
from src.simulation.mujoco_env import MujocoSimulator


def test_all_v040_task_specs_can_patch_base_mjcf_without_mujoco_runtime():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    xml_text = (ROOT / cfg["model_path"]).read_text(encoding="utf-8")
    for name, spec in cfg["task_objects"].items():
        patched = MujocoSimulator._configure_task_xml(xml_text, cfg["task_object_body"], spec)
        root = ET.fromstring(patched)
        body = next(b for b in root.iter("body") if b.get("name") == cfg["task_object_body"])
        geom = next(g for g in body.findall("geom") if g.get("name") == "task_object_geom")
        assert geom.get("type") == spec["type"], name
        assert len(geom.get("size", "").split()) == len(spec["size"]), name
