from pathlib import Path
from xml.etree import ElementTree as ET

from src.common import ROOT, load_yaml


EXPECTED_JOINTS = [
    "thumb_mcp", "thumb_pip", "thumb_dip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "little_mcp", "little_pip", "little_dip",
]
EXPECTED_ACTUATORS = {
    "act_thumb_mcp", "act_thumb_pip", "act_thumb_dip",
    "act_index_flexor", "act_middle_flexor", "act_ring_flexor", "act_little_flexor",
}


def _load_root():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    path = Path(ROOT / cfg["model_path"])
    return cfg, path, ET.parse(path).getroot()


def test_v03_model_contract_15dof_7actuators():
    cfg, path, root = _load_root()
    assert path.is_file()
    assert root.attrib["model"] == "humanoid_virtual_dexterous_hand_v034_object_aware"

    joints = {j.attrib.get("name") for j in root.findall(".//joint") if j.attrib.get("name")}
    actuators = {a.attrib.get("name") for a in root.findall("./actuator/position") if a.attrib.get("name")}
    tendons = {t.attrib.get("name") for t in root.findall("./tendon/fixed") if t.attrib.get("name")}

    assert set(EXPECTED_JOINTS).issubset(joints)
    assert actuators == EXPECTED_ACTUATORS
    assert tendons == {"index_flexor_tendon", "middle_flexor_tendon", "ring_flexor_tendon", "little_flexor_tendon"}

    cameras = {c.attrib.get("name") for c in root.findall(".//camera")}
    assert {"overview", "closeup", "side"}.issubset(cameras)
    assert {"overview", "closeup", "side"}.issubset(set(cfg["camera_presets"]))
    assert cfg["default_camera"] in cfg["camera_presets"]

    bodies = {b.attrib.get("name") for b in root.findall(".//body")}
    assert "palm" in bodies
    assert cfg["task_object_body"] in bodies


def test_v03_non_thumb_joint_springs_match_literature_values():
    _, _, root = _load_root()
    joint_by_name = {j.attrib["name"]: j for j in root.findall(".//joint") if "name" in j.attrib}
    for finger in ("index", "middle", "ring", "little"):
        assert abs(float(joint_by_name[f"{finger}_mcp"].attrib["stiffness"]) - 0.024106) < 1e-9
        assert abs(float(joint_by_name[f"{finger}_pip"].attrib["stiffness"]) - 0.012857) < 1e-9
        assert abs(float(joint_by_name[f"{finger}_dip"].attrib["stiffness"]) - 0.012857) < 1e-9


def test_v03_flexion_axis_matches_python_kinematics_sign():
    _, _, root = _load_root()
    joint_by_name = {j.attrib["name"]: j for j in root.findall(".//joint") if "name" in j.attrib}
    for name in ("thumb_mcp", "thumb_pip", "thumb_dip"):
        assert joint_by_name[name].attrib.get("axis") == "1 0 0"
    for name in EXPECTED_JOINTS:
        if not name.startswith("thumb_"):
            assert joint_by_name[name].attrib.get("axis") == "-1 0 0"


def test_v03_has_contact_pads_and_task_scene():
    _, _, root = _load_root()
    sites = {s.attrib.get("name") for s in root.findall(".//site")}
    assert {
        "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip",
        "pinch_focus", "object_center",
    }.issubset(sites)

    geoms = {g.attrib.get("name") for g in root.findall(".//geom") if g.attrib.get("name")}
    assert "task_object_geom" in geoms
    assert "floor" in geoms
    assert {"thumb_pad", "index_pad", "middle_pad", "ring_pad", "little_pad"}.issubset(geoms)
    assert {"palm_core", "palm_heel", "thenar", "hypothenar"}.issubset(geoms)


def test_pinch_focus_site_is_retained_but_invisible():
    _, _, root = _load_root()
    site = next(s for s in root.findall(".//site") if s.attrib.get("name") == "pinch_focus")
    assert site.attrib["pos"] == "-0.037 0.003 0.184"
    assert float(site.attrib["rgba"].split()[3]) == 0.0


def test_v031_has_exactly_five_digit_roots_and_no_forearm_digit():
    _, _, root = _load_root()
    palm = next(b for b in root.findall(".//body") if b.attrib.get("name") == "palm")
    child_names = {b.attrib.get("name") for b in palm.findall("./body")}
    assert child_names == {"thumb1", "index1", "middle1", "ring1", "little1"}

    geom_names = {g.attrib.get("name") for g in palm.findall("./geom") if g.attrib.get("name")}
    assert "forearm" not in geom_names
    assert "wrist_base" in geom_names


def test_v031_fixed_tendons_use_only_supported_structural_attributes():
    _, _, root = _load_root()
    for tendon in root.findall("./tendon/fixed"):
        assert "width" not in tendon.attrib
        assert "rgba" not in tendon.attrib
