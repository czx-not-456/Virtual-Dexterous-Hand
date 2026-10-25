from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
from src.hand_model.robot_hand import RobotHandModel


def _vec(values) -> str:
    return " ".join(str(float(x)) for x in values)


def build_ch_m6_scene(source_path: Path, robot: RobotHandModel, scene_cfg: dict) -> tuple[str, dict[str, bytes]]:
    """Build the project scene in memory without changing the third-party CH-M6 files."""
    source_path = source_path.resolve()
    root = ET.fromstring(source_path.read_text(encoding="utf-8"))
    root.set("model", "ch_m6_project_scene")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        compiler = root.find("compiler")
        root.insert(list(root).index(compiler) + 1 if compiler is not None else 0, option)
    option.set("timestep", str(float(scene_cfg.get("physics_timestep_s", 0.002))))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")

    world = root.find("worldbody")
    if world is None:
        raise ValueError("CH-M6 source has no worldbody")
    palm_name = str(robot.simulation_names["palm_body"])
    palm = next((b for b in world.iter("body") if b.get("name") == palm_name), None)
    if palm is None:
        raise ValueError(f"CH-M6 source has no palm body {palm_name}")
    palm.set("pos", _vec(scene_cfg["palm_pos_m"]))
    palm.set("quat", _vec(scene_cfg["palm_quat_wxyz"]))

    for body in world.iter("body"):
        for i, geom in enumerate(body.findall("geom")):
            if not geom.get("name"):
                name = str(robot.simulation_names["palm_geom"]) if body.get("name") == palm_name else f"{body.get('name')}_collision_{i}"
                geom.set("name", name)

    chains = robot.cfg["kinematic_chains"]
    names = robot.simulation_names
    for finger, body_name in names["fingertip_bodies"].items():
        body = next((b for b in world.iter("body") if b.get("name") == body_name), None)
        if body is None:
            raise ValueError(f"CH-M6 source has no fingertip body {body_name}")
        pos = chains[finger]["tip_local_m"]
        radius = float(names["fingertip_radius_m"][finger])
        ET.SubElement(body, "site", {
            "name": str(names["fingertip_sites"][finger]), "type": "sphere",
            "pos": _vec(pos), "size": str(radius), "rgba": "0.95 0.45 0.12 0.35",
        })
        ET.SubElement(body, "geom", {
            "name": str(names["fingertip_geoms"][finger]), "type": "sphere",
            "pos": _vec(pos), "size": str(radius), "rgba": "0.95 0.45 0.12 0.30",
            "density": "250", "friction": "1.3 0.1 0.01", "contype": "1", "conaffinity": "1",
        })

    table = scene_cfg["table"]
    ET.SubElement(world, "geom", {
        "name": "floor", "type": "plane", "size": "0.6 0.6 0.02",
        "rgba": "0.15 0.17 0.20 1", "friction": "1.0 0.1 0.01",
    })
    table_body = ET.SubElement(world, "body", {"name": "task_table", "pos": _vec(table["pos"])})
    ET.SubElement(table_body, "geom", {
        "name": "task_table_geom", "type": "box", "size": _vec(table["size"]),
        "rgba": "0.28 0.30 0.34 1", "friction": "1.0 0.1 0.01",
    })
    obj = ET.SubElement(world, "body", {"name": str(scene_cfg["task_object_body"]), "pos": "0 0 0.1"})
    ET.SubElement(obj, "freejoint", {"name": "task_object_free"})
    ET.SubElement(obj, "geom", {
        "name": "task_object_geom", "type": "sphere", "size": "0.025",
        "density": "300", "rgba": "0.20 0.65 0.95 1", "friction": "1.35 0.12 0.01",
    })
    ET.SubElement(obj, "site", {"name": "object_center", "type": "sphere", "size": "0.003", "rgba": "1 1 1 0.8"})

    assets = {f"meshes/{p.name}": p.read_bytes() for p in (source_path.parent / "meshes").glob("*.STL")}
    if len(assets) != len(root.findall("./asset/mesh")):
        raise ValueError("CH-M6 STL asset set is incomplete")
    return ET.tostring(root, encoding="unicode"), assets