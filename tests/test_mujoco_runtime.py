from pathlib import Path

import pytest

from src.common import ROOT, load_yaml


mujoco = pytest.importorskip("mujoco", reason="MuJoCo runtime is optional in CI/generation environment")


def test_v03_mujoco_schema_loads_and_named_entities_exist():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    path = Path(ROOT / cfg["model_path"])
    model = mujoco.MjModel.from_xml_string(path.read_text(encoding="utf-8"))

    assert model.nu == 7
    assert model.nv >= 21  # 15 hand hinge DoF + free object 6 DoF
    for name in ("overview", "closeup", "side"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "grasp_object") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm") >= 0
    for name in ("index_flexor_tendon", "middle_flexor_tendon", "ring_flexor_tendon", "little_flexor_tendon"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name) >= 0
