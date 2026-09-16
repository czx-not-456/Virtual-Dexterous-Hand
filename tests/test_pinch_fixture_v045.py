from pathlib import Path

from src.common import load_yaml


def test_pinch_block_is_wider_taller_and_keeps_table_support():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]["task_objects"]["pinch"]
    hx, hy, hz = map(float, cfg["size"])
    _, _, z = map(float, cfg["pos"])

    # MuJoCo box size is half extent: v0.4.6 uses a 90 x 30 x 140 mm block.
    assert 2 * hx >= 0.090
    assert 2 * hy >= 0.030
    assert 2 * hz >= 0.140
    assert abs((z - hz) - 0.070) < 1e-9


def test_pinch_approach_holds_object_until_bilateral_tip_contact():
    main_src = Path("src/main.py").read_text(encoding="utf-8")
    sim_src = Path("src/simulation/mujoco_env.py").read_text(encoding="utf-8")

    assert 'hold_pinch_object = bool(' in main_src
    assert 'last_contact.get("thumb_tip_contact", False)' in main_src
    assert 'last_contact.get("index_tip_contact", False)' in main_src
    assert 'simulator.step(q_out, hold_object=hold_pinch_object)' in main_src
    assert 'def step(self, target_q: np.ndarray, *, hold_object: bool = False)' in sim_src
    assert 'def _hold_task_object_pose' in sim_src
