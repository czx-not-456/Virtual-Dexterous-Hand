from src.common import load_yaml


def test_pinch_object_is_placed_in_thumb_index_workspace():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    spec = cfg["task_objects"]["pinch"]
    sx, sy, sz = [float(x) for x in spec["size"]]
    x, y, z = [float(x) for x in spec["pos"]]

    # task_table top = 0.035 + 0.035 = 0.070 m in the current MJCF.
    table_top = 0.070
    assert abs((z - sz) - table_top) < 0.002
    assert 0.178 <= z + sz <= 0.186
    assert -0.020 <= x <= -0.008
    assert -0.050 <= y <= -0.038
    assert sx <= 0.008 and sy <= 0.008


def test_wrap_cylinder_spans_thumb_and_finger_workspace():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    spec = cfg["task_objects"]["wrap"]
    radius, half_height = [float(x) for x in spec["size"]]
    x, y, z = [float(x) for x in spec["pos"]]

    table_top = 0.070
    assert abs((z - half_height) - table_top) < 0.002
    assert z + half_height >= 0.198
    # The cylinder should extend to the thumb side (about y=-0.05)
    # and also into the curled non-thumb finger workspace (around y=0.00).
    assert y - radius <= -0.050
    assert y + radius >= 0.010
    assert -0.012 <= x <= 0.002
