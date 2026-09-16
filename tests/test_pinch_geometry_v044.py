from src.common import load_yaml


def test_pinch_calibration_block_overlaps_precision_workspace():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]["task_objects"]["pinch"]
    hx, hy, hz = map(float, cfg["size"])
    x, y, z = map(float, cfg["pos"])

    # MuJoCo box size uses half extents. Preserve table support.
    assert abs((z - hz) - 0.070) < 1e-9
    # The precision pinch centerline of this model is ~0.190-0.193 m.
    assert z + hz >= 0.200
    # Keep a forgiving calibration target in X and a 20 mm physical Y thickness.
    assert 2 * hx >= 0.070
    assert 2 * hy >= 0.020
