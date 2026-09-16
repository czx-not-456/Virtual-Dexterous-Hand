from src.common import load_yaml


def test_pinch_object_is_placed_in_thumb_index_workspace():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    spec = cfg["task_objects"]["pinch"]
    sx, sy, sz = [float(x) for x in spec["size"]]
    x, y, z = [float(x) for x in spec["pos"]]

    # task_table top = 0.035 + 0.035 = 0.070 m in the current MJCF.
    table_top = 0.070
    assert abs((z - sz) - table_top) < 0.002
    assert 0.208 <= z + sz <= 0.212
    assert -0.020 <= x <= -0.008
    assert -0.050 <= y <= -0.038
    # v0.4.6 keeps the wide/thick calibration target and raises the top face
    # just enough to intersect the one-tendon index fingertip workspace.
    assert sx <= 0.050 and sy <= 0.020
    assert sx > sy  # wide support footprint, still thinner on the pinch axis


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


def test_v040_standardized_task_catalog_is_well_formed():
    cfg = load_yaml("configs/simulation.yaml")["mujoco"]
    tasks = cfg["task_objects"]
    assert {"wrap", "pinch", "sphere", "card", "bottle", "box"}.issubset(tasks)
    for name, spec in tasks.items():
        assert spec["interaction"] in {"pinch", "wrap", "sphere"}
        assert spec["active_intent"] in {"PINCH", "WRAP"}
        assert spec["type"] in {"box", "cylinder", "sphere"}
        assert len(spec["pos"]) == 3
        assert all(float(x) > 0 for x in spec["size"])


def test_v040_pinch_posture_prior_is_near_object_workspace():
    import math
    import numpy as np
    from src.hand_model.robot_hand import RobotHandModel

    algo = load_yaml("configs/algorithm.yaml")
    sim = load_yaml("configs/simulation.yaml")["mujoco"]
    prior = algo["task_space_servo"]["pinch"]["posture_prior_deg"]
    robot = RobotHandModel()
    idx = {name: i for i, name in enumerate(robot.joint_order)}
    q = np.zeros(len(robot.joint_order), dtype=float)
    for joint, deg in prior.items():
        q[idx[joint]] = math.radians(float(deg))
    tips = robot.fingertips(q)
    palm_world = np.array([0.0, 0.0, 0.240])
    obj = np.array(sim["task_objects"]["pinch"]["pos"], dtype=float)
    # The prior only needs to pre-position the two digits into the local task
    # workspace; final contact is closed by MuJoCo Jacobian servo.
    assert np.linalg.norm((tips["thumb"] + palm_world) - obj) < 0.075
    assert np.linalg.norm((tips["index"] + palm_world) - obj) < 0.075
